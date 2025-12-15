import os
import re
import asyncio
import torch
import threading
import signal
from typing import Any, Dict, List, Tuple, Optional, Generator, AsyncGenerator
from threading import Thread, Semaphore
from contextlib import contextmanager, asynccontextmanager
from pydantic import Field, PrivateAttr
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    TextIteratorStreamer,
)
from langchain_core.language_models.llms import BaseLLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.outputs import LLMResult, Generation
from config.logger_config import logger


class StableLLM(BaseLLM):
    """
    稳定版 LLM 封装
    - 支持 ChatGLM / Qwen2.5 / Qwen3
    - 面向 RAG / 文档问答
    - 标准 LangChain LLM 接口
    """

    model_name_cuda: str = Field(default="THUDM/glm-4-9b-chat")
    model_name_cpu: str = Field(default="Qwen/Qwen2.5-0.5B-Instruct")
    model_path_cuda: Optional[str] = Field(default=None)
    model_path_cpu: Optional[str] = Field(default=None)
    max_new_tokens: int = Field(default=512)
    temperature: float = Field(default=0.3)
    top_p: float = Field(default=0.8)
    max_concurrent_requests: int = Field(default=3, description="最大并发请求数")
    request_timeout: float = Field(default=120.0, description="请求超时时间（秒）")

    # 使用 PrivateAttr 存储非 Pydantic 字段
    _device: str = PrivateAttr()
    _model_path: str = PrivateAttr()
    _history: List[Tuple[str, str]] = PrivateAttr(default_factory=list)
    _tokenizer: Any = PrivateAttr()
    _model: Any = PrivateAttr()
    _generation_config: Any = PrivateAttr()
    _sync_semaphore: Semaphore = PrivateAttr()
    _async_semaphore: asyncio.Semaphore = PrivateAttr()

    def __init__(
        self,
        model_name_cuda: str = "THUDM/glm-4-9b-chat",
        model_name_cpu: str = "Qwen/Qwen2.5-0.5B-Instruct",
        model_path_cuda: Optional[str] = None,
        model_path_cpu: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.8,
        max_concurrent_requests: int = 3,
        request_timeout: float = 120.0,
        **kwargs
    ):
        super().__init__(
            model_name_cuda=model_name_cuda,
            model_name_cpu=model_name_cpu,
            model_path_cuda=model_path_cuda,
            model_path_cpu=model_path_cpu,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            max_concurrent_requests=max_concurrent_requests,
            request_timeout=request_timeout,
            **kwargs
        )
        
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_new_tokens = min(self.max_new_tokens, 768)
        
        # 初始化并发限流器
        self._sync_semaphore = Semaphore(self.max_concurrent_requests)
        # 异步信号量需要在事件循环中创建，延迟初始化
        self._async_semaphore = None

        # 选择模型路径
        if self._device == "cuda" and self.model_path_cuda:
            self._model_path = self.model_path_cuda
        elif self._device == "cpu" and self.model_path_cpu:
            self._model_path = self.model_path_cpu
        else:
            self._model_path = self.model_name_cuda if self._device == "cuda" else self.model_name_cpu

        logger.info(f"Using model: {self._model_path} on {self._device}")

        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

        self._load_model()

    # -------------------------
    # LangChain BaseLLM 接口
    # -------------------------
    @property
    def _llm_type(self) -> str:
        """返回 LLM 类型标识"""
        if "qwen" in self._model_path.lower():
            return "qwen"
        elif "glm" in self._model_path.lower():
            return "chatglm"
        return "stable_llm"

    @property
    def model_path(self) -> str:
        """返回模型路径（向后兼容）"""
        return self._model_path

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """LangChain 标准接口：同步调用"""
        # 从 prompt 中提取信息（如果是格式化后的字符串）
        # 简单处理：直接作为 query
        query = prompt.strip()
        context = kwargs.get("context", "")
        history = kwargs.get("history", [])

        return self.invoke({
            "query": query,
            "context": context,
            "history": history
        })

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """LangChain 标准接口：批量生成"""
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)

    # -------------------------
    # 模型加载
    # -------------------------
    def _load_model(self):
        # 加载 tokenizer - 添加回退机制
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_path,
                trust_remote_code=True,
                use_fast=True,
            )
        except Exception as fast_error:
            logger.warning(f"Failed to load fast tokenizer: {fast_error}")
            logger.info("Falling back to slow tokenizer (use_fast=False)")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_path,
                trust_remote_code=True,
                use_fast=False,
            )

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        torch_dtype = torch.float16 if self._device == "cuda" else torch.float32

        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            trust_remote_code=True,
            dtype=torch_dtype,
            device_map="auto" if self._device == "cuda" else None,
            low_cpu_mem_usage=True,
        )

        self._model.eval()

        self._generation_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            do_sample=False,          # RAG 场景强烈推荐 False
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=1.1,
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
        )

        logger.info("Model loaded successfully")

    # -------------------------
    # 并发限流和超时保护
    # -------------------------
    @contextmanager
    def _rate_limit_context(self):
        """同步并发限流上下文管理器"""
        acquired = False
        try:
            acquired = self._sync_semaphore.acquire(timeout=30.0)
            if not acquired:
                raise RuntimeError("请求限流：无法获取处理槽位，请稍后重试")
            yield
        finally:
            if acquired:
                self._sync_semaphore.release()

    @asynccontextmanager
    async def _async_rate_limit_context(self):
        """异步并发限流上下文管理器"""
        if self._async_semaphore is None:
            # 延迟初始化异步信号量
            try:
                loop = asyncio.get_event_loop()
                self._async_semaphore = asyncio.Semaphore(self.max_concurrent_requests)
            except RuntimeError:
                # 如果没有事件循环，创建一个新的
                self._async_semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        
        acquired = False
        try:
            await asyncio.wait_for(
                self._async_semaphore.acquire(),
                timeout=30.0
            )
            acquired = True
            yield
        except asyncio.TimeoutError:
            raise RuntimeError("请求限流：无法获取处理槽位，请稍后重试")
        finally:
            if acquired:
                self._async_semaphore.release()

    def _execute_with_timeout(self, func, *args, **kwargs):
        """同步执行函数，带超时保护"""
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                result = future.result(timeout=self.request_timeout)
                return result
            except concurrent.futures.TimeoutError:
                future.cancel()
                raise TimeoutError(f"请求超时：超过 {self.request_timeout} 秒未完成")

    # -------------------------
    # 构建 messages（压缩 30% token）
    # -------------------------
    def build_messages(
        self,
        query: str,
        context: str,
        history: List[Tuple[str, str]],
    ) -> List[Dict[str, str]]:
        # 压缩系统提示词（减少约 30% token）
        messages = [
            {
                "role": "system",
                "content": "基于文档回答问题。要求：1.直接结论 2.严格基于文档 3.无信息则说明未找到 4.无客套话"
            }
        ]

        # 压缩历史对话格式
        for q, a in history:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})

        # 压缩上下文格式（减少约 30% token）
        if context:
            user_content = f"文档：{context}\n问题：{query}"
        else:
            user_content = query

        messages.append({"role": "user", "content": user_content})
        return messages

    # -------------------------
    # 历史截断（token 级）
    # -------------------------
    def truncate_history(
        self,
        history: List[Tuple[str, str]],
        max_tokens: int,
    ) -> List[Tuple[str, str]]:
        total = 0
        result = []

        for q, a in reversed(history):
            tokens = len(self._tokenizer.encode(q, add_special_tokens=False)) + len(self._tokenizer.encode(a, add_special_tokens=False))
            if total + tokens > max_tokens:
                break
            result.insert(0, (q, a))
            total += tokens

        return result

    # -------------------------
    # invoke（非流式）- 带并发限流和超时保护
    # -------------------------
    def invoke(self, input: Any, config: Optional[dict] = None) -> str:
        """调用模型生成回复，带并发限流和超时保护"""
        def _invoke_internal():
            if isinstance(input, dict):
                query = input.get("query", "")
                context = input.get("context", "")
                history = input.get("history", [])
            else:
                query = str(input)
                context = ""
                history = []

            max_context_tokens = 2048
            max_history_tokens = 1024

            context = self._truncate_text(context, max_context_tokens)
            history = self.truncate_history(history, max_history_tokens)

            messages = self.build_messages(query, context, history)

            # 安全调用 chat template
            if hasattr(self._tokenizer, "apply_chat_template"):
                text = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                text = messages[-1]["content"]

            inputs = self._tokenizer(text, return_tensors="pt").to(self._device)

            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    generation_config=self._generation_config,
                )

            output = self._tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            output = self._clean_response(output)
            self._history.append((query, output))
            return output

        # 应用并发限流和超时保护
        with self._rate_limit_context():
            try:
                return self._execute_with_timeout(_invoke_internal)
            except TimeoutError as e:
                logger.error(f"请求超时: {e}")
                raise RuntimeError(str(e))
            except Exception as e:
                logger.error(f"请求处理失败: {e}")
                raise

    # -------------------------
    # Stream（同步流式）- 带并发限流
    # -------------------------
    def stream(self, input: Any, config: Optional[dict] = None) -> Generator[str, None, None]:
        """同步流式生成，带并发限流"""
        with self._rate_limit_context():
            if isinstance(input, dict):
                query = input.get("query", "")
                context = input.get("context", "")
                history = input.get("history", [])
            else:
                query = str(input)
                context = ""
                history = []

            max_context_tokens = 2048
            max_history_tokens = 1024

            context = self._truncate_text(context, max_context_tokens)
            history = self.truncate_history(history, max_history_tokens)

            messages = self.build_messages(query, context, history)

            if hasattr(self._tokenizer, "apply_chat_template"):
                text = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                text = messages[-1]["content"]

            model_inputs = self._tokenizer([text], return_tensors="pt").to(self._device)

            # 创建流式生成器，设置超时
            streamer = TextIteratorStreamer(
                self._tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
                timeout=self.request_timeout
            )

            generation_kwargs = dict(
                **model_inputs,
                generation_config=self._generation_config,
                streamer=streamer
            )

            # 在单独线程中生成
            thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
            thread.start()

            # 流式输出，带超时检查
            import time
            start_time = time.time()
            for new_text in streamer:
                # 检查超时
                if time.time() - start_time > self.request_timeout:
                    logger.warning(f"流式生成超时: 超过 {self.request_timeout} 秒")
                    break
                yield new_text

    # -------------------------
    # astream（异步流式）- 带并发限流和超时保护
    # -------------------------
    async def astream(self, input: Any, config: Optional[dict] = None) -> AsyncGenerator[str, None]:
        """异步流式生成（SSE 兼容），带并发限流和超时保护"""
        async with self._async_rate_limit_context():
            try:
                loop = asyncio.get_event_loop()

                def _stream_wrapper():
                    return self.stream(input, config=config)

                # 在线程池中执行同步流式生成，带超时
                try:
                    stream = await asyncio.wait_for(
                        loop.run_in_executor(None, _stream_wrapper),
                        timeout=self.request_timeout
                    )
                    
                    for chunk in stream:
                        yield chunk
                except asyncio.TimeoutError:
                    logger.error(f"异步流式生成超时: 超过 {self.request_timeout} 秒")
                    yield "[ERROR] 请求超时，请稍后重试"
            except RuntimeError as e:
                logger.error(f"异步流式生成限流: {e}")
                yield f"[ERROR] {str(e)}"
            except Exception as e:
                logger.error(f"异步流式生成失败: {e}")
                yield f"[ERROR] 处理失败: {str(e)}"

    # -------------------------
    # 工具函数
    # -------------------------
    def _truncate_text(self, text: str, max_tokens: int) -> str:
        tokens = self._tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= max_tokens:
            return text
        return self._tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)

    def _clean_response(self, text: str) -> str:
        text = re.sub(r"\n+", "\n", text).strip()
        patterns = [
            r"如果还有其他问题.*$",
            r"希望以上信息.*$",
            r"感谢.*提问.*$",
        ]
        for p in patterns:
            text = re.sub(p, "", text, flags=re.IGNORECASE)
        return text.strip()


# -------------------------
# LLMFactory：自动选择模型
# -------------------------
class LLMFactory:
    """LLM 工厂类：根据模型名称自动选择 Qwen / GLM"""

    @staticmethod
    def create_llm(
        model_name: Optional[str] = None,
        model_name_cuda: str = "THUDM/glm-4-9b-chat",
        model_name_cpu: str = "Qwen/Qwen2.5-0.5B-Instruct",
        max_concurrent_requests: int = 3,
        request_timeout: float = 120.0,
        **kwargs
    ) -> StableLLM:
        """
        创建 LLM 实例，自动识别模型类型
        
        Args:
            model_name: 指定模型名称（可选，会覆盖默认值）
            model_name_cuda: CUDA 默认模型
            model_name_cpu: CPU 默认模型
            **kwargs: 其他参数传递给 StableLLM
        
        Returns:
            StableLLM 实例
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 如果指定了 model_name，根据设备选择
        if model_name:
            if device == "cuda":
                model_name_cuda = model_name
            else:
                model_name_cpu = model_name
        
        # 自动识别模型类型并设置默认值
        default_cuda = model_name_cuda
        default_cpu = model_name_cpu
        
        # 如果模型名称包含 qwen，优先使用 Qwen 系列
        if model_name and "qwen" in model_name.lower():
            if device == "cuda":
                default_cuda = model_name
            else:
                default_cpu = model_name
        # 如果模型名称包含 glm，优先使用 GLM 系列
        elif model_name and "glm" in model_name.lower():
            if device == "cuda":
                default_cuda = model_name
            else:
                default_cpu = model_name
        
        logger.info(f"LLMFactory: Creating LLM for {device}, model_cuda={default_cuda}, model_cpu={default_cpu}")
        
        return StableLLM(
            model_name_cuda=default_cuda,
            model_name_cpu=default_cpu,
            max_concurrent_requests=max_concurrent_requests,
            request_timeout=request_timeout,
            **kwargs
        )

    @staticmethod
    def create_qwen_llm(**kwargs) -> StableLLM:
        """创建 Qwen 系列 LLM"""
        return LLMFactory.create_llm(
            model_name_cuda="Qwen/Qwen2.5-7B-Instruct",
            model_name_cpu="Qwen/Qwen2.5-0.5B-Instruct",
            max_concurrent_requests=5,
            request_timeout=60.0,
            **kwargs
        )

    @staticmethod
    def create_glm_llm(**kwargs) -> StableLLM:
        """创建 GLM 系列 LLM"""
        return LLMFactory.create_llm(
            model_name_cuda="THUDM/glm-4-9b-chat",
            model_name_cpu="THUDM/glm-4-9b-chat",  # GLM 主要支持 CUDA
            max_concurrent_requests=5,
            request_timeout=60.0,
            **kwargs
        )

    @staticmethod
    def create_qwen3_llm(**kwargs) -> StableLLM:
        """创建 Qwen3 系列 LLM"""
        return LLMFactory.create_llm(
            model_name_cuda="Qwen/Qwen3-7B-Instruct",
            model_name_cpu="Qwen/Qwen3-0.6B",
            max_concurrent_requests=5,
            request_timeout=60.0,
            **kwargs
        )

# 为了向后兼容，保留别名
ChatGLMLLM = LLMFactory.create_qwen3_llm()
