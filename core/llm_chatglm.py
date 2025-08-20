import asyncio
import os
import sys
from threading import Thread
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import torch
from langchain_core.runnables import Runnable
from requests.exceptions import ChunkedEncodingError, ConnectionError
from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM,
                          AutoTokenizer, GenerationConfig,
                          TextIteratorStreamer)

from config.logger_config import logger


class ChatGLMLLM(Runnable):
    def __init__(self,
                 model_name_cuda: str = "THUDM/glm-4-9b-chat",  # 使用官方确认的模型名称
                 model_name_cpu: str = "Qwen/Qwen2.5-0.5B-Instruct",
                 model_path_cuda: Optional[str] = None,  # 支持本地路径
                 model_path_cpu: Optional[str] = None,   # 支持本地路径
                 revision: str = "main",
                 max_new_tokens: int = 1024,
                 use_modelscope: bool = False):  # 是否使用ModelScope
        """
        初始化大语言模型

        Args:
            model_name_cuda: CUDA设备下HF模型名称
            model_name_cpu: CPU设备下HF模型名称
            model_path_cuda: CUDA设备下本地模型路径（优先级高于model_name_cuda）
            model_path_cpu: CPU设备下本地模型路径（优先级高于model_name_cpu）
            revision: 模型版本
            max_new_tokens: 最大生成token数
            use_modelscope: 是否使用ModelScope（针对国内网络优化）
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_modelscope = True if not torch.cuda.is_available() else False
        # 确定最终模型路径/名称
        if self.device == "cuda" and model_path_cuda:
            self.model_name_or_path = model_path_cuda
            logger.info(
                f"Using local CUDA model path: {self.model_name_or_path}")
        elif self.device == "cpu" and model_path_cpu:
            self.model_name_or_path = model_path_cpu
            logger.info(
                f"Using local CPU model path: {self.model_name_or_path}")
        else:
            self.model_name_or_path = model_name_cuda if self.device == "cuda" else model_name_cpu
            logger.info(f"Using HF model name: {self.model_name_or_path}")

        # 设置HF端点（国内镜像）
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

        # 如果使用 ModelScope，尝试导入并设置
        self.use_modelscope = use_modelscope
        if self.use_modelscope:
            try:
                import modelscope
                from modelscope import snapshot_download
                self.snapshot_download = snapshot_download
                logger.info("ModelScope initialized successfully.")
                # 如果是HF名称，且非本地路径，则尝试通过ModelScope下载
                if not os.path.exists(self.model_name_or_path):
                    try:
                        self.model_name_or_path = self.snapshot_download(
                            self.model_name_or_path, revision=revision)
                        logger.info(
                            f"Model downloaded via ModelScope to: {self.model_name_or_path}")
                    except Exception as dl_e:
                        logger.warning(
                            f"ModelScope download failed: {dl_e}. Falling back to HF.")
            except ImportError:
                logger.warning(
                    "ModelScope not installed. Please install with 'pip install modelscope' for better experience in China.")
                self.use_modelscope = False

        # 获取模型配置
        try:
            config = AutoConfig.from_pretrained(
                self.model_name_or_path,
                revision=revision,
                trust_remote_code=True
            )
            model_max_length = getattr(config, "model_max_length",
                                       getattr(config, "max_position_embeddings",
                                               getattr(config, "n_positions", 8192)))
        except Exception as e:
            logger.warning(
                f"Could not load config, using default length (8192): {e}")
            model_max_length = 8192

        self.max_new_tokens = max_new_tokens
        self.max_total_tokens = model_max_length - self.max_new_tokens

        logger.info(f'Device: {self.device}')
        logger.info(f'Model source: {self.model_name_or_path}')
        logger.info(
            f'Max context: {model_max_length}, Max new tokens: {self.max_new_tokens}, Max input: {self.max_total_tokens}')

        # 加载模型和分词器
        try:
            self.load_model_and_tokenizer(revision)
        except (ConnectionError, ChunkedEncodingError) as conn_e:
            logger.error(
                f"Network error during loading: {conn_e}. This might be due to network issues to Hugging Face.")
            logger.info(
                "You can try: 1) Use a VPN; 2) Set use_modelscope=True; 3) Manually download the model and provide local path.")
            raise RuntimeError(
                f"Model loading failed due to network issues: {str(conn_e)}")
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise RuntimeError(f"Model initialization failed: {str(e)}")

    def load_model_and_tokenizer(self, revision: str):
        """加载分词器和模型"""
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            revision=revision
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(
                f"Set pad_token to eos_token: {self.tokenizer.pad_token}")

        # 判断模型类型
        self.is_chatglm = "glm" in self.model_name_or_path.lower()
        self.is_qwen = "qwen" in self.model_name_or_path.lower()

        # 动态计算所需的 torch_dtype
        if self.device == "cuda":
            # 优先使用 bfloat16 以节省显存，不支持则 fallback 到 float16
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            torch_dtype = torch.float32  # CPU 上使用 float32

        # 加载模型
        model_class = AutoModel if self.is_chatglm else AutoModelForCausalLM
        self.model = model_class.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            revision=revision,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True  # 优化CPU内存使用
        )

        # 设置生成配置
        self.generation_config = GenerationConfig.from_model_config(
            self.model.config)
        self.generation_config.max_new_tokens = self.max_new_tokens
        self.generation_config.do_sample = False
        self.generation_config.repetition_penalty = 1.1
        self.generation_config.pad_token_id = self.tokenizer.pad_token_id

        self.model.eval()
        self._history = []
        logger.info(
            f"Model loaded successfully. Type: {'ChatGLM' if self.is_chatglm else 'Qwen/Other'}")

    def truncate_history(self, history: List[Tuple[str, str]], max_tokens: int) -> List[Tuple[str, str]]:
        """截断 history，保证总 token 数不超 max_tokens"""
        if max_tokens <= 0:
            return []

        total_tokens = 0
        truncated = []
        # 逆序保留最近对话（最新的对话更重要）
        for q, a in reversed(history):
            # 使用更精确的token计数方法
            q_tokens = self.tokenizer.encode(q, add_special_tokens=False)
            a_tokens = self.tokenizer.encode(a, add_special_tokens=False)
            tokens_count = len(q_tokens) + len(a_tokens)

            if total_tokens + tokens_count > max_tokens:
                break
            truncated.insert(0, (q, a))
            total_tokens += tokens_count

        logger.debug(
            f"Truncated history from {len(history)} to {len(truncated)} items, tokens: {total_tokens}")
        return truncated

    def convert_history(self, history: List[Tuple[str, str]]) -> List[Dict[str, str]]:
        """
        将 List[Tuple[str, str]] 转换成 ChatGLM 需要的 List[Dict[str, str]] 格式
        """
        converted = []
        for question, answer in history:
            converted.append({"role": "user", "content": question})
            converted.append({"role": "assistant", "content": answer})
        return converted

    def truncate_text(self, text: str, max_tokens: int) -> str:
        """截断文本到指定token数量"""
        if not text or max_tokens <= 0:
            return ""

        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            truncated_text = self.tokenizer.decode(
                tokens, skip_special_tokens=True)
            logger.debug(
                f"Truncated text from {len(tokens)} to {max_tokens} tokens")
            return truncated_text
        return text

    def build_prompt(self, query: str, context: str, history: List[Tuple[str, str]]) -> str:
        """构建适合不同模型的提示词"""
        try:
            if self.is_chatglm:
                # ChatGLM格式
                system_prompt = "你是一个智能助手，请基于提供的文档内容，并准确回答用户问题。\n- 如果文档中找不到答案，请明确说无法找到。\n- 回答要简洁，不要编造信息。"

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"文档内容：\n{context}\n\n问题：{query}"}
                ]

                # 添加历史对话
                history_messages = self.convert_history(history)
                messages = history_messages + messages

                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            elif self.is_qwen:
                # Qwen格式
                system_prompt = "你是一个智能助手，请基于提供的文档内容，参考历史对话，并准确回答用户问题。\n- 如果文档中找不到答案，请明确说无法找到。\n- 回答要简洁，不要编造信息。"

                messages = [{"role": "system", "content": system_prompt}]

                # 添加历史对话
                for q, a in history:
                    messages.append({"role": "user", "content": q})
                    messages.append({"role": "assistant", "content": a})

                # 添加当前问题和上下文
                messages.append(
                    {"role": "user", "content": f"文档内容：\n{context}\n\n问题：{query}"})

                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # 通用格式
                return f"""你是一个智能助手，请基于提供的文档内容，参考历史对话，并准确回答用户问题。
- 如果文档中找不到答案，请明确说无法找到。
- 回答要简洁，不要编造信息。

文档内容：
{context}

历史对话：
{history}

当前问题：
{query}

请给出答案："""
        except Exception as e:
            logger.warning(
                f"Failed to build prompt with template, using fallback: {e}")
            # 回退到简单格式
            return f"基于以下内容回答问题：\n文档：{context}\n问题：{query}"

    def invoke(self, input: Any, config: Optional[dict] = None, **kwargs) -> Any:
        if not isinstance(config, dict):
            config = {}

        # 解析输入
        if isinstance(input, str):
            query = input
            history = []
            context = ""
        elif isinstance(input, dict):
            query = input.get("query", "")
            history = input.get("history", [])
            context = input.get("context", "")
        else:
            query = str(input)
            history = []
            context = ""

        question = query

        try:
            # 计算可用tokens，减去新生成tokens
            max_input_tokens = self.max_total_tokens

            # 预留生成空间
            reserved_for_answer = int(self.max_new_tokens * 1.2)
            available_tokens = max_input_tokens - reserved_for_answer

            # 为 context、history 和 query 分配 token
            max_context_tokens = int(available_tokens * 0.5)  # 50% 给 context
            max_history_tokens = int(available_tokens * 0.3)  # 30% 给 history
            max_query_tokens = available_tokens - max_context_tokens - max_history_tokens

            # 截断 context / query
            context = self.truncate_text(context, max_context_tokens)
            query = self.truncate_text(query, max_query_tokens)

            # 截断历史对话
            formatted_history = self.truncate_history(
                history, max_history_tokens)

            if self.is_chatglm:
                # ChatGLM 专用处理
                full_query = self.build_prompt(
                    query, context, formatted_history)

                logger.debug(f"ChatGLM模型输入: {full_query}")

                # 使用模型chat方法
                inputs = self.tokenizer(query, return_tensors="pt")
                outputs = self.model.generate(**inputs)
                result = self.tokenizer.decode(
                    outputs[0], skip_special_tokens=True)

                if isinstance(result, tuple) and len(result) == 2:
                    response, _ = result
                else:
                    response = result

                self._history.append((question, response))
                logger.info(f"ChatGLM模型回复: {response}")
                return response

            else:
                # 其他模型处理（包括Qwen）
                prompt = self.build_prompt(query, context, formatted_history)
                logger.debug(f"模型输入: {prompt}")

                inputs = self.tokenizer(
                    prompt,
                    truncation=True,
                    max_length=self.max_total_tokens,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)

                # 使用生成配置
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id
                )

                response = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[-1]:],
                    skip_special_tokens=True
                ).strip()

                self._history.append((question, response))
                logger.info(f"模型回复: {response}")
                return response

        except Exception as e:
            logger.error(f"invoke 模型调用失败: {e}, query: {query}", exc_info=True)
            raise RuntimeError(f"处理问题失败: {str(e)}")

    def get_history(self) -> List[Tuple[str, str]]:
        return self._history

    def clear_history(self):
        """清空对话历史"""
        self._history = []
        logger.info("对话历史已清空")

    def stream(self, input: Any, config: Optional[dict] = None, **kwargs) -> Generator[str, None, None]:
        """流式生成响应"""
        if isinstance(input, str):
            query = input
            history = []
            context = ""
        elif isinstance(input, dict):
            query = input.get("query", "")
            history = input.get("history", [])
            context = input.get("context", "")
        else:
            query = str(input)
            history = []
            context = ""

        question = query

        try:
            # 计算可用tokens
            max_input_tokens = self.max_total_tokens
            reserved_for_answer = int(self.max_new_tokens * 1.2)
            available_tokens = max_input_tokens - reserved_for_answer

            max_context_tokens = int(available_tokens * 0.5)
            max_history_tokens = int(available_tokens * 0.3)
            max_query_tokens = available_tokens - max_context_tokens - max_history_tokens

            # 截断内容
            context = self.truncate_text(context, max_context_tokens)
            query = self.truncate_text(query, max_query_tokens)
            safe_history = self.truncate_history(history, max_history_tokens)

            if self.is_chatglm:
                # ChatGLM流式处理
                full_query = self.build_prompt(query, context, safe_history)

                partial_response = ""
                for response, _ in self.model.stream_chat(
                    self.tokenizer,
                    full_query,
                    history=self.convert_history(safe_history),
                    max_length=self.max_total_tokens
                ):
                    partial_response += response
                    yield response

                self._history.append((question, partial_response))
                logger.info(f"ChatGLM模型（流式）回复完成")

            else:
                # 其他模型流式处理（使用TextIteratorStreamer）
                prompt = self.build_prompt(query, context, safe_history)
                inputs = self.tokenizer(
                    prompt,
                    truncation=True,
                    max_length=self.max_total_tokens,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)

                # 创建流式生成器
                streamer = TextIteratorStreamer(
                    self.tokenizer,
                    skip_prompt=True,
                    skip_special_tokens=True
                )

                generation_kwargs = dict(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    streamer=streamer
                )

                # 在单独线程中生成
                thread = Thread(target=self.model.generate,
                                kwargs=generation_kwargs)
                thread.start()

                # 流式输出
                partial_response = ""
                for new_text in streamer:
                    partial_response += new_text
                    yield new_text

                self._history.append((question, partial_response))
                logger.info(f"模型（流式）回复完成")

        except Exception as e:
            logger.error(
                f"stream 模型流式调用失败: {e}, query: {query}", exc_info=True)
            raise RuntimeError(f"流式处理失败: {str(e)}")

    async def astream(self, input: Any, config: Optional[dict] = None, **kwargs) -> Generator[str, None, None]:
        """异步流式生成"""
        loop = asyncio.get_event_loop()
        # 将同步的stream生成器转换为异步

        def _stream_wrapper():
            return self.stream(input, config=config, **kwargs)

        stream = await loop.run_in_executor(None, _stream_wrapper)
        for chunk in stream:
            yield chunk
