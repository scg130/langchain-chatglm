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
                 model_name_cuda: str = "THUDM/glm-4-9b-chat",
                 model_name_cpu: str = "Qwen/Qwen2.5-0.5B-Instruct",
                 model_path_cuda: Optional[str] = None,
                 model_path_cpu: Optional[str] = None,
                 revision: str = "main",
                 max_new_tokens: int = 1024,
                 use_modelscope: bool = False):
        """
        初始化大语言模型（支持 ChatGLM-4 和 Qwen）
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

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

        # ModelScope 处理
        self.use_modelscope = use_modelscope
        if self.use_modelscope:
            try:
                import modelscope
                from modelscope import snapshot_download
                self.snapshot_download = snapshot_download
                logger.info("ModelScope initialized successfully.")
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
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            torch_dtype = torch.float32

        # 加载模型 - 对于 ChatGLM-4 使用 AutoModelForCausalLM
        if self.is_chatglm:
            model_class = AutoModelForCausalLM  # ChatGLM-4 使用 CausalLM
        else:
            model_class = AutoModelForCausalLM  # Qwen 也使用 CausalLM

        self.model = model_class.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            revision=revision,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True
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
            f"Model loaded successfully. Type: {'ChatGLM-4' if self.is_chatglm else 'Qwen/Other'}")

    def truncate_history(self, history: List[Tuple[str, str]], max_tokens: int) -> List[Tuple[str, str]]:
        """截断 history，保证总 token 数不超 max_tokens"""
        if max_tokens <= 0:
            return []

        total_tokens = 0
        truncated = []
        for q, a in reversed(history):
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

    def convert_history_to_messages(self, history: List[Tuple[str, str]]) -> List[Dict[str, str]]:
        """将历史对话转换为消息格式"""
        messages = []
        for question, answer in history:
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer})
        return messages

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

    def build_messages(self, query: str, context: str, history: List[Tuple[str, str]]) -> List[Dict[str, str]]:
        """构建消息列表（适用于 ChatGLM-4 和 Qwen）"""
        messages = []

        # 系统提示词
        system_prompt = "你是一个智能助手，请基于提供的文档内容，并准确回答用户问题。\n- 如果文档中找不到答案，请明确说无法找到。\n- 回答要简洁，不要编造信息。"
        messages.append({"role": "system", "content": system_prompt})

        # 添加历史对话
        history_messages = self.convert_history_to_messages(history)
        messages.extend(history_messages)

        # 添加当前查询和上下文
        user_content = f"文档内容：\n{context}\n\n问题：{query}" if context else query
        messages.append({"role": "user", "content": user_content})

        return messages

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

            # 构建消息
            messages = self.build_messages(query, context, safe_history)

            # 应用聊天模板
            if hasattr(self.tokenizer, 'apply_chat_template'):
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # 回退到简单格式
                prompt = f"{messages[-1]['content']}\n\n请回答："

            logger.debug(f"模型输入: {prompt}")

            # 编码输入
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_total_tokens
            ).to(self.device)

            # 生成回复
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.7,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

            # 解码输出（跳过输入部分）
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

            # 构建消息
            messages = self.build_messages(query, context, safe_history)

            # 应用聊天模板
            if hasattr(self.tokenizer, 'apply_chat_template'):
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                prompt = f"{messages[-1]['content']}\n\n请回答："

            # 编码输入
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_total_tokens
            ).to(self.device)

            # 创建流式生成器
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
                timeout=60.0
            )

            generation_kwargs = dict(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
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

        def _stream_wrapper():
            return self.stream(input, config=config, **kwargs)

        stream = await loop.run_in_executor(None, _stream_wrapper)
        for chunk in stream:
            yield chunk
