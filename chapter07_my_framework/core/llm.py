"""统一大模型客户端 (LLM Client) 接口实现。

基于 Google Cloud Vertex AI (google-genai SDK + ADC 凭证) 构建。
对外提供标准化的 invoke、stream_invoke 和 think 方法，
对内负责消息格式转换、参数校验、流式聚合与异常捕获。
"""

from typing import Any, Iterator, List, Optional

from google import genai
from google.genai import errors, types

from chapter07_my_framework.core.config import Config
from chapter07_my_framework.core.exceptions import LLMCallException
from chapter07_my_framework.core.message import Message


class HelloAgentsLLM:
    """HelloAgents 框架的统一大模型客户端。

        封装与 Google Cloud Vertex AI 的底层通信。
    """

    def __init__(self, config: Optional[Config] = None, **kwargs: Any) -> None:
        # 1. 加载并校验配置
        self.config = config or Config.from_env(**kwargs)
        self.config.validate_for_vertex()

        self.model = kwargs.get("model", self.config.model)
        self.temperature = kwargs.get("temperature", self.config.temperature)
        self.timeout = kwargs.get("timeout", self.config.timeout)

        try:
            self._client = genai.Client(
                vertexai=True,
                project=self.config.google_cloud_project,
                location=self.config.google_cloud_location,
            )
        except Exception as e:
            raise LLMCallException(
                f"初始化 Vertex AI Client 失败，请检查 GCP 凭证 (ADC): {e}",
                original_error=e,
            ) from e
        if self.config.debug:
            print(
                f"🔧 [HelloAgentsLLM] 成功初始化 Vertex AI 客户端: model={self.model}, location={self.config.google_cloud_location}")

    def stream_invoke(
            self,
            messages: List[Message],
            temperature: Optional[float] = None,
            **kwargs: Any,
    ) -> Iterator[str]:
        """
        流式调用模型，逐块生成文本（生成器模式）。
        适合实时打字机显示效果或长时间推理任务
        Args:
            messages: 统一 Message 对象列表。
            temperature: 临时覆盖当前调用的采样温度
        Yields:
            str: 模型生成的文本片段
        Raises:
            LLMCallException: API 调用失败、网络异常或配额受限时抛出。
        """
        system_instruction, contents = self._prepare_gemini_payload(messages)
        temp = temperature if temperature is not None else self.temperature

        generate_config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temp,
            # 手写 Agent 范式（如 ReAct），必须禁用 SDK 的自动工具执行，交由上层调度
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        )

        try:
            response_stream = self._client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=generate_config,
            )

            for chunk in response_stream:
                chunk_text = chunk.text or ""
                if chunk_text:
                    yield chunk_text

        except errors.APIError as e:
            raise LLMCallException(
                f"Vertex AI API 远程调用失败 [错误码 {e.code}]: {e.message}",
                original_error=e,
            ) from e
        except Exception as e:
            raise LLMCallException(
                f"模型调用过程中发生未预期异常: {e}",
                original_error=e,
            ) from e

    def invoke(
            self,
            messages: List[Message],
            temperature: Optional[float] = None,
            **kwargs: Any,
    ) -> str:
        """
        非流式调用模型，返回完整的回答文本。
         Agent 的内部思考逻辑（如 ReAct 正则解析、提取 Action）通常需要完整字符串。
         Args:
             messages: 统一 Message 对象列表。
             temperature: 临时覆盖当前调用的采样温度。
         Returns:
             str: 模型返回的完整文本内容。
        """
        collected_chunks: List[str] = []
        for chunk in self.stream_invoke(messages, temperature=temperature, **kwargs):
            collected_chunks.append(chunk)

        full_text = "".join(collected_chunks).strip()
        if not full_text:
            raise LLMCallException("Vertex AI 调用成功但返回了空内容。")

        return full_text

    def think(
            self,
            messages: List[Message],
            **kwargs: Any,
    ) -> Iterator[str]:
        """兼容教程历史版本的思考方法（等同于 stream_invoke）。"""
        return self.stream_invoke(messages, **kwargs)

    def _prepare_gemini_payload(
            self,
            messages: List[Message]
    ) -> tuple[Optional[str], List[types.Content]]:
        """
        辅助方法：将框架统一的 Message 列表转换为 Gemini 专属的参数结构。
        Gemini SDK 规范要求：
        - 系统提示词必须单独放在 generate_config.system_instruction 中；
        - 用户与助手的多轮对话放在 contents 中。
        """
        system_prompts: List[str] = []
        contents: List[types.Content] = []

        for message in messages:
            if not message.content.strip():
                continue
            if message.role == "system":
                system_prompts.append(message.content.strip())
            else:
                contents.append(message.to_gemini_content())

        if not contents:
            raise LLMCallException("发送给模型的消息列表不能为空（必须至少包含一条非系统消息）。")

        system_instruction = "\n\n".join(system_prompts) if system_prompts else None
        return system_instruction, contents


    def close(self) -> None:
        """关闭底层网络连接与客户端会话。"""
        if hasattr(self, "_client") and self._client:
            self._client.close()


if __name__ == "__main__":
    # --- 模块自测逻辑：向 Vertex AI 发起一次真实的连通性验证 ---
    print("--- 正在测试 HelloAgentsLLM 与 Vertex AI 的真实连通性 ---")

    llm = HelloAgentsLLM()
    test_messages = [
        Message.system("你是一个专业的金融量化分析助手。回答要简明扼要。"),
        Message.user("你好！请用一句话告诉我什么是量化交易。"),
    ]

    try:
        # 测试 1: 非流式 invoke 调用
        print("\n[测试 1: 完整内容生成 invoke]")
        reply = llm.invoke(test_messages)
        print(f"🤖 模型回复: {reply}")

        # 测试 2: 流式 stream_invoke 调用
        print("\n[测试 2: 打字机流式生成 stream_invoke]")
        print("🤖 实时输出: ", end="", flush=True)
        for chunk in llm.stream_invoke([Message.user("请列出美股前三大科技巨头简称")]):
            print(chunk, end="", flush=True)
        print()

        print("\n✅ HelloAgentsLLM 与 Vertex AI 连通性测试全部通过！")

    except LLMCallException as e:
        print(f"\n❌ 模型调用失败: {e}")
        if e.original_error:
            print(f"   底层详细错误: {e.original_error}")
    finally:
        llm.close()