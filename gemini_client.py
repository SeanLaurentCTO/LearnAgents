"""Hello Agents 第四章共用的 Vertex AI 客户端。"""

import os
from typing import TypedDict

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

# 将项目根目录 .env 中的配置加载到当前进程的环境变量。
load_dotenv()


class Message(TypedDict):
    """与具体模型供应商无关的统一 LLM 消息格式。"""

    role: str
    content: str


def build_gemini_client() -> genai.Client:
    """创建连接 Vertex AI 的 Gemini 客户端。"""
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "global")

    if not project:
        raise RuntimeError("缺少 GOOGLE_CLOUD_PROJECT 配置，请检查项目根目录下的 .env 文件。")

    # 不传固定 API Key；客户端会使用本机配置的 ADC 身份凭证。
    return genai.Client(
        vertexai=True,
        project=project,
        location=location,
    )


class HelloAgentsLLM:
    """供 Hello Agents 第四章使用的 Vertex AI LLM 客户端。"""

    def __init__(self, model: str | None = None) -> None:
        # 显式传入的模型优先，未传入时才读取 .env。
        self.model = model or os.getenv("GEMINI_MODEL")

        if not self.model:
            raise RuntimeError("缺少 GEMINI_MODEL 配置，请检查项目根目录下的 .env 文件。")
        self.client = build_gemini_client()

    def think(self, messages: list[Message], temperature: float = 0) -> str:
        """调用 Vertex AI，并返回完整的文本响应。"""

        # Gemini 将系统指令和普通对话放在不同参数中，因此需要分开收集。
        system_instructions: list[str] = []
        contents: list[types.Content] = []

        for message in messages:
            role = message["role"]
            content = message["content"].strip()

            if not content:
                continue

            if role == "system":
                system_instructions.append(content)
                continue

            # 通用接口使用 assistant；Gemini 原生接口使用 model。
            if role == "user":
                gemini_role = "user"
            elif role in {"assistant", "model"}:
                gemini_role = "model"
            else:
                raise ValueError(f"不支持的消息角色：{role}")

            # Content 表示一轮消息；Part 允许一轮消息包含文本、图片或工具调用。
            contents.append(
                types.Content(
                    role=gemini_role,
                    parts=[types.Part.from_text(text=content)],
                )
            )

        if not contents:
            raise ValueError("messages 中没有可以发送给模型的消息...")

        # 多条系统指令合并后传给 Gemini；没有系统指令时使用 None。
        system_instruction = "\n\n".join(system_instructions) or None

        print(f"正在调用模型：{self.model}")

        # 流式输出会返回多个文本片段，需要收集后再组成完整响应。
        collected_content: list[str] = []

        try:
            for chunk in self.client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=temperature,
                    # 第四章将手写 Agent Loop，不能让 SDK 自动执行工具。
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True,
                    ),
                ),
            ):
                text = chunk.text or ""
                if not text:
                    continue
                # 一边实时显示，一边保留文本供 Agent 后续解析。
                print(text, end="", flush=True)
                collected_content.append(text)

        except errors.APIError as error:
            # 转换为项目层异常，同时用 from 保留原始异常链。
            raise RuntimeError(
                f"Vertex AI 请求失败：状态码={error.code}，"
                f"错误信息={error.message}"
            ) from error

        print()

        # Agent 需要的是完整字符串，而不是零散的流式片段。
        response_text = "".join(collected_content).strip()

        if not response_text:
            raise RuntimeError("Vertex AI 返回了空响应。")

        return response_text

    def close(self) -> None:
        """关闭客户端持有的网络资源。"""
        self.client.close()


if __name__ == "__main__":
    # 只有直接运行本文件时才执行连通性测试；被导入时不会自动请求 API。
    llm = HelloAgentsLLM()

    try:
        response = llm.think(
            messages=[
                {
                    "role": "system",
                    "content": "你是一位简洁、准确的 Python 教师。",
                },
                {
                    "role": "user",
                    "content": "请用一句话解释什么是智能体。",
                },
            ],
            temperature=0,
        )

        print("\n--- 调用完成 ---")
        print(f"响应字符数：{len(response)}")

    finally:
        # 即使模型调用发生异常，也确保底层 HTTP 资源被释放。
        llm.close()
