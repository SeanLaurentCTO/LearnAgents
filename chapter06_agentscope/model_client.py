"""使用AgentScope连接Vertex AI Gemini的模型适配器。"""

import asyncio
import os
from pathlib import Path

import google.auth
from agentscope.formatter import GeminiChatFormatter
from agentscope.message import Msg
from agentscope.model import ChatResponse, GeminiChatModel
from dotenv import load_dotenv
from google.auth.credentials import Credentials
from google.auth.exceptions import DefaultCredentialsError, RefreshError
from google.auth.transport.requests import Request

# 当前文件位于 LearnAgents/chapter06_agentscope/model_client.py，
# parents[1] 即项目根目录 LearnAgents。
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 显式读取项目根目录下的 .env，避免程序从其他工作目录启动时找不到配置。
load_dotenv(PROJECT_ROOT / ".env")

# ADC请求Vertex AI所需的Google Cloud OAuth权限范围。
GOOGLE_CLOUD_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


def _normalize_gemini_model_id(model_id: str) -> str:
    """转换为原生Gemini SDK需要的模型ID。

    OpenAI-compatible端点使用``google/<model>``，原生Gemini SDK使用
    不带``google/``前缀的模型名称。
    """
    return model_id.removeprefix("google/")


def _get_google_credentials(project_id: str) -> Credentials:
    """获取并刷新ADC凭证对象。

    与只把当前短期Token交给OpenAI客户端不同，原生Google客户端持有
    credentials对象后可以在Token过期时继续自动刷新。
    """

    try:
        credentials, detected_project_id = google.auth.default(
            scopes=[GOOGLE_CLOUD_SCOPE],
        )

        # 刷新ADC凭证并取得短期Token。Token具有有效期，不是固定API Key。
        credentials.refresh(Request())

    except DefaultCredentialsError as error:
        raise RuntimeError(
            "没有找到Google Application Default Credentials。"
            "请先运行：gcloud auth application-default login",
        ) from error
    except RefreshError as error:
        raise RuntimeError(
            "Google ADC存在，但刷新Vertex AI access token失败。"
            "请检查登录状态、网络连接和项目权限。",
        ) from error

    if not credentials.token:
        raise RuntimeError("Google ADC刷新完成，但没有返回access token。")

    if detected_project_id and detected_project_id != project_id:
        print(
            "提示：ADC检测到的项目与.env配置不同：\n"
            f"  ADC项目：{detected_project_id}\n"
            f"  请求项目：{project_id}\n"
            "本次请求将使用.env中的GOOGLE_CLOUD_PROJECT。",
        )
    return credentials


def create_vertex_ai_chat_model(
    *,
    stream: bool = False,
    temperature: float = 0.0,
) -> GeminiChatModel:
    """创建使用Vertex AI原生Gemini接口的AgentScope模型。

    这里不用OpenAI-compatible适配层，因为Gemini 3工具调用要求在后续
    对话中原样回传``thought_signature``。AgentScope的原生
    ``GeminiChatModel``和Gemini Formatter会保留该字段。
    """

    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
    model_id = os.getenv("GEMINI_MODEL")

    if not project_id:
        raise RuntimeError(
            "缺少GOOGLE_CLOUD_PROJECT，请检查项目根目录下的.env文件。",
        )

    if not model_id:
        raise RuntimeError(
            "缺少GEMINI_MODEL，请检查项目根目录下的.env文件。",
        )

    vertex_model_id = _normalize_gemini_model_id(model_id)
    credentials = _get_google_credentials(project_id)

    print("正在创建AgentScope Vertex AI聊天模型：")
    print(f"  Project：{project_id}")
    print(f"  Location：{location}")
    print(f"  Model：{vertex_model_id}")
    print("  API：Vertex AI native Gemini API")

    return GeminiChatModel(
        model_name=vertex_model_id,
        # Vertex AI模式使用ADC credentials，不使用Gemini Developer API Key。
        api_key=None,  # type: ignore[arg-type]
        stream=stream,
        client_kwargs={
            "vertexai": True,
            "project": project_id,
            "location": location,
            "credentials": credentials,
        },
        generate_kwargs={
            "temperature": temperature,
        },
    )


def _extract_text(response: ChatResponse) -> str:
    """从AgentScope ChatResponse中提取全部文本块。"""
    text_parts = [
        block["text"]
        for block in response.content
        if block.get("type") == "text" and block.get("text")
    ]
    return "".join(text_parts).strip()


async def check_model_connection() -> None:
    """发送一条最小消息，验证AgentScope到Vertex AI的完整调用链。"""

    model = create_vertex_ai_chat_model(stream=False)

    try:
        # 原生Gemini模型需要Gemini格式的contents；Formatter负责把统一
        # AgentScope Msg转换为该格式。
        formatter = GeminiChatFormatter()
        messages = await formatter.format(
            [
                Msg(
                    name="system",
                    role="system",
                    content="你是一位简洁、准确的AI智能体教师。",
                ),
                Msg(
                    name="user",
                    role="user",
                    content="请用一句中文解释AgentScope中的Msg是什么。",
                ),
            ],
        )
        response = await model(
            messages=messages,
        )

        if not isinstance(response, ChatResponse):
            raise RuntimeError(
                "预期AgentScope返回ChatResponse，实际返回："
                f"{type(response).__name__}",
            )

        response_text = _extract_text(response)
        if not response_text:
            raise RuntimeError(
                f"Vertex AI返回了响应，但没有可显示的文本：{response.content!r}",
            )

        print("\nAgentScope Vertex AI API调用成功：")
        print(response_text)

    finally:
        # 原生google-genai客户端分别持有异步和同步资源。
        await model.client.aio.aclose()
        model.client.close()


if __name__ == "__main__":
    asyncio.run(check_model_connection())
