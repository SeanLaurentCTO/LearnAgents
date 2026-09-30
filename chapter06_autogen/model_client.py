"""AutoGen 使用的 Vertex AI Gemini 模型客户端。"""

import asyncio
import os
from pathlib import Path

import google.auth
from autogen_core.models import ModelFamily, UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
from google.auth.exceptions import DefaultCredentialsError, RefreshError
from google.auth.transport.requests import Request


# model_client.py 位于：
# LearnAgents/chapter06_autogen/model_client.py
# parents[1] 因此是项目根目录 LearnAgents。
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 显式读取项目根目录下的 .env。
# 这样无论从哪个工作目录启动程序，都能找到相同的配置文件。
load_dotenv(PROJECT_ROOT / ".env")


def create_vertex_model_client() -> OpenAIChatCompletionClient:
    """创建供 AutoGen 使用的 Vertex AI Gemini 模型客户端。

    调用链如下：

        AutoGen Agent
            ↓
        OpenAIChatCompletionClient
            ↓ OpenAI-compatible 请求格式
        Vertex AI openapi endpoint
            ↓
        Gemini 模型

    Returns:
        已配置完成的 AutoGen OpenAIChatCompletionClient。

    Raises:
        RuntimeError: 环境配置或 Google ADC 凭证不可用时抛出。
    """
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
    model_id = os.getenv("GEMINI_MODEL")

    # 单独检查变量，方便快速定位配置遗漏。
    if not project_id:
        raise RuntimeError(
            "缺少 GOOGLE_CLOUD_PROJECT。"
            "请检查项目根目录下的 .env 文件。"
        )

    if not model_id:
        raise RuntimeError(
            "缺少 GEMINI_MODEL。"
            "请检查项目根目录下的 .env 文件。"
        )

    # global endpoint 使用 aiplatform.googleapis.com；
    # 区域 endpoint 使用 <region>-aiplatform.googleapis.com。
    if location == "global":
        host = "aiplatform.googleapis.com"
    else:
        host = f"{location}-aiplatform.googleapis.com"

    try:
        # google.auth.default() 返回二元组：
        # (credentials 对象, ADC 检测到的 project_id 或 None)。
        # 必须解包，不能把整个 tuple 当成 credentials 使用。
        credentials, detected_project_id = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )

        # 使用本机 ADC 获取短期 OAuth access token。
        credentials.refresh(Request())

    except DefaultCredentialsError as error:
        raise RuntimeError(
            "没有找到可用的 Google Application Default Credentials。"
            "请先配置 ADC，例如运行："
            "gcloud auth application-default login"
        ) from error

    except RefreshError as error:
        raise RuntimeError(
            "Google ADC 存在，但刷新 access token 失败。"
            "请检查登录状态、网络连接以及 Vertex AI 权限。"
        ) from error

    if not credentials.token:
        raise RuntimeError(
            "Google ADC 刷新完成，但没有取得 access token。"
        )

    # ADC 检测到的项目与 .env 项目不同时给出提示；
    # 实际请求明确使用 GOOGLE_CLOUD_PROJECT。
    if detected_project_id and detected_project_id != project_id:
        print(
            "提示：ADC 检测到的项目与 .env 配置不同：\n"
            f"  ADC 项目：{detected_project_id}\n"
            f"  请求项目：{project_id}\n"
            "本次请求将使用 .env 中的请求项目。"
        )

    # Vertex AI 的 OpenAI-compatible endpoint 路径。
    base_url = (
        f"https://{host}/v1/"
        f"projects/{project_id}/"
        f"locations/{location}/endpoints/openapi"
    )

    # Vertex endpoint 使用 google/模型ID 格式。
    # 若环境变量里已经带此前缀，则避免重复添加。
    vertex_model_id = (
        model_id if model_id.startswith("google/") else f"google/{model_id}"
    )

    print("正在创建 AutoGen Vertex AI 模型客户端：")
    print(f"  Project：{project_id}")
    print(f"  Location：{location}")
    print(f"  Model：{vertex_model_id}")
    print(f"  Endpoint：{base_url}")

    return OpenAIChatCompletionClient(
        model=vertex_model_id,
        base_url=base_url,
        # OpenAI SDK 使用 Bearer 认证；这里传入 ADC 获取的短期 token，
        # 不是固定 API Key。
        api_key=credentials.token,
        # 明确告诉 AutoGen 此模型支持哪些功能。
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": ModelFamily.UNKNOWN,
            "structured_output": True,
        },
    )


async def test_model_client() -> None:
    """发送一条最小请求，验证 Vertex AI API 是否连通。"""
    model_client = create_vertex_model_client()

    try:
        result = await model_client.create(
            messages=[
                UserMessage(
                    content="请用一句中文解释什么是 AI 智能体。",
                    source="user",
                )
            ]
        )

        # 本次没有注册工具，预期模型返回普通文本。
        if not isinstance(result.content, str):
            raise RuntimeError(
                "预期模型返回文本，但实际返回了"
                f"{type(result.content).__name__}：{result.content!r}"
            )

        print("\nVertex AI API 调用成功：")
        print(result.content)

    finally:
        # 即使请求失败，也要关闭 AutoGen 持有的异步 HTTP 资源。
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(test_model_client())
