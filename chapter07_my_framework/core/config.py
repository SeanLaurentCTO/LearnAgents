"""框架统一配置中心。

负责集中读取、校验并管理环境变量与运行时配置。
重点适配 Google Cloud Vertex AI 所需的凭证与模型配置。
"""

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from chapter07_my_framework.core.exceptions import ConfigurationException

# 定位到项目根目录（LearnAgents/），确保跨工作目录运行时也能正确加载 .env
# config.py -> core -> chapter07_my_framework -> LearnAgents (上溯 3 级)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


class Config(BaseModel):
    """统一配置模型类"""

    google_cloud_project: str = Field(
        default_factory=lambda: os.getenv("GOOGLE_CLOUD_PROJECT", ""),
        description="Google Cloud 项目 ID Vertex AI 必填项",
    )
    google_cloud_location: str = Field(
        default_factory=lambda: os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
        description="Vertex AI 区域节点，默认使用 global",
    )
    model: str = Field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        description="默认调用的 Gemini 模型名称",
    )

    # --- LLM 采样与调用参数 ---
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="采样温度，0.0 偏严谨确定，1.0+ 偏创造发散",
    )
    max_tokens: int | None = Field(
        default=None,
        description="最大生成 Token 数，None 表示模型默认",
    )
    timeout: float = Field(
        default=60.0,
        description="请求超时时间（秒）",
    )

    # --- 系统与调试参数 ---
    debug: bool = Field(
        default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true",
        description="是否开启框架内部调试日志",
    )
    max_history_length: int = Field(
        default=50,
        description="Agent 维护的历史消息保留最大轮数",
    )

    def validate_for_vertex(self) -> None:
        """检查 Vertex AI 运行所必须的核心配置是否存在。"""

        if not self.google_cloud_project:
            raise ConfigurationException(
                "缺少 GOOGLE_CLOUD_PROJECT 配置。请在项目根目录 .env 文件中设置。"
            )
        if not self.model:
            raise ConfigurationException(
                "缺少 GEMINI_MODEL 配置。请在项目根目录 .env 文件中设置。"
            )

    @classmethod
    def from_env(cls, **overrides: Any) -> "Config":
        """工厂方法：从环境变量读取配置，并允许传入显式参数进行覆盖。"""
        return cls(**overrides)


if __name__ == "__main__":
    # 自测逻辑：检查配置读取情况
    print("--- 正在验证 Config 配置加载 ---")
    try:
        cfg = Config.from_env()
        cfg.validate_for_vertex()
        print(f"✅ 配置读取成功！")
        print(f"   Project:  {cfg.google_cloud_project}")
        print(f"   Location: {cfg.google_cloud_location}")
        print(f"   Model:    {cfg.model}")
        print(f"   Debug:    {cfg.debug}")
    except ConfigurationException as e:
        print(f"❌ 配置校验失败: {e}")