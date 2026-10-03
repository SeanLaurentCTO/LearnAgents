"""统一消息系统定义。

规范智能体与大模型、工具、用户之间流转的消息数据结构。
支持标准 OpenAI 字典格式与 Google GenAI (Vertex AI) Content 格式的相互转换。
"""

from datetime import datetime
from typing import Any, Dict, Literal, Optional
from google.genai import types
from pydantic import BaseModel, Field

# 严格限制消息角色的取值范围：
# - user: 用户或外部驱动者的输入
# - assistant: 大语言模型的回答或思考过程
# - system: 设定角色身份、规则与工具描述的系统提示词
# - tool: 工具执行返回的观察结果 (Observation)
MessageRole = Literal["user", "assistant", "system", "tool"]


class Message(BaseModel):
    """HelloAgents 统一消息类。

    基于 Pydantic 构建，保证角色类型安全，并自带时间戳与元数据存储能力。
    """
    content: str = Field(
        ...,
        description="消息的具体文本内容",
    )
    role: MessageRole = Field(
        ...,
        description="消息的角色身份，严格限定在 MessageRole 范围内",
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="消息创建的时间戳，默认自动生成当前时间",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="扩展元数据字典（例如：token 统计、工具名称、调用状态等）",
    )

    def to_dict(self) -> Dict[str, Any]:
        """转换为通用的字典格式（兼容标准 OpenAI 消息格式）。

               Returns:
                   符合 {"role": ..., "content": ...} 的纯字典。
        """

        return {
            "role": self.role,
            "content": self.content,
        }


    def to_gemini_content(self) -> types.Content:
        """转换为 Google GenAI (Vertex AI) SDK 原生的 Content 数据对象。

               Gemini 角色映射规则：
               - user / tool  -> 对应 Gemini 的 'user'
               - assistant    -> 对应 Gemini 的 'model'
               - 注意：system 消息在 Gemini 中通常作为系统的 generate_config.system_instruction 传入，
                      如果强行转为 Content，默认映射为 'user' 角色。

               Returns:
                   google.genai.types.Content 实例，可直接传给 client.models.generate_content。
        """

        gemini_role = "model" if self.role == "assistant" else "user"
        return types.Content(
            role=gemini_role,
            parts=[
                types.Part.from_text(text=self.content),
            ]
        )

    @classmethod
    def user(cls, content: str, **metadata: Any) -> "Message":
        """快捷工厂方法：快速创建用户消息。"""
        return cls(role="user", content=content, metadata=metadata)

    @classmethod
    def assistant(cls, content: str, **metadata: Any) -> "Message":
        """快捷工厂方法：快速创建助手回复消息。"""
        return cls(role="assistant", content=content, metadata=metadata)

    @classmethod
    def system(cls, content: str, **metadata: Any) -> "Message":
        """快捷工厂方法：快速创建系统提示词消息。"""
        return cls(role="system", content=content, metadata=metadata)

    @classmethod
    def tool(cls, content: str, tool_name: str, **metadata: Any) -> "Message":
        """快捷工厂方法：快速创建工具执行结果消息。"""
        meta = {"tool_name": tool_name, **metadata}
        return cls(role="tool", content=content, metadata=meta)

    def __str__(self):
        time_str = self.timestamp.strftime("%H:%M:%S")
        return f"[{time_str}] [{self.role.upper()}]: {self.content}"



if __name__ == "__main__":
    # --- 模块自测逻辑 ---
    print("--- 正在验证 Message 数据结构与格式转换 ---")

    # 1. 验证快捷构建与 Pydantic 校验
    msg_user = Message.user("你好，请帮我分析美股大盘。")
    msg_ai = Message.assistant("收到，正在查询美股三大指数数据。")
    msg_tool = Message.tool("SPY: 580.20, QQQ: 490.15", tool_name="market_quote")

    print(f"✅ 创建用户消息: {msg_user}")
    print(f"✅ 创建模型消息: {msg_ai}")
    print(f"✅ 创建工具消息: {msg_tool} (元数据: {msg_tool.metadata})")

    # 2. 验证通用字典转换（OpenAI 格式）
    dict_format = msg_user.to_dict()
    assert dict_format == {"role": "user", "content": "你好，请帮我分析美股大盘。"}
    print(f"✅ 通用字典转换验证通过: {dict_format}")

    # 3. 验证 Vertex AI 原生 Content 转换
    gemini_content = msg_ai.to_gemini_content()
    assert gemini_content.role == "model"
    assert gemini_content.parts[0].text == "收到，正在查询美股三大指数数据。"
    print(f"✅ Vertex AI Content 转换验证通过 (role={gemini_content.role})")

