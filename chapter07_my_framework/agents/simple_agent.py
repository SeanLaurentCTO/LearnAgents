"""基础对话智能体 (SimpleAgent) 实现。

继承自 Agent 基类，实现标准的单次对话、多轮会话记忆维护与流式响应输出。
作为后续扩展复杂范式（ReAct / 工具调用）的基础形态。
"""

from typing import Any, Iterator, List, Optional

from chapter07_my_framework.core.agent import Agent
from chapter07_my_framework.core.config import Config
from chapter07_my_framework.core.llm import HelloAgentsLLM
from chapter07_my_framework.core.message import Message


class SimpleAgent(Agent):
    """
    基础对话智能体。
    具备完整的端到端上下文组装能力与多轮对话记忆状态机。
    """
    def __init__(
            self,
            name: str,
            llm: HelloAgentsLLM,
            system_prompt: Optional[str] = None,
            config: Optional[Config] = None,
    ) -> None:
        """初始化 SimpleAgent
        Args:
            name: 智能体名称。
            llm: 大模型通信客户端实例。
            system_prompt: 角色的人设、规则与行为准则系统提示词。
            config: 运行时配置对象。
        """
        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt,
            config=config,
        )


    def run(
            self,
            input_text: str,
            **kwargs: Any
    ) -> str:
        """运行智能体，进行一次完整的同步对话（非流式）。
        流程：
        1. 将用户输入追加到内部对话历史；
        2. 组装包含系统提示词与历史记录的完整上下文；
        3. 调用大模型生成回答；
        4. 将大模型回复追加到内部对话历史；
        5. 返回回复文本。

        Args:
            input_text: 用户输入的指令或提问。
            **kwargs: 允许传入临时控制参数（如 temperature 等）。

        Returns:
            str: 智能体给出的最终完整回答。
        """

        # 记录本轮用户信息到短期记忆中
        user_msg = Message.user(input_text)
        self.add_message(user_msg)

        # 组装信息发送给大模型完整的上下文序列
        context_messages = self._build_context_messages()

        # 调用模型完成回答
        reply_text = self.llm.invoke(context_messages, **kwargs)

        # 将模型的回答添加到短期记忆中，形成完整闭环
        assistant_msg = Message.assistant(reply_text)
        self.add_message(assistant_msg)

        return reply_text


    def stream_run(
            self,
            input_text: str,
            **kwargs: Any
    ) -> Iterator[str]:
        """流式运行智能体，逐块输出文本片段（打字机效果）
        同样保证多轮对话记忆的完整记录
        Args:
            input_text: 用户输入的指令或提问。
            **kwargs: 传给底层的模型调用参数
        Yields:
            str: 逐块生成的回答片段。
        """

        # 记录用户消息
        user_msg = Message.user(input_text)
        self.add_message(user_msg)

        # 组装上下文
        context_messages = self._build_context_messages()

        # 流式输出片段，实现一边生成回答一边汇总的效果
        collected_chunks: List[Message] = []
        for chunk in self.llm.stream_invoke(context_messages, **kwargs):
            collected_chunks.append(chunk)
            yield chunk
        # 4. 流式传输完毕后，将拼接好的完整内容存入历史记忆
        full_reply = "".join(collected_chunks).strip()
        assistant_msg = Message.assistant(full_reply)
        self.add_message(assistant_msg)


    def _build_context_messages(self) -> List[Message]:
        """内部方法：组装包含系统人设与历史对话的上下文消息列表
        Returns:
            List[Message]: 符合调用标准的有序消息序列。
        """

        messages: List[Message] = []

        # 设置了系统提示词，始终作为上下文的第一条消息
        if self.system_prompt and self.system_prompt.strip():
            messages.append(Message.system(self.system_prompt.strip()))

        # 追加历史中所有的已发生的对话记录（包含最新的用户输入）
        messages.extend(self.get_history())

        return messages


if __name__ == "__main__":
    # --- 模块自测逻辑：验证多轮对话连贯性与流式响应 ---
    print("--- 正在测试 SimpleAgent 与 Vertex AI 的真实多轮交互 ---")

    # 1. 实例化 LLM 与 SimpleAgent
    llm = HelloAgentsLLM()
    agent = SimpleAgent(
        name="投资小助手",
        llm=llm,
        system_prompt="你是一位擅长量化分析的金融投资顾问。回答请简洁专业，字数控制在60字以内。",
    )
    print(f"✅ 成功初始化: {agent}")

    # 2. 第一轮问答（测试非流式 run）
    print("\n[轮次 1: 建立记忆 (非流式 run)]")
    q1 = "你好！我手头有 30 澳元闲置资金，请问我的本金是多少澳元？"
    print(f"👤 用户: {q1}")
    a1 = agent.run(q1)
    print(f"🤖 助手: {a1}")

    # 3. 第二轮问答（测试是否具备多轮记忆能力）
    print("\n[轮次 2: 验证多轮上下文关联 (非流式 run)]")
    q2 = "如果按照 1 AUD = 0.65 USD 汇率计算，我刚才提到的这笔资金折合多少美元？"
    print(f"👤 用户: {q2}")
    a2 = agent.run(q2)
    print(f"🤖 助手: {a2}")


    # # 4. 第三轮问答（测试流式 stream_run）
    # print("\n[轮次 2.1]")
    # q2_1 = "你认为我用这 30 澳元可以进行一些什么样的投资？"
    # print(f"👤 用户: {q2_1}")
    # print("🤖 助手实时输出: ", end="", flush=True)
    # for chunk in agent.stream_run(q2_1):
    #     print(chunk, end="", flush=True)
    # print()


    # 4. 第三轮问答（测试流式 stream_run）
    print("\n[轮次 3: 验证流式生成 (stream_run)]")
    q3 = "请用一句话给新手一句极小资金投资的风险警示。"
    print(f"👤 用户: {q3}")
    print("🤖 助手实时输出: ", end="", flush=True)
    for chunk in agent.stream_run(q3):
        print(chunk, end="", flush=True)
    print()


    # 5. 验证对话历史条数
    history = agent.get_history()
    print(f"\n📊 对话历史审计: 当前 Agent 记忆中共保存了 {len(history)} 条消息（应为 6 条：3问 + 3答）")
    assert len(history) == 6, f"预期 6 条消息，实际保存了 {len(history)} 条！"

    print("\n🎉 SimpleAgent 真实多轮对话与流式功能全部验证通过！")
    llm.close()