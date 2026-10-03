"""智能体 (Agent) 抽象基类定义。

定义所有具体智能体（如 SimpleAgent, ReActAgent 等）通用的属性、
生命周期管理、多轮历史消息维护契约以及必须实现的执行接口。
"""


from abc import ABC, abstractmethod
from typing import Any, Iterator, List, Optional

from chapter07_my_framework.core.config import Config
from chapter07_my_framework.core.llm import HelloAgentsLLM
from chapter07_my_framework.core.message import Message


class Agent(ABC):
    """
    所有智能体的抽象基类 (Abstract Base Class)。
    继承自 abc.ABC，不能被直接实例化，必须由具体的 Agent 子类继承并实现抽象方法。
    """

    def __init__(
            self,
            name: str,
            llm: HelloAgentsLLM,
            system_prompt: Optional[str] = None,
            config: Optional[Config] = None,
    ) -> None:
        """初始化智能体基础属性
         Args:
             name: 智能体名称（如 'InvestmentAdvisor'）。
             llm: 绑定的大模型通信客户端 (HelloAgentsLLM 实例)。
             system_prompt: 设定角色身份、规则与行为准则的系统提示词。
             config: 框架配置对象，不传则自动从环境加载默认配置。
         """
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config.from_env()

        # 内部状态：维护该 Agent 专属的多轮对话历史记录
        # 单下划线前缀表示受保护属性，建议外部通过标准方法访问与维护
        self._history: List[Message] = []

    @abstractmethod
    def run(self, input_text, **kwargs:Any) -> str:
        """
        运行智能体，接收用户输入并返回最终执行结果。
        这是所有 Agent 子类必须实现的核心抽象方法。

        Args:
            input_text: 用户的自然语言指令或任务描述。
            **kwargs: 额外的执行参数（如特定调用的采样温度、最大步数等）。

        Returns:
            str: 智能体最终给出的完整回复或行动结论。
        """
        pass


    def add_message(self, message: Message) -> None:
        """向当前智能体的历史记录中追加一条新消息
         Args:
             message: 符合框架标准的 Message 实例。
         """

        self._history.append(message)

        # 历史记录长度保护：防止长期对话超过上下文窗口或内存溢出
        max_len = self.config.max_history_length
        if len(self._history) > max_len:
            # 始终保留最近的 max_len 条记录
            self._history = self._history[-max_len:]


    def get_history(self) -> List[Message]:
        """获取当前智能体的完整历史记录副本
        Returns:
            List[Message]: 历史记录的浅拷贝列表，防止外部直接修改私有列表引用。
        """
        return self._history.copy()


    def clear_history(self) -> None:
        """清空当前智能体的历史记忆（重置上下文）。"""
        self._history.clear()

    def __str__(self) -> str:
        """人性化字符串表示。"""
        return f"Agent(name='{self.name}', model='{self.llm.model}')"

    def __repr__(self) -> str:
        return self.__str__()


if __name__ == "__main__":
    # --- 模块自测逻辑：验证抽象基类的约束机制与记忆管理 ---
    print("--- 正在验证 Agent 抽象基类的契约机制 ---")

    # 1. 验证抽象约束：直接实例化基类必须报错
    print("\n[测试 1: 验证抽象类不可直接实例化]")
    try:
        dummy_llm = HelloAgentsLLM()
        agent = Agent("TestAgent", dummy_llm)  # type: ignore
        print("❌ 错误：抽象类竟然被实例化了！")
    except TypeError as e:
        print(f"✅ 成功阻止直接实例化抽象类: {e}")

    # 2. 验证子类继承与记忆维护
    print("\n[测试 2: 验证子类实现与多轮历史维护]")

    class MockSimpleAgent(Agent):
        """用于测试基类机制的最小子类实现。"""

        def run(self, input_text: str, **kwargs: Any) -> str:
            # 记录用户消息
            self.add_message(Message.user(input_text))
            # 模拟回复
            reply = f"Mock 回复: 收到关于 '{input_text}' 的指令"
            self.add_message(Message.assistant(reply))
            return reply

    test_agent = MockSimpleAgent("测试助手", dummy_llm)
    print(f"✅ 成功实例化子类: {test_agent}")

    # 执行两轮对话
    r1 = test_agent.run("第一条指令")
    r2 = test_agent.run("第二条指令")

    history = test_agent.get_history()
    assert len(history) == 4  # 2 条 user + 2 条 assistant
    print(f"✅ 历史记录追踪正常，当前共有 {len(history)} 条消息:")
    for msg in history:
        print(f"   {msg}")

    # 清空历史测试
    test_agent.clear_history()
    assert len(test_agent.get_history()) == 0
    print("✅ 历史记录清空成功！")

    print("\n🎉 Agent 基类设计全部验证通过！")
