"""工具 (Tool) 抽象基类定义。

贯彻“万物皆为工具”的框架设计理念。
所有具体工具（如计算器、汇率换算器、股票行情检索器等）均继承自此类，
对外提供统一的名称、功能描述、参数说明以及安全的执行接口。
"""

from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseTool(ABC):
    """所有工具的抽象基类。
    为智能体提供感知外部世界与执行具体任务的统一标准。
    """

    def __init__(
            self,
            name: str,
            description: str,
            parameters_description: Optional[str] = None,
    ) -> None:
        """初始化工具元数据
        Args:
            name: 工具的唯一名称（大模型据此生成 Action 调用指令，如 'calculator'）。
            description: 工具功能详述（大模型据此判断是否需要使用该工具）。
            parameters_description: 工具入参格式要求（指导大模型如何正确传参）。
        """

        self.name = name
        self.description = description
        self.parameters_description = (
            parameters_description.strip()
            if parameters_description
            else '无需特殊参数，或传入单个字符串文本...'
        )


    @abstractmethod
    def execute(self, **kwargs: Any) -> str:
        """工具的具体业务执行逻辑
        这是所有工具子类必须实现的抽象方法
        Args:
            **kwargs: 工具所需的命名参数
        Returns:
            str: 工具执行后的字符串结果（将作为 Observation 反馈给大模型）
        Raises:
            Exception: 工具内部发生错误时抛出。
        """
        pass


    def run(self,**kwargs: Any) -> str:
        """安全执行工具的安全包装层（带有自动异常捕获与容错）
        智能体在调度工具时应优先调用此方法，防止工具内部抛出未捕获异常导致 Agent 崩溃
        Args:
            **kwargs: 传给 execute 的具体参数
        Returns:
            str: 成功时返回 execute 执行结果；失败时返回格式化的错误说明字符串。
        """

        try:
            result = self.execute(**kwargs)
            return str(result)
        except Exception as e:
            # 优雅降级：将异常转化为文本，让大模型能感知错误并尝试反思重试
            error_msg = f"工具 '{self.name}' 执行出错: {str(e)}"
            return error_msg


    def get_tool_info(self) -> str:
        """生成格式化的工具信息描述，用于直接注入到系统提示词中供大模型阅读
        Returns:
            str: 包含工具名称、功能及参数规范的文本块。
        """

        return (
            f"  - **{self.name}**:\n"
            f"  - 功能说明: {self.description}\n"
            f"  - 参数规范: {self.parameters_description}"
        )

    def __str__(self) -> str:
        return f"Tool(name='{self.name}')"

    def __repr__(self) -> str:
        return self.__str__()


if __name__ == "__main__":
    # --- 模块自测逻辑：验证工具基类的抽象约束与容错机制 ---
    print("--- 正在验证 BaseTool 抽象基类机制 ---")

    # 1. 验证抽象约束：不能直接实例化基类
    print("\n[测试 1: 验证 BaseTool 抽象类无法直接实例化]")
    try:
        tool = BaseTool("test", "desc")  # type: ignore
        print("❌ 错误：抽象类竟然被实例化了！")
    except TypeError as e:
        print(f"✅ 成功阻止直接实例化抽象类: {e}")

    # 2. 验证子类实现与正常调用
    print("\n[测试 2: 验证子类正常业务执行与工具描述生成]")

    class MockCalculatorTool(BaseTool):
        """测试用模拟计算器工具。"""

        def execute(self, expression: str, **kwargs: Any) -> str:
            # 简单的测试模拟
            if "error" in expression:
                raise ValueError("表达式包含非法字符！")
            return f"计算结果: {eval(expression)}"

    calc = MockCalculatorTool(
        name="mock_calculator",
        description="用于计算基础数学四则运算表达式。",
        parameters_description="expression: 必填，数学表达式字符串，如 '30 * 0.65'。",
    )
    print(f"✅ 成功创建子类工具: {calc}")
    print("📋 工具格式化 Prompt 描述:\n" + calc.get_tool_info())

    # 正常运行测试
    result = calc.run(expression="30 * 0.65")
    assert "19.5" in result
    print(f"✅ 正常执行结果: {result}")

    # 异常容错测试
    print("\n[测试 3: 验证工具异常安全保护 (容错不崩溃)]")
    error_result = calc.run(expression="error_test")
    assert "执行出错" in error_result
    print(f"✅ 异常安全捕获，返回友好错误文本供模型反思: {error_result}")

    print("\n🎉 BaseTool 基类机制全部验证通过！")



