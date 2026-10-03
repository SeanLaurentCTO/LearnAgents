"""工具注册中心 (ToolRegistry) 实现。

负责全局或局部工具的生命周期管理：
1. 工具注册与注销 (register / unregister)；
2. 动态查找与存在性校验 (get_tool / has_tool)；
3. 自动生成注入大模型系统提示词的工具描述文本 (get_tools_description)；
4. 智能参数适配与安全执行 (execute_tool)。
"""

from typing import Any, Dict, List, Optional, Union

from chapter07_my_framework.tools.base import BaseTool


class ToolRegistry:
    """工具注册中心管理器。
    充当智能体调度具体工具的路由中枢。
    """

    def __init__(self) -> None:
        """初始化注册容器"""
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """向注册表中注册一个工具实例。
        Args:
            tool: 继承自 BaseTool 的工具对象。

        Raises:
            ValueError: 当传入对象不是 BaseTool 实例或名称为空时抛出。
        """

        if not isinstance(tool, BaseTool):
            raise ValueError(f"被注册的对象必须是 BaseTool 的子类实例，当前收到: {type(tool)}")

        if not tool.name:
            raise ValueError("被注册工具的 name 属性不能为空。")

        if tool.name in self._tools:
            # 允许覆盖，但输出提示
            print(f"⚠️ [ToolRegistry] 工具 '{tool.name}' 已存在，将被新实例覆盖。")

        self._tools[tool.name] = tool


    def unregister(self, tool_name: str) -> bool:
        """从注册表中注销指定工具
        Args:
            tool_name: 要移除的工具名称
        Returns:
            bool: 成功注销返回 True；若工具本来就不存在则返回 False。
        """

        if tool_name in self._tools:
            del self._tools[tool_name]
            return True
        return False


    def has_tool(self, tool_name: str) -> bool:
        return tool_name in self._tools


    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """根据工具名称获取工具实例
        Args:
            tool_name: 工具名称
        Returns:
            Optional[BaseTool]: 找到则返回工具实例，未找到返回 None。
        """

        if tool_name in self._tools:
            return self._tools[tool_name]
        return None


    def list_tools(self) -> List[str]:
        """获取当前已注册的所有工具名称列表。"""
        return list(self._tools.keys())


    def get_tools_description(self) -> str:
        """生成供 Prompt 注入的格式化工具描述文本块。

        大模型（如 ReAct 或 SimpleAgent）就是阅读这段文本来理解当前有哪些可用武器。

        Returns:
            str: 格式化的 Markdown 字符串；如果没有注册任何工具，返回友好的占位提示。
        """
        if not self._tools:
            return "暂无可用的外部工具。"

        descriptions: List[str] = []
        for tool in self._tools.values():
            descriptions.append(tool.get_tool_info())

        return "\n\n".join(descriptions)


    def execute_tool(
            self,
            tool_name: str,
            parameters: Union[str, Dict[str, Any], None] = None,
    ) -> str:
        """智能调度并执行指定名称的工具
        支持自动适配纯文本入参或结构化字典入参
        Args:
            tool_name: 要执行的目标工具名称。
            parameters: 大模型传递的入参。可以是字典、单个字符串或 None
        Returns:
            str: 工具执行后的 Observation 结果字符串。
        """

        tool = self.get_tool(tool_name)

        if not tool:
            return  f"❌ 调度失败：未找到名为 '{tool_name}' 的工具。当前可用工具列表: {self.list_tools()}"

        try:
            if isinstance(parameters, dict):
                return tool.run(**parameters)
            elif isinstance(parameters, str):
                params_str = parameters.strip()
                # 常见单参数工具推断映射
                if tool_name in {"calculator", "math"}:
                    return tool.run(expression=params_str)
                elif tool_name in {"search", "web_search"}:
                    return tool.run(query=params_str)
                elif tool_name in {"currency_converter", "fx"}:
                    return tool.run(query=params_str)
                else:
                    # 默认以 input 作为参数名调用
                    return tool.run(input=params_str)
            elif parameters is None:
                # 无参数，空参执行
                return tool.run()
            else:
                return tool.run(input=str(parameters))
        except Exception as e:
            return f"❌ 工具 '{tool_name}' 调度过程中发生异常: {str(e)}"

    def __len__(self) -> int:
        """返回已注册的工具总数。"""
        return len(self._tools)

    def __str__(self) -> str:
        return f"ToolRegistry(registered_tools={self.list_tools()})"



if __name__ == "__main__":
    # --- 模块自测逻辑：验证注册表的完整生命周期与智能执行 ---
    print("--- 正在验证 ToolRegistry 工具注册中心 ---")

    # 1. 初始化注册表
    registry = ToolRegistry()
    assert len(registry) == 0

    # 2. 引入 BaseTool 中的 Mock 工具进行测试
    from chapter07_my_framework.tools.base import BaseTool

    class EchoTool(BaseTool):
        """测试用回显工具"""

        def execute(self, message: str, **kwargs: Any) -> str:
            return f"Echo: {message}"

    echo = EchoTool(
        name="echo",
        description="将输入的文本原样回显。",
        parameters_description="message: 必填，待回显的字符串内容。",
    )

    # 3. 注册工具测试
    registry.register(echo)
    assert registry.has_tool("echo")
    assert len(registry) == 1
    print(f"✅ 成功注册工具: {registry}")

    # 4. 验证 Prompt 描述生成
    print("\n📋 注册表自动生成的 Prompt 工具描述:")
    print(registry.get_tools_description())

    # 5. 验证工具调度执行（字典传参）
    res1 = registry.execute_tool("echo", {"message": "Hello Agent!"})
    assert res1 == "Echo: Hello Agent!"
    print(f"\n✅ 字典传参调度成功: {res1}")

    # 6. 验证容错：调度不存在的工具
    res_err = registry.execute_tool("unknown_tool", "test")
    assert "未找到名为" in res_err
    print(f"✅ 未知工具安全拦截: {res_err}")

    # 7. 注销工具测试
    assert registry.unregister("echo") is True
    assert len(registry) == 0
    print("✅ 注销工具成功，当前注册表已清空！")

    print("\n🎉 ToolRegistry 工具注册中心全部验证通过！")