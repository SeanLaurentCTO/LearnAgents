import os
from collections.abc import Callable
from dataclasses import dataclass

from google.genai import types

from gemini_client import build_gemini_client

# 统一工具协议：每个工具都接收一个字符串，并返回一个字符串。
# 这样工具的返回值可以直接作为 ReAct 循环中的 Observation。
ToolFunction = Callable[[str], str]


@dataclass(frozen=True)
class ToolDefinition:
    """工具的资料卡；创建后不可修改。"""

    # name 和 description 会用于告诉 LLM 有哪些工具以及它们的用途。
    name: str
    description: str

    # 保存函数对象本身，而不是提前执行函数。
    function: ToolFunction


class ToolExecutor:
    """统一负责工具的注册、查询和安全执行。"""

    def __init__(self) -> None:
        # 这是一个显式注册表：工具名映射到对应的工具定义。
        # 它具有类似反射的动态分发效果，但不会扫描或反射类的方法。
        self._tools: dict[str, ToolDefinition] = {}

    def register_tool(
            self,
            name: str,
            description: str,
            function: ToolFunction,
    ) -> None:
        """注册一个接收字符串并返回字符串的工具。"""

        # 清除首尾空格，避免 "Echo" 和 " Echo " 被视为两个工具。
        normalized_name = name.strip()
        normalized_description = description.strip()

        if not normalized_name:
            raise ValueError("工具名称不能为空。")

        if not normalized_description:
            raise ValueError("工具描述不能为空。")

        if normalized_name in self._tools:
            raise ValueError(f"工具已注册：{normalized_name}")

        # 注册阶段只保存函数对象；真正调用发生在 execute() 中。
        self._tools[normalized_name] = ToolDefinition(
            name=normalized_name,
            description=normalized_description,
            function=function,
        )

    def get_tool(self, name: str) -> ToolFunction | None:
        """根据名称取得工具函数；工具不存在时返回 None。"""

        # 使用工具名称查询注册表，再取出可调用的 Python 函数。
        definition = self._tools.get(name)
        return definition.function if definition else None

    def get_available_tools(self) -> str:
        """生成供 LLM 阅读的工具名称和描述。"""

        if not self._tools:
            return "当前没有可用工具。"

        # 只把名称和描述展示给 LLM，不暴露 Python 函数对象。
        return "\n".join(
            f"- {tool.name}: {tool.description}"
            for tool in self._tools.values()
        )

    def execute(self, name: str, tool_input: str) -> str:
        """执行指定工具，并将错误转换成可反馈给 LLM 的观察结果。"""

        # 根据字符串名称动态找到对应函数，类似一个命令路由表。
        tool = self.get_tool(name)

        if tool is None:
            return f"错误：未找到名为 '{name}' 的工具。"

        try:
            # 所有注册工具遵循相同协议，因此可以统一调用。
            result = tool(tool_input)
        except Exception as error:
            # 将工具异常转换为文本，供 Agent 作为 Observation 继续推理。
            return f"错误：工具 '{name}' 执行失败：{error}"

        if not isinstance(result, str):
            return (
                f"错误：工具 '{name}' 必须返回字符串，"
                f"实际返回 {type(result).__name__}。"
            )

        return result


def echo(text: str) -> str:
    """用于本地验证工具执行器，不依赖网络或模型。"""

    return f"Echo 工具收到：{text}"


def search(query: str) -> str:
    """使用 Gemini Google Search Grounding 查询互联网信息。"""

    normalized_query = query.strip()

    if not normalized_query:
        raise ValueError("搜索内容不能为空。")

    model = os.getenv("GEMINI_MODEL")
    if not model:
        raise RuntimeError(
            "缺少 GEMINI_MODEL 配置，请检查项目根目录下的 .env 文件。"
        )

    client = build_gemini_client()

    try:
        response = client.models.generate_content(
            model=model,
            contents=(
                "请使用 Google Search 查询下面的问题，并生成适合作为智能体 "
                "Observation 的搜索摘要。\n\n"
                "要求：\n"
                "1. 优先采用官方网站、官方文档和可信新闻来源。\n"
                "2. 明确区分已确认事实、官方预览或规划，以及未经证实的传闻。\n"
                "3. 对重要结论注明来源名称；能够确认网址时同时提供网址。\n"
                "4. 如果没有可靠来源支持某项说法，请明确写出“无法验证”，"
                "不要补充或猜测不存在的信息。\n"
                "5. 只返回与查询直接相关的简洁信息。\n\n"
                f"查询：{normalized_query}"
            ),
            config=types.GenerateContentConfig(
                tools=[
                    types.Tool(
                        google_search=types.GoogleSearch(),
                    )
                ],
                temperature=0,
                # Search Grounding 在服务端执行；关闭的是 SDK 客户端的
                # Automatic Function Calling，不会禁用 Google Search。
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True,
                ),
            ),
        )
    finally:
        client.close()

    result = (response.text or "").strip()
    if not result:
        raise RuntimeError("Google Search 返回了空结果")
    return result


if __name__ == "__main__":
    # 以下代码只验证“注册 → 查找 → 执行”流程，不属于 ReAct 核心循环。
    executor = ToolExecutor()
    # executor.register_tool(
    #     name="Echo",
    #     description="原样返回输入，用于验证工具注册和执行流程。",
    #     function=echo,
    # )
    #
    # print("--- 可用工具 ---")
    # print(executor.get_available_tools())
    #
    # print("\n--- 正常执行 ---")
    # print(executor.execute("Echo", "Hello Agents"))
    #
    # print("\n--- 未知工具 ---")
    # print(executor.execute("Unknown", "Hello Agents"))

    executor.register_tool(
        name="Search",
        description="查询互联网中的实时信息或模型不知道的外部事实。",
        function=search,
    )

    print("\n--- 苹果 ---")
    print(executor.execute("Search", "Apple 最新的手机是哪一款？它的主要卖点是什么？"))

    print("\n--- Google Search Grounding ---")
    print(
        executor.execute(
            "Search",
            "请查询 Google Cloud Vertex AI 最近一项 Gemini 更新，并说明来源。",
        )
    )
