"""框架核心异常体系定义。

通过继承自定义基类，将大模型调用、工具执行、配置缺失等不同类型的错误结构化，
避免底层 SDK 原始异常直接击穿到应用层。
"""

class HelloAgentsException(Exception):

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return f"[{self.__class__.__name__}] {self.message}"

class ConfigurationException(HelloAgentsException):
    """配置相关异常，例如缺少必要的环境变量（如 GOOGLE_CLOUD_PROJECT）或参数不合法。"""

    pass


class LLMCallException(HelloAgentsException):
    """大模型调用异常，封装网络中断、API 配额不足、鉴权失败或模型返回空内容等错误。"""

    def __init__(self, message: str, original_error: Exception | None = None) -> None:
        super().__init__(message)
        # 保留底层原始异常链，便于调试和查看堆栈
        self.original_error = original_error


class ToolExecutionException(HelloAgentsException):
    """工具执行异常，工具参数解析失败或工具内部执行崩溃时抛出。"""

    def __init__(self, tool_name: str, message: str) -> None:
        super().__init__(f"工具 '{tool_name}' 执行失败: {message}")
        self.tool_name = tool_name


class MaxStepsExceededException(HelloAgentsException):
    """智能体循环保护异常，当 ReAct 或反思循环超过预设的最大步数时抛出，防止死循环。"""

    def __init__(self, steps: int) -> None:
        super().__init__(f"Agent 执行超过最大限制步数 ({steps} 步)，已强制终止。")
        self.steps = steps











































































