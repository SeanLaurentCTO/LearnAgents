import re
from dataclasses import dataclass

from chapter04_classic_patterns.tools import ToolExecutor, search
from gemini_client import HelloAgentsLLM, Message


REACT_SYSTEM_PROMPT = """
你是一个使用 ReAct 方法解决问题的智能体。

可用工具：
{available_tools}

你必须严格遵循下面的格式，每轮只输出一个 Thought 和一个 Action：

Thought: 分析当前信息，并说明下一步应该做什么。
Action: 工具名称[工具输入]

如果需要查询信息，请使用：

Thought: 说明为什么需要查询。
Action: Search[需要查询的内容]

如果已经可以回答用户，请使用：

Thought: 说明为什么已经可以回答。
Action: Finish[最终答案]

规则：
1. 只能使用“可用工具”中列出的工具。
2. 每轮只能生成一个 Action。
3. 不要自己编造 Observation，工具结果会由程序提供。
4. 工具返回错误时，根据错误调整下一步行动。
5. 最终答案必须放在 Finish[...] 中。
""".strip()


@dataclass(frozen=True)
class ParsedAction:
    """从模型响应中解析出的 ReAct 行动。"""

    name: str
    tool_input: str


ACTION_PATTERN = re.compile(
    r"Action:\s*([A-Za-z_][A-Za-z0-9_]*)\s*\[(.*)]\s*$",
    re.DOTALL,
)


def parse_action(response: str) -> ParsedAction:
    """从模型响应中提取 Action 名称和输入。"""

    # 响应以 Thought 开头，因此应在整段响应中搜索 Action，而不是从开头匹配。
    match = ACTION_PATTERN.search(response.strip())

    if match is None:
        raise ValueError(
            "无法解析模型行动，预期格式为："
            "Action: 工具名称[工具输入]"
        )

    name = match.group(1).strip()
    tool_input = match.group(2).strip()

    if not tool_input:
        raise ValueError(f"行动 '{name}' 的输入不能为空。")

    return ParsedAction(
        name=name,
        tool_input=tool_input,
    )


class ReActAgent:
    """按照 Thought → Action → Observation 循环解决问题。"""

    def __init__(
        self,
        llm: HelloAgentsLLM,
        tool_executor: ToolExecutor,
        max_steps: int = 5,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps 必须大于 0。")

        self.llm = llm
        self.tool_executor = tool_executor
        self.max_steps = max_steps

    def run(self, question: str) -> str:
        """执行 ReAct 循环，并返回 Finish 中的最终答案。"""

        normalized_question = question.strip()

        if not normalized_question:
            raise ValueError("用户问题不能为空。")

        system_prompt = REACT_SYSTEM_PROMPT.format(
            available_tools=self.tool_executor.get_available_tools(),
        )

        messages: list[Message] = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": normalized_question,
            },
        ]

        for step in range(1, self.max_steps + 1):
            print(f"\n--- ReAct 第 {step}/{self.max_steps} 步 ---")
            response = self.llm.think(messages, temperature=0)

            # 保留模型本轮响应，让下一轮能够看到完整的推理轨迹。
            messages.append(
                {
                    "role": "assistant",
                    "content": response,
                }
            )

            try:
                action = parse_action(response)
            except ValueError as error:
                # 格式错误时不执行工具，而是把错误反馈给模型并允许它重试。
                format_observation = f"格式错误：{error}"
                print(f"Observation: {format_observation}")
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Observation: {format_observation}\n"
                            "请严格按照规定格式重新生成一个 Action。"
                        ),
                    }
                )
                continue

            if action.name.casefold() == "finish":
                # Finish 是 ReAct 的正常结束信号，不需要交给 ToolExecutor。
                # 方括号中的内容就是 Agent 最终返回给用户的答案。
                return action.tool_input

            observation = self.tool_executor.execute(
                name=action.name,
                tool_input=action.tool_input,
            )
            print(f"Observation: {observation}")

            # 工具结果作为新的用户消息反馈给模型，驱动下一轮 Thought。
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Observation: {observation}\n"
                        "请根据这个观察结果继续推理，并生成下一个 Action。"
                    ),
                }
            )

        # 循环走到这里，说明已经用完 max_steps，但模型始终没有返回 Finish。
        # 主动终止可以防止 Agent 因重复搜索或格式错误而无限运行。
        raise RuntimeError(
            f"ReAct Agent 在 {self.max_steps} 步内没有生成 Finish 行动。"
        )


if __name__ == "__main__":
    # 第一步：创建工具注册表，并注册 ReAct 可以调用的 Search 工具。
    executor = ToolExecutor()
    executor.register_tool(
        name="Search",
        description=(
            "查询互联网中的实时信息或模型不知道的外部事实；"
            "输入应当是明确、具体的搜索问题。"
        ),
        function=search,
    )

    # 第二步：创建 LLM 和 ReAct Agent。最多允许执行 5 轮推理。
    llm = HelloAgentsLLM()
    agent = ReActAgent(
        llm=llm,
        tool_executor=executor,
        max_steps=5,
    )

    try:
        # 第三步：运行完整的 Thought → Action → Observation 循环。
        final_answer = agent.run(
            "请查询 Google Cloud Vertex AI 最近一项 Gemini 更新，"
            "并明确区分已发布信息和未经证实的传闻。"
        )

        print("\n--- ReAct 最终答案 ---")
        print(final_answer)
    finally:
        # ReActAgent 不拥有客户端生命周期，由创建 llm 的调用方负责关闭。
        llm.close()

