"""ReAct (Reasoning + Acting) 智能体范式实现。

继承自 Agent 基类，结合 ToolRegistry 工具注册中心，
实现经典的 Thought（思考）-> Action（行动）-> Observation（观察）循环。
具备工具自主调度、单步推理、正则解析以及最大执行步数防死循环保护机制。
"""

import re
from typing import Any, List, Optional, Tuple

from chapter07_my_framework.core.agent import Agent
from chapter07_my_framework.core.config import Config
from chapter07_my_framework.core.llm import HelloAgentsLLM
from chapter07_my_framework.core.message import Message
from chapter07_my_framework.tools.registry import ToolRegistry

# 标准 ReAct 引导提示词模板
DEFAULT_REACT_PROMPT = """你是一个具备严谨逻辑推理与外部工具调用能力的专业 AI 投资助手。
你可以通过“思考（Thought）”分析当前现状，决定是否需要调用“工具（Action）”获取外部事实，并根据工具返回的结果（Observation）继续推演，直至给出最终答案。

## 可用工具列表
{tools_description}

## 工作规范与格式要求
你必须严格按照以下格式进行回应，【每次回复只能执行一个单一步骤，绝不要伪造 Observation】：

Thought: 分析当前需要什么信息、计算什么指标或下一步该采取什么行动。
Action: 选择一个行动，格式必须是以下二者之一：
- `tool_name[参数]`：调用上述工具列表中的某个具体工具（例如 `currency_converter[amount=30, to_currency=USD]` 或 `market_quote[SPY]`）。
- `Finish[最终答复]`：当你已经收集齐足够的数据和事实，能够给出完整、专业的最终答复时使用。

## 重要纪律
1. 每次回复必须且只能包含一组 Thought 和 Action。
2. 工具调用格式必须严格遵循 `tool_name[参数]`，工具名必须与可用工具列表完全一致。
3. 严禁自行编造 Observation！工具执行结果会由系统在下一轮提供给你。
4. 只有当你掌握了全部客观数据时，才使用 `Finish[...]` 输出最终建议。

## 当前任务
**用户提问/任务目标:** {question}

## 历史执行轨迹
{history}

现在开始你的第 1 步思考与行动：
"""


class ReActAgent(Agent):
    """推理协同（ReAct）智能体"""

    def __init__(
            self,
            name: str,
            llm: HelloAgentsLLM,
            tool_registry: ToolRegistry,
            system_prompt: Optional[str] = None,
            config: Optional[Config] = None,
            max_steps: int = 5,
            custom_prompt: Optional[str] = None,
    ) -> None:
        """初始化 ReActAgent
        Args:
            name: 智能体名称。
            llm: 模型通信客户端。
            tool_registry: 工具注册中心实例。
            system_prompt: 角色定位系统提示词。
            config: 运行时配置。
            max_steps: 思考-行动循环的最大步数安全保护（默认 5 步）。
            custom_prompt: 可选的自定义 ReAct 提示词模板。
        """
        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt,
            config=config,
        )
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.prompt_template = custom_prompt or DEFAULT_REACT_PROMPT

        # 内部追踪当前单次任务的动态轨迹（Thought / Action / Observation 序列）
        self._current_trace: List[str] = []

    def run(self, input_text, **kwargs: Any) -> str:
        """运行 ReAct 动态思考-行动决策循环
        Args:
            input_text: 用户输入的投资任务或问题。
            **kwargs: 传给底层的模型参数
        Returns:
            str: 经过事实检验后的最终综合研报/答复。
        """
        # 记录用户需求实现长期记忆
        self.add_message(Message.user(input_text))
        # 清空当前任务的执行轨迹
        self._current_trace = []
        current_step = 0

        print(f"\n🚀 [{self.name}] 启动 ReAct 决策循环，目标: '{input_text}'")

        while current_step < self.max_steps:
            current_step += 1
            print(f"\n🔄 --- 循环轮次 [{current_step}/{self.max_steps}] ---")

            # 组装 prompt 注入循环清单，当前问题和历史轨迹
            tools_desc = self.tool_registry.get_tools_description()
            history_str = "\n".join(self._current_trace) if self._current_trace else "暂无历史轨迹"

            prompt_content = self.prompt_template.format(
                tools_description=tools_desc,
                question=input_text,
                history=history_str,
            )

            # 构造上下文消息
            messages: List[Message] = []
            if self.system_prompt and self.system_prompt.strip():
                messages.append(Message.system(self.system_prompt.strip()))
            messages.append(Message.user(prompt_content))

            # 调用大模型单步推理，
            step_response = self.llm.invoke(messages, temperature=0.2, **kwargs)

            # C. 解析大模型输出中的 Thought 和 Action
            thought, action = self._parse_output(step_response)
            print(f"💭 Thought: {thought}")
            print(f"⚡ Action : {action}")

            # 将本轮模型回复记录到轨迹内容中
            if thought:
                self._current_trace.append(f"Thought: {thought}")

            # 分支判断：模型决定结束任务
            if action and action.startswith("Finish"):
                final_answer = self._parse_action_parameter(action)
                print(f"🏁 [{self.name}] 决策完成，生成最终方案！")

                # 将最终成果记录入记忆并返回
                self.add_message(Message.assistant(final_answer))
                return final_answer

            # 分支判断：决定调用外部工具 tool_name
            if action:
                self._current_trace.append(f"Action: {action}")
                tool_name, tool_params = self._extract_tool_and_params(action)

                # 通过注册表智能调度真实工具
                print(f"🔧 正在执行工具: '{tool_name}'，参数: '{tool_params}'")
                observation = self.tool_registry.execute_tool(tool_name, tool_params)
                print(f"👁️ Observation:\n{observation}\n")

                # 将工具返回的结果添加到 observation 内容，共下一步思考使用
                self._current_trace.append(f"Observation: {observation}")
            else:
                # 容错：如果模型既没调工具也没说 Finish，提示模型给出明确行动
                warn_msg = "Observation: 系统未检测到有效的 Action 格式，请按照 `tool_name[参数]` 或 `Finish[最终答复]` 给出行动。"
                self._current_trace.append(warn_msg)
        # 超过规定 step 的熔断保护
        fallback_msg = (
                f"⚠️ 决策中断：已达到最大允许步数 ({self.max_steps} 步)，但尚未完全得出结论。\n"
                f"已收集到的最新线索如下：\n" + "\n".join(self._current_trace[-4:])
        )

        self.add_message(Message.assistant(fallback_msg))
        return fallback_msg

    def _parse_output(self, text: str) -> Tuple[str, str]:
        """从模型回复文本中用正则表达式提取 Thought 和 Action。"""
        thought = ""
        action = ""

        # 匹配 Thought：提取 Thought: 到 Action: 之间的全部文本
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL | re.IGNORECASE)
        if thought_match:
            thought = thought_match.group(1).strip()

        # 匹配 Action：注意使用 DOTALL 捕获 Action 之后的整段内容（尤其当 Finish[...] 包含长文本换行时）
        action_match = re.search(r"Action:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
        if action_match:
            action = action_match.group(1).strip()

        # 如果没有严格按标签输出，做容错回退
        if not thought and not action:
            thought = text.strip()

        return thought, action

    def _parse_action_parameter(self, finish_text: str) -> str:
        """从 `Finish[final_answer]` 提取方括号内部的最终回答。"""
        match = re.match(r"Finish[\[\(](.*)[\]\)]", finish_text.strip(), re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        # 如果格式有微小变形，尝试剥离前缀
        return finish_text.replace("Finish", "").strip("[](): ")

    def _extract_tool_and_params(self, action_text: str) -> Tuple[str, str]:
        """从 `tool_name[params]` 文本中提取工具名称与入参内容。"""
        # 清洗可能包裹在两端的 Markdown 代码标记 `（反引号）
        cleaned_action = action_text.strip().strip("`").strip()

        match = re.match(r"([a-zA-Z0-9_\-\.]+)[\[\(](.*)[\]\)]", cleaned_action, re.DOTALL)
        if match:
            tool_name = match.group(1).strip()
            params = match.group(2).strip()
            return tool_name, params
        return cleaned_action, ""


if __name__ == "__main__":
    # --- 模块自测逻辑：装配三件金融武器，验证 30 AUD 实盘智能决策闭环 ---
    print("--- 正在测试 ReActAgent 投资顾问综合决策闭环 ---")

    # 1. 组装武器库 (ToolRegistry)
    registry = ToolRegistry()

    # 引入我们亲手完成的三大核心工具
    from chapter07_my_framework.tools.builtin.investment.currency import CurrencyConverterTool
    from chapter07_my_framework.tools.builtin.investment.market import MarketQuoteTool
    from chapter07_my_framework.tools.builtin.search import SearchTool

    registry.register(CurrencyConverterTool())
    registry.register(MarketQuoteTool())
    registry.register(SearchTool())

    print(f"✅ 成功挂载 3 大金融工具: {registry.list_tools()}")

    # 2. 实例化 ReActAgent
    llm = HelloAgentsLLM()
    advisor = ReActAgent(
        name="资深智能投顾",
        llm=llm,
        tool_registry=registry,
        system_prompt=(
            "你是一位兼具宏观视野与微观风控意识的资深量化投顾专家。"
            "面对小额资金（如 30 AUD），你必须格外关注换汇磨损、交易佣金，并基于客观市场行情提供务实、审慎的配置策略。"
        ),
        max_steps=5,
    )

    # 3. 发起真实的复杂投资问答测试（考验 Agent 连续调用汇率、行情和给出策略的能力）
    complex_question = (
        "我手头只有 30 澳元闲置资金，请帮我分析：如果现在换成美元，我的实际净购买力是多少？"
        "同时看一下美股大盘标普500 (SPY) 的最新现价。面对我这笔小资金，你建议我应该怎么配置？"
    )

    print(f"\n👤 用户提问:\n{complex_question}")
    final_advice = advisor.run(complex_question)

    print("\n" + "=" * 50)
    print("🎯 智能投顾最终给出的研报建议:")
    print("=" * 50)
    print(final_advice)

    llm.close()