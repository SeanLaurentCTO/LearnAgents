import ast
import re

from gemini_client import HelloAgentsLLM, Message


PLANNER_SYSTEM_PROMPT = """
你是一名任务规划专家。

你的任务是将用户提出的问题拆分成一组有明确顺序、可以逐步执行的子任务。

规划要求：
1. 只制定计划，不要直接回答用户问题。
2. 每个步骤只描述一个明确任务。
3. 步骤之间必须保持合理的依赖顺序。
4. 不要在计划中提前编造执行结果。
5. 不要加入与问题无关的步骤。
6. 必须输出一个 Python 字符串列表。
7. 除 Python 列表外，不要输出解释、标题或其他内容。

正确格式：

["步骤1", "步骤2", "步骤3"]
""".strip()


EXECUTOR_SYSTEM_PROMPT = """
你是一名任务执行专家。

你的任务是按照已经制定的计划，一次只执行当前指定的步骤。

执行要求：
1. 始终围绕原始问题执行任务。
2. 只执行当前步骤，不要提前执行后续步骤。
3. 如果当前步骤依赖之前的结果，必须使用历史步骤中的结果。
4. 不要修改完整计划。
5. 不要重新执行已经完成的步骤。
6. 返回当前步骤的明确结果。
7. 不要输出与当前步骤无关的内容。
""".strip()


CODE_BLOCK_PATTERN = re.compile(
    r"```(?:python)?\s*(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


class Planner:
    """根据用户问题生成有序执行计划。"""

    def __init__(self, llm: HelloAgentsLLM) -> None:
        self.llm = llm

    def plan(self, question: str) -> list[str]:
        """调用LLM生成计划，并将响应解析成字符串列表。"""

        # 只清理首尾空格，保留用户问题中的专有名称和英文大小写。
        normalized_question = question.strip()

        if not normalized_question:
            raise ValueError("用户问题不能为空。")

        messages: list[Message] = [
            {
                "role": "system",
                "content": PLANNER_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": normalized_question,
            },
        ]

        print("\n--- 正在生成计划 ---")

        # Planner只负责生成步骤，不执行步骤。
        # think()始终返回字符串，结构化计划需要由Python继续解析和验证。
        response = self.llm.think(
            messages=messages,
            temperature=0,
        )

        # Gemini可能直接返回列表文本，也可能将列表包裹在Markdown代码块中。
        plan_text = self._extract_plan_text(response)

        try:
            # literal_eval只解析Python字面量，不会像eval()一样执行任意代码。
            # 这里将字符串形式的列表转换成真正的Python对象。
            parsed_plan = ast.literal_eval(plan_text)
        except (SyntaxError, ValueError) as error:
            raise ValueError(
                "无法把模型响应解析为 Python 列表。\n"
                f"模型原始响应：{response}"
            ) from error

        # literal_eval只能保证语法合法，不能保证结果一定是非空list[str]。
        return self._validate_plan(parsed_plan)

    @staticmethod
    def _extract_plan_text(response: str) -> str:
        """提取Markdown代码块中的计划；没有代码块时使用完整响应。"""

        normalized_response = response.strip()
        code_block = CODE_BLOCK_PATTERN.search(normalized_response)

        if code_block:
            return code_block.group(1).strip()

        return normalized_response

    @staticmethod
    def _validate_plan(parsed_plan: object) -> list[str]:
        """检查计划是否为非空字符串列表。"""

        if not isinstance(parsed_plan, list):
            raise ValueError("计划必须是一个 Python 列表。")

        if not parsed_plan:
            raise ValueError("模型生成了空计划。")

        validated_plan: list[str] = []

        for index, step in enumerate(parsed_plan, start=1):
            if not isinstance(step, str):
                raise ValueError(
                    f"计划第 {index} 步必须是字符串，"
                    f"实际类型是 {type(step).__name__}。"
                )

            normalized_step = step.strip()

            if not normalized_step:
                raise ValueError(f"计划第 {index} 步不能为空。")
            validated_plan.append(normalized_step)

        return validated_plan


class Executor:
    """按照Planner生成的计划逐步执行任务。"""

    def __init__(self, llm: HelloAgentsLLM) -> None:
        self.llm = llm

    def execute(
        self,
        question: str,
        plan: list[str],
    ) -> str:
        """逐步执行计划，并返回最后一步的执行结果。"""

        normalized_question = question.strip()

        if not normalized_question:
            raise ValueError("用户问题不能为空。")

        if not plan:
            raise ValueError("执行计划不能为空。")

        # 完整计划帮助模型理解当前步骤在整个任务中的位置，
        # 但Executor Prompt会限制模型每轮只执行current_step。
        formatted_plan = "\n".join(
            f"{index}. {step}"
            for index, step in enumerate(plan, start=1)
        )

        # 每次think()调用彼此独立，不会自动记住上一步。
        # Python必须显式保存历史，并在后续请求中重新传给模型。
        execution_history: list[str] = []
        final_result = ""

        print("\n--- 正在执行计划 ---")

        for index, current_step in enumerate(plan, start=1):
            print(
                f"\n--- 执行步骤 {index}/{len(plan)} ---\n"
                f"{current_step}"
            )

            history_text = (
                "\n\n".join(execution_history)
                if execution_history
                else "暂无，当前是第一个步骤。"
            )

            # 单步上下文包含四部分：
            # 原始问题防止偏离目标；完整计划说明当前位置；
            # 执行历史提供前置结果；当前步骤限制本轮任务范围。
            step_prompt = (
                f"# 原始问题\n{normalized_question}\n\n"
                f"# 完整计划\n{formatted_plan}\n\n"
                f"# 已完成步骤与结果\n{history_text}\n\n"
                f"# 当前步骤\n{current_step}"
            )

            messages: list[Message] = [
                {
                    "role": "system",
                    "content": EXECUTOR_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": step_prompt,
                },
            ]

            # 仍然使用统一think()接口，但Executor Prompt与Planner Prompt不同，
            # 因此同一个Gemini模型在本轮临时承担“执行者”角色。
            step_result = self.llm.think(
                messages=messages,
                temperature=0,
            ).strip()

            if not step_result:
                raise RuntimeError(f"计划第 {index} 步返回了空结果。")

            print(f"\n步骤结果：{step_result}")

            # 保存当前步骤及结果，使后续步骤能够引用本轮结果。
            # 例如步骤3需要读取步骤2得到的“周二销量为30”。
            execution_history.append(
                f"步骤 {index}：{current_step}\n"
                f"结果：{step_result}"
            )

            # 当前设计约定计划的最后一步负责汇总最终答案，
            # 因此循环结束后最后一个step_result就是Executor返回值。
            final_result = step_result

        return final_result


class PlanAndSolveAgent:
    """协调Planner和Executor，完成先规划、后执行的完整流程。"""

    def __init__(self, llm: HelloAgentsLLM) -> None:
        # 使用组合而不是继承：协调器分别持有规划器和执行器。
        # 两者共享同一个LLM客户端，但通过不同Prompt承担不同角色。
        self.planner = Planner(llm)
        self.executor = Executor(llm)

    def run(self, question: str) -> str:
        """生成计划、执行计划，并返回最终结果。"""

        print("\n=== 开始 Plan-and-Solve 任务 ===")
        print(f"问题：{question}")

        # Plan阶段：把原始问题转换成经过验证的有序计划。
        plan = self.planner.plan(question)

        print("\n--- 生成的计划 ---")

        for index, step in enumerate(plan, start=1):
            print(f"{index}. {step}")

        # Solve阶段：按照计划逐步执行，并显式传递历史状态。
        final_result = self.executor.execute(
            question=question,
            plan=plan,
        )

        print("\n=== Plan-and-Solve 任务完成 ===")

        return final_result


if __name__ == "__main__":
    llm = HelloAgentsLLM()

    try:
        question = (
            "一个水果店周一卖出15个苹果，"
            "周二卖出的数量是周一的两倍，"
            "周三卖出的数量比周二少5个。"
            "请问三天总共卖出多少个苹果？"
        )

        # 调用方只面对统一入口，不需要了解内部规划与执行细节。
        agent = PlanAndSolveAgent(llm)
        final_result = agent.run(question)

        print("\n--- 最终答案 ---")
        print(final_result)

    finally:
        # 谁创建外部资源，谁负责释放；Agent本身不关闭传入的LLM。
        llm.close()






























