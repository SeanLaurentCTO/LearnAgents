import re
from dataclasses import dataclass

from gemini_client import HelloAgentsLLM, Message

GENERATOR_SYSTEM_PROMPT = """
你是一名认真、专业的内容生成者。

你的任务是根据用户要求生成一份完整初稿。

生成要求：
1. 准确理解用户的目标和约束。
2. 初稿必须完整，不能只提供提纲。
3. 不要编造用户没有提供的具体事实。
4. 不要评价或反思自己的回答。
5. 不要描述你的生成过程。
6. 只输出初稿内容。
""".strip()

REFLECTOR_SYSTEM_PROMPT = """
你是一名严格、客观的内容评审员。

你的任务是根据原始任务检查当前版本的质量，并给出具体、可执行的反馈。

评审维度：
1. 是否完成了原始任务。
2. 是否满足用户提出的所有约束。
3. 是否存在事实错误、逻辑矛盾或无依据内容。
4. 是否遗漏关键信息。
5. 结构和表达是否清晰、准确、简洁。
6. 是否存在需要实际修改的问题。

评审规则：
1. 只评审当前版本，不要直接重写内容。
2. 必须指出具体问题，不能只说“需要改进”。
3. 修改建议必须能够被后续Refiner直接执行。
4. 如果没有需要修改的问题，返回PASS。
5. 只有存在实际影响质量的问题时才返回REVISE。
6. 必须严格使用下面的输出格式。

需要修改时：

Verdict: REVISE
Feedback:
- 具体问题1及修改建议
- 具体问题2及修改建议

无需修改时：

Verdict: PASS
Feedback:
- 当前版本已经满足任务要求，无需继续修改
""".strip()


REFINER_SYSTEM_PROMPT = """
你是一名严谨的内容修订专家。

你的任务是根据原始任务和评审反馈，修改当前版本并生成完整修订稿。

修订要求：
1. 逐项解决评审反馈中指出的问题。
2. 保留当前版本中正确且符合要求的内容。
3. 修订稿必须满足原始任务的全部约束。
4. 不要编造原始任务没有提供的具体事实。
5. 不要评价评审意见，也不要解释修改过程。
6. 只输出完整修订稿，不要输出标题或修改说明。
""".strip()

VERDICT_PATTERN = re.compile(
    r"^Verdict:\s*(PASS|REVISE)\s*$",
    re.MULTILINE | re.IGNORECASE,
)


@dataclass(frozen=True)
class ReflectionFeedback:
    """Reflector返回的结构化评审结果。"""

    verdict: str
    content: str

    @property
    def needs_revision(self) -> bool:
        """判断当前版本是否需要继续修改。"""

        return self.verdict == "REVISE"


@dataclass(frozen=True)
class ReflectionRecord:
    """保存一轮评审与修订，构成任务期间的短期记忆。"""

    iteration: int
    draft: str
    feedback: ReflectionFeedback
    revised_draft: str | None


class Generator:
    """根据用户任务生成Reflection流程中的初稿。"""

    def __init__(self, llm: HelloAgentsLLM) -> None:
        # Generator不创建LLM客户端，只使用外部传入的统一客户端。
        # 因此它不负责关闭客户端。
        self.llm = llm

    def generate(self, task: str) -> str:
        """根据任务生成一份完整初稿。"""

        normalized_task = task.strip()

        if not normalized_task:
            raise ValueError("生成任务不能为空。")

        # System消息规定Generator的职责；
        # User消息提供本次需要完成的具体任务。
        messages: list[Message] = [
            {
                "role": "system",
                "content": GENERATOR_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": normalized_task,
            },
        ]

        print("\n--- Reflection：正在生成初稿 ---")

        # think()仍然是通用模型接口。
        # Generator身份来自GENERATOR_SYSTEM_PROMPT，而不是模型本身发生变化。
        draft = self.llm.think(
            messages=messages,
            temperature=0,
        ).strip()

        if not draft:
            raise RuntimeError("Generator返回了空初稿。")

        return draft


class Reflector:
    """检查当前版本，并生成结构化评审反馈。"""

    def __init__(self, llm: HelloAgentsLLM) -> None:
        # Reflector与Generator可以共享同一个LLM客户端，
        # 但使用不同System Prompt承担不同职责。
        self.llm = llm

    def reflect(
        self,
        task: str,
        draft: str,
    ) -> ReflectionFeedback:
        """根据原始任务评审当前版本。"""

        normalized_task = task.strip()
        normalized_draft = draft.strip()

        if not normalized_task:
            raise ValueError("原始任务不能为空。")

        if not normalized_draft:
            raise ValueError("待评审内容不能为空。")

        review_prompt = (
            f"# 原始任务\n"
            f"{normalized_task}\n\n"
            f"# 当前版本\n"
            f"{normalized_draft}"
        )

        messages: list[Message] = [
            {
                "role": "system",
                "content": REFLECTOR_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": review_prompt,
            },
        ]

        print("\n--- Reflection：正在评审当前版本 ---")

        # Reflector只生成评审意见，不负责修改当前版本。
        raw_feedback = self.llm.think(
            messages=messages,
            temperature=0,
        ).strip()

        if not raw_feedback:
            raise RuntimeError("Reflector返回了空反馈。")

        verdict_match = VERDICT_PATTERN.search(raw_feedback)

        if verdict_match is None:
            raise ValueError(
                "无法解析Reflector结论，预期格式为："
                "Verdict: PASS 或 Verdict: REVISE。\n"
                f"模型原始响应：{raw_feedback}"
            )

        verdict = verdict_match.group(1).upper()

        return ReflectionFeedback(
            verdict=verdict,
            content=raw_feedback,
        )


class Refiner:
    """根据Reflector反馈修订当前版本。"""

    def __init__(self, llm: HelloAgentsLLM) -> None:
        # Refiner与Generator、Reflector共享客户端，
        # 但通过独立System Prompt承担“修订者”职责。
        self.llm = llm

    def refine(
        self,
        task: str,
        draft: str,
        feedback: ReflectionFeedback,
    ) -> str:
        """根据原始任务和评审反馈生成修订稿。"""

        normalized_task = task.strip()
        normalized_draft = draft.strip()

        if not normalized_task:
            raise ValueError("原始任务不能为空。")

        if not normalized_draft:
            raise ValueError("待修订内容不能为空。")

        # PASS表示当前版本已经满足要求，无需产生一次额外模型调用。
        if not feedback.needs_revision:
            return normalized_draft

        # 当前初稿和本轮反馈就是Refiner所需的短期记忆。
        # Gemini不会自动读取前两次调用，必须由Python显式传入。
        revision_prompt = (
            f"# 原始任务\n{normalized_task}\n\n"
            f"# 当前版本\n{normalized_draft}\n\n"
            f"# 评审反馈\n{feedback.content}"
        )

        messages: list[Message] = [
            {
                "role": "system",
                "content": REFINER_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": revision_prompt,
            },
        ]

        print("\n--- Reflection：正在根据反馈修订 ---")

        revised_draft = self.llm.think(
            messages=messages,
            temperature=0,
        ).strip()

        if not revised_draft:
            raise RuntimeError("Refiner返回了空修订稿。")

        return revised_draft


class ReflectionAgent:
    """协调生成、评审和修订，并维护多轮反思短期记忆。"""

    def __init__(
        self,
        llm: HelloAgentsLLM,
        max_iterations: int = 3,
    ) -> None:
        if max_iterations <= 0:
            raise ValueError("max_iterations必须大于0。")

        # 三个组件共享LLM客户端，但通过不同Prompt承担不同职责。
        self.generator = Generator(llm)
        self.reflector = Reflector(llm)
        self.refiner = Refiner(llm)
        self.max_iterations = max_iterations

        # history只保存当前任务的反思轨迹；每次run()开始时都会重置。
        self.history: list[ReflectionRecord] = []

    def run(self, task: str) -> str:
        """执行生成 → 评审 → 修订循环，并返回最终版本。"""

        normalized_task = task.strip()

        if not normalized_task:
            raise ValueError("Reflection任务不能为空。")

        # 防止上一个任务的短期记忆污染当前任务。
        self.history = []
        current_draft = self.generator.generate(normalized_task)

        print("\n--- 初稿 ---")
        print(current_draft)

        for iteration in range(1, self.max_iterations + 1):
            print(
                f"\n=== Reflection 第 {iteration}/"
                f"{self.max_iterations} 轮 ==="
            )

            feedback = self.reflector.reflect(
                task=normalized_task,
                draft=current_draft,
            )

            print("\n--- 评审反馈 ---")
            print(feedback.content)

            if not feedback.needs_revision:
                # PASS轮没有修订稿，但仍保存记录，便于解释停止原因。
                self.history.append(
                    ReflectionRecord(
                        iteration=iteration,
                        draft=current_draft,
                        feedback=feedback,
                        revised_draft=None,
                    )
                )
                print("\n当前版本已通过评审，Reflection提前结束。")
                return current_draft

            revised_draft = self.refiner.refine(
                task=normalized_task,
                draft=current_draft,
                feedback=feedback,
            )

            # 完整保存“旧版本 → 反馈 → 新版本”，形成可检查的短期历史。
            self.history.append(
                ReflectionRecord(
                    iteration=iteration,
                    draft=current_draft,
                    feedback=feedback,
                    revised_draft=revised_draft,
                )
            )

            print("\n--- 修订稿 ---")
            print(revised_draft)

            # 新版本成为下一轮Reflector要检查的当前工作状态。
            current_draft = revised_draft

        # 达到迭代上限时返回最新版本，避免无限反思和持续产生费用。
        print("\n已达到最大反思轮数，返回最新版本。")
        return current_draft


if __name__ == "__main__":
    llm = HelloAgentsLLM()

    try:
        task = (
            "请写一封发给Python课程老师的邮件，"
            "申请将作业提交日期延期到本周五。"
            "原因是最近身体不适。"
            "邮件需要礼貌、简洁，并表达歉意和感谢。"
        )

        agent = ReflectionAgent(
            llm=llm,
            max_iterations=3,
        )
        final_result = agent.run(task)

        print("\n=== Reflection 最终结果 ===")
        print(final_result)

        print("\n--- 短期记忆摘要 ---")
        for record in agent.history:
            revision_status = (
                "已生成修订稿"
                if record.revised_draft is not None
                else "评审通过，未修订"
            )
            print(
                f"第 {record.iteration} 轮："
                f"{record.feedback.verdict}，{revision_status}"
            )

    finally:
        llm.close()
