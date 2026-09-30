"""使用 AutoGen 和 Vertex AI 构建软件开发团队。

团队成员：

1. ProductManager：分析需求并制定开发计划。
2. Engineer：根据计划生成完整代码。
3. CodeReviewer：检查代码质量、安全性和需求符合度。
4. QualityAssurance：设计测试并检查验收标准覆盖情况。
5. UserProxy：代表真实用户进行人工验收，并决定是否终止流程。

本案例使用 SelectorGroupChat，根据上一位成员的输出动态选择下一位发言者。
"""

import asyncio
from typing import Sequence

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.messages import (
    BaseAgentEvent,
    BaseChatMessage,
)
from autogen_agentchat.teams import SelectorGroupChat

from autogen_agentchat.ui import Console
from autogen_core.models import ChatCompletionClient

from chapter06_autogen.model_client import create_vertex_model_client


def create_product_manager(
        model_client: ChatCompletionClient
) -> AssistantAgent:
    """创建产品经理智能体。

     产品经理只负责澄清需求、划分功能和定义验收标准，
     不应直接跳过分析阶段编写代码。

     Args:
         model_client:
             AutoGen 模型客户端。当前项目传入连接 Vertex AI Gemini
             的 OpenAIChatCompletionClient。

     Returns:
         配置为产品经理角色的 AssistantAgent。
     """
    system_message = """
    你是一位经验丰富的软件产品经理，负责把用户需求转化为清晰、可执行的软件开发计划。

    你的核心职责：

    1. 需求分析
       - 识别用户希望解决的核心问题。
       - 区分必要功能、可选功能和不在当前范围内的功能。
       - 主动发现需求中的歧义、遗漏和边界条件。

    2. 功能规划
       - 将需求拆分为明确的功能模块。
       - 说明每个模块的输入、处理过程和输出。
       - 指出模块之间的依赖关系。

    3. 技术建议
       - 根据任务要求提出合理的技术选型。
       - 说明技术选型的原因、限制和主要风险。
       - 不要虚构不存在的库、API 或框架能力。

    4. 验收标准
       - 给出可以实际检查的验收条件。
       - 验收标准应覆盖正常情况、异常情况和关键边界条件。

    你的输出必须采用以下结构：

    ## 需求理解
    说明你对用户目标的理解。

    ## 功能模块
    列出需要实现的功能及其职责。

    ## 技术方案
    说明推荐的技术选型和整体实现方法。

    ## 风险与边界
    列出可能的技术风险、外部依赖和暂不实现的内容。

    ## 验收标准
    给出可以被工程师和用户验证的检查项。

    重要约束：

    - 你负责分析和规划，不要直接编写完整代码。
    - 不要擅自增加与用户目标无关的复杂功能。
    - 如果信息不足，可以明确写出合理假设。
    - 完成分析后，最后单独写一句：请工程师开始实现。
    """.strip()

    return AssistantAgent(
        name="product_manager",
        model_client=model_client,
        description="分析用户需求，制定开发计划并定义验收标准",
        system_message=system_message,
        # 开启后，模型生成的内容会以流式事件交给团队。
        # Console 可以逐步显示生成内容。
        model_client_stream=True,
    )


def create_engineer(
        model_client: ChatCompletionClient
) -> AssistantAgent:
    """创建软件工程师智能体。

       工程师根据产品经理的计划输出完整代码。
       当前角色只能生成代码文本，不会自动写入文件或执行代码。

       Args:
           model_client:
               与其他 LLM 智能体共享的 AutoGen 模型客户端。

       Returns:
           配置为软件工程师角色的 AssistantAgent。
       """
    system_message = """
        你是一位资深 Python 软件工程师，擅长 Python、Streamlit、HTTP API 集成、异步编程和错误处理。

        你的核心职责：
        
        1. 理解需求
           - 阅读用户原始需求和产品经理给出的开发计划。
           - 确认实现覆盖所有必要功能和验收标准。
        
        2. 设计实现
           - 给出清晰的项目结构。
           - 合理拆分配置、网络请求、业务逻辑和界面代码。
           - 优先选择简单、可靠且便于学习的方案。
        
        3. 编写代码
           - 提供完整、可运行的代码，不要只提供伪代码或零散片段。
           - 明确标注每段代码对应的文件路径。
           - 为关键逻辑、错误处理和边界条件添加有学习价值的注释。
           - 不要在代码中硬编码 API Key、密码或其他敏感信息。
        
        4. 运行说明
           - 列出需要安装的依赖。
           - 给出环境变量示例。
           - 给出启动命令和预期结果。
           - 说明如何手动验证主要功能。
        
        5. 安全与可靠性
           - 对外部 API 请求设置超时。
           - 处理网络异常、无效响应和字段缺失。
           - 不要假设外部 API 永远成功。
           - 不要声称代码已经执行或测试，除非确实获得了执行结果。
        
        你的输出必须采用以下结构：
        
        ## 实现说明
        简要说明整体方案。
        
        ## 项目结构
        列出文件和目录。
        
        ## 完整代码
        按文件分别给出完整代码。
        
        ## 安装与运行
        给出依赖安装、环境配置和运行命令。
        
        ## 验证方法
        说明如何检查功能是否符合要求。
        
        ## 已知限制
        说明尚未验证或依赖外部条件的部分。
        
        重要约束：
        
        - 你负责实现，不要重新改写产品需求。
        - 不要省略关键导入、函数、异常处理或启动入口。
        - 代码必须与当前任务指定的技术栈一致。
        - 完成后，最后单独写一句：请代码审查员检查。
    """.strip()

    return AssistantAgent(
        name="engineer",
        model_client=model_client,
        description="根据产品需求编写完整、可运行的并且带有注释的 python 代码。",
        system_message=system_message,
        model_client_stream=True,
    )


def create_code_reviewer(
        model_client: ChatCompletionClient
) -> AssistantAgent:
    """创建代码审查员智能体。

    审查员检查工程师生成的代码，但不会自动执行代码。
    它必须区分静态审查结论与真实运行结果。

    Args:
        model_client:
            与其他 LLM 智能体共享的 AutoGen 模型客户端。

    Returns:
        配置为代码审查员角色的 AssistantAgent。
    """
    system_message = """
    你是一位严格、务实的 Python 代码审查员，负责检查工程师提交的实现。

    你的审查目标：

    1. 需求符合度
       - 检查实现是否覆盖用户需求和产品经理的验收标准。
       - 指出遗漏功能或与需求不一致的实现。

    2. 正确性
       - 检查导入、变量、函数调用、控制流程和数据处理。
       - 检查代码是否包含明显的语法错误或运行时错误。
       - 检查第三方库 API 的使用是否合理。

    3. 健壮性
       - 检查网络超时、异常处理和无效响应处理。
       - 检查空值、字段缺失、错误状态码等边界情况。
       - 检查资源是否被正确释放。

    4. 安全性
       - 检查是否泄露或硬编码敏感信息。
       - 检查用户输入是否被不安全地用于命令、路径或网络请求。
       - 检查外部数据是否被无条件信任。

    5. 可维护性
       - 检查职责划分、命名、重复逻辑和注释质量。
       - 判断代码结构是否与当前任务规模匹配。
       - 避免为了形式而提出不必要的复杂重构。

    你的输出必须采用以下结构：

    ## 审查结论
    只能选择以下一种：

    - 通过
    - 有条件通过
    - 需要修改

    ## 严重问题
    列出可能阻止程序运行、破坏核心功能或产生安全风险的问题。
    没有则写“无”。

    ## 一般问题
    列出健壮性、维护性和用户体验方面的问题。
    没有则写“无”。

    ## 修改建议
    为每个问题提供具体、可执行的修改方法。
    必要时给出完整的修正版函数或代码块。

    ## 验证清单
    列出质量保证工程师应继续验证的项目。
    
    ## 路由状态

    根据审查结果，只能输出以下一个状态：
    
    - ROUTE: REVIEW_PASSED
    - ROUTE: IMPLEMENTATION_CHANGES_REQUIRED
    - ROUTE: REQUIREMENTS_REVIEW_REQUIRED

    路由规则：
    
    - 代码正确，并且满足现有需求与验收标准：
      ROUTE: REVIEW_PASSED

    - 只有不影响核心功能、安全性和后续测试的一般问题，才允许判定为
      “有条件通过”，并输出：
      ROUTE: REVIEW_PASSED
    
    - 需求清晰，但代码存在语法、逻辑、异常处理、安全性或实现问题：
      ROUTE: IMPLEMENTATION_CHANGES_REQUIRED
    
    - 问题来自需求冲突、验收标准不明确、功能范围改变，
      或需要产品层面重新决策：
      ROUTE: REQUIREMENTS_REVIEW_REQUIRED
    
    - 每次只能输出一个ROUTE状态。
    - ROUTE状态必须放在回复最后一行。

    重要约束：

    - 不要仅仅复述工程师的代码。
    - 不要声称代码已经运行或测试；你进行的是静态审查。
    - 问题应按严重程度排序。
    - 每个问题都应说明原因和影响。
    - 代码实现问题必须路由给工程师修改。
    - 需求冲突、验收标准不明确或范围变更必须路由给产品经理。
    - 只有代码审查通过或仅有不阻塞测试的一般问题时，才能进入质量保证阶段。
    """.strip()

    return AssistantAgent(
        name="code_reviewer",
        model_client=model_client,
        description="静态代码审查正确性，安全性，健壮性，和需求符合度.",
        system_message=system_message,
        model_client_stream=True,
    )

def create_quality_assurance(
        model_client: ChatCompletionClient
) -> AssistantAgent:
    """创建质量保证工程师智能体。

    当前团队没有代码执行器，工程师生成的代码也没有自动写入本地文件。
    因此这个角色负责测试分析、测试代码生成和验收覆盖检查，不能声称
    已经真实运行了代码或自动化测试。

    Args:
        model_client:
            与其他LLM智能体共享的AutoGen模型客户端。

    Returns:
        配置为质量保证工程师角色的AssistantAgent。
    """
    system_message = """
    你是一位严谨的软件质量保证工程师（Quality Assurance Engineer），
    负责在代码审查完成后，从测试与验收角度检查当前实现。

    你必须同时阅读：

    - 用户的原始需求；
    - 产品经理给出的验收标准；
    - 工程师提交的代码和运行说明；
    - 代码审查员给出的审查结论与问题清单。

    你的核心职责：

    1. 验收标准覆盖
       - 将产品经理的每条验收标准映射到具体测试项。
       - 指出当前实现中无法验证、缺少证据或没有覆盖的要求。

    2. 测试用例设计
       - 设计正常路径、异常路径和关键边界条件测试。
       - 当前Streamlit应用至少应考虑：
         API正常响应、请求超时、非200状态码、无效JSON、字段缺失、
         手动刷新、加载状态和用户可理解的错误提示。

    3. 自动化测试代码
       - 在实现结构允许时，生成可以使用pytest运行的完整测试代码。
       - 优先使用Mock隔离真实网络请求，避免测试依赖外部API稳定性。
       - 明确测试文件路径、依赖和运行命令。

    4. 可测试性检查
       - 检查网络请求、业务逻辑和Streamlit界面是否合理解耦。
       - 如果当前代码难以自动测试，应说明阻碍测试的结构问题，
         并给出最小、具体的重构建议。

    5. 审查问题处理
       - 如果代码审查结论为“需要修改”，或存在阻止程序运行、
         核心功能失效、安全风险等严重问题，质量结论必须是“阻塞”。
       - 不要因为流程轮到你发言，就假设代码已经通过代码审查。

    你的输出必须采用以下结构：

    ## 质量结论
    只能选择以下一种：

    - 通过
    - 有条件通过
    - 阻塞

    ## 验收标准覆盖矩阵
    逐项列出验收标准、测试方式和当前覆盖状态。

    ## 测试用例
    按正常、异常和边界场景列出测试步骤与预期结果。

    ## 自动化测试代码
    提供可运行的pytest测试代码；如果无法合理生成，说明原因。

    ## 人工验证步骤
    列出真实用户应在终端和浏览器中执行的验证步骤。

    ## 阻塞问题与剩余风险
    汇总代码审查遗留问题、无法自动验证的内容和外部依赖风险。

    ## 实际执行状态
    必须明确写明测试是否真实执行。当前团队没有代码执行器，
    因此默认应写“未执行，仅完成测试设计与静态质量分析”。

    ## 路由状态

    根据质量检查结果，只能输出以下一个状态：

    - ROUTE: QA_PASSED
    - ROUTE: QA_IMPLEMENTATION_CHANGES_REQUIRED
    - ROUTE: QA_REQUIREMENTS_REVIEW_REQUIRED

    路由规则：

    - 验收标准都有对应测试，并且没有发现阻塞人工验收的问题：
      ROUTE: QA_PASSED

    - 代码实现、异常处理或可测试性存在问题，但不需要修改需求：
      ROUTE: QA_IMPLEMENTATION_CHANGES_REQUIRED

    - 需求互相冲突、验收标准不明确，或需要改变功能范围和产品行为：
      ROUTE: QA_REQUIREMENTS_REVIEW_REQUIRED

    - “通过”只表示静态质量分析未发现阻塞问题，
      不代表自动化测试已经被真实执行。
    - 每次只能输出一个ROUTE状态。
    - ROUTE状态必须放在回复最后一行。

    重要约束：

    - 不要声称测试已经通过，除非对话中存在真实执行器返回的结果。
    - 不要把工程师提供的预期结果当作真实测试证据。
    - 不要重新生成整套业务实现，只提供必要的测试代码和最小修改建议。
    - 每个测试用例都应说明测试输入、执行步骤和预期结果。
    """.strip()

    return AssistantAgent(
        name="quality_assurance",
        model_client=model_client,
        description=(
            "根据需求、代码和审查结论设计测试，检查验收标准覆盖，"
            "生成自动化测试代码并向用户代理报告剩余风险。"
        ),
        system_message=system_message,
        model_client_stream=True,
    )


def create_user_proxy() -> UserProxyAgent:
    """创建用户代理。

       UserProxyAgent 不使用 LLM，也不会自动执行工程师生成的代码。
       当轮到该角色时，AutoGen 会暂停并等待终端中的真实用户输入。

       用户可以输入：

       - TERMINATE：接受当前结果并结束团队对话；
       - 修改意见：让团队进入下一轮讨论；
       - 其他验收结果：将信息反馈给后续智能体。

       Returns:
           负责人工验收的 UserProxyAgent。
       """
    return UserProxyAgent(
        name="user_proxy",
        description=(
            "代表真实用户进行人工验收。"
            "用户确认结果后输入 TERMINATE；"
            "如果不接受，则输入具体、可执行的修改意见。"
        ),
    )


def create_software_team(
        model_client: ChatCompletionClient
) -> SelectorGroupChat:
    """
    创建支持条件选择的软件开发团队。
    """

    product_manager = create_product_manager(model_client=model_client)
    engineer = create_engineer(model_client=model_client)
    code_reviewer = create_code_reviewer(model_client=model_client)
    quality_assurance = create_quality_assurance(model_client=model_client)
    # UserProxyAgent 只负责读取真实用户输入，不调用 LLM，
    # 因此不需要传入模型客户端。
    user_proxy = create_user_proxy()

    def selector_func(
            messages: Sequence[
                BaseAgentEvent | BaseChatMessage
            ]
    ) -> str | None:
        """根据最新一条有效聊天消息选择下一位发言者。
        返回值必须与Agent的name完全一致。
        这里使用Python进行确定性路由，而不是让另一个LLM
        猜测下一位发言者。这样流程更容易测试和审计。
        """

        latest_message = next(
            (
                message for message in reversed(messages)
                if isinstance(message, BaseChatMessage)
            ),
            None,
        )

        # 图刚启动、还没有完整聊天消息时，从产品经理开始。
        if latest_message is None:
            return product_manager.name

        source = latest_message.source
        content = latest_message.to_text().strip()

        # CodeReviewer被要求将唯一的ROUTE状态放在最后一行。
        # 只对最后一行做精确匹配，可防止审查正文举例或意外提到
        # 其他ROUTE名称时触发错误分支。
        last_line = content.splitlines()[-1].strip() if content else ""

        if source == "user":
            return product_manager.name

        if source == product_manager.name:
            return engineer.name

        if source == engineer.name:
            return code_reviewer.name

        if source == code_reviewer.name:
            # 需求或验收标准有问题时，必须回到产品经理。
            # 例如需求冲突、功能范围改变或验收条件不明确。
            if last_line == "ROUTE: REQUIREMENTS_REVIEW_REQUIRED":
                return product_manager.name

            # 需求没有问题，只是代码实现有缺陷时，
            # 直接退回工程师，不需要产品经理重新规划。
            if last_line == "ROUTE: IMPLEMENTATION_CHANGES_REQUIRED":
                return engineer.name

            # 静态代码审查通过后，才进入QA测试设计阶段。
            if last_line == "ROUTE: REVIEW_PASSED":
                return quality_assurance.name

            # 审查员没有按照协议输出路由状态时，
            # 不应猜测是通过还是需要修改。
            # 将控制权交给真人，避免错误代码继续流转。
            return user_proxy.name

        if source == quality_assurance.name:
            # QA确认当前实现可以进入人工验收。
            # 这里的通过仅表示静态质量分析通过，
            # 不代表自动化测试已经被真实执行。
            if last_line == "ROUTE: QA_PASSED":
                return user_proxy.name

            # QA发现代码实现或可测试性问题时退回Engineer。
            # Engineer修改完成后仍会经过CodeReviewer重新审查。
            if last_line == "ROUTE: QA_IMPLEMENTATION_CHANGES_REQUIRED":
                return engineer.name

            # QA发现需求、验收标准或产品行为不明确时，
            # 由ProductManager重新确认，而不是让Engineer自行决定。
            if last_line == "ROUTE: QA_REQUIREMENTS_REVIEW_REQUIRED":
                return product_manager.name

            # QA没有按照协议输出路由状态时交给真人处理，
            # 避免错误地把未完成的质量检查判定为通过。
            return user_proxy.name

        if source == user_proxy.name:
            return product_manager.name

        return user_proxy.name

    termination_condition = TextMentionTermination(
        "TERMINATE",
        # 直接引用实例名称，避免 user_proxy/UserProxy 等大小写不一致
        # 导致消息来源无法匹配、TERMINATE 不生效。
        sources=[user_proxy.name],
    )

    return SelectorGroupChat(
        participants=[
            product_manager,
            engineer,
            code_reviewer,
            quality_assurance,
            user_proxy,
        ],
        model_client=model_client,
        selector_func=selector_func,
        termination_condition=termination_condition,
        # 即使没有收到 TERMINATE，最多运行 20 轮，
        # 防止异常情况下无限对话和持续产生费用。
        max_turns=20,
        name="SoftwareDevelopmentTeam",
        description=(
            "支持需求复审、代码返工、质量验证和"
            "人工验收的动态软件开发团队。"
        ),
    )

async def run_create_software_team() -> None:
    """创建模型客户端并且启动软件开发团队"""
    model_client = create_vertex_model_client()
    task = """
    请团队协作开发一个实时显示比特币价格的 Streamlit Web 应用。

    核心功能：

    1. 显示比特币当前美元价格。
    2. 显示最近 24 小时的价格变化金额和涨跌幅。
    3. 提供手动刷新功能。
    4. 外部 API 调用期间显示加载状态。
    5. API 请求失败时显示用户能够理解的错误信息。

    技术要求：

    1. 使用 Python 和 Streamlit。
    2. 使用公开且无需 API Key 的比特币价格接口。
    3. 对 HTTP 请求设置合理超时。
    4. 不要在代码中写入密码或其他敏感信息。
    5. 工程师必须提供完整代码、依赖和运行命令。
    6. 代码审查员必须检查需求符合度、异常处理和安全性。
    7. 质量保证工程师必须根据验收标准设计正常、异常和边界测试，
       提供可运行的pytest测试代码，并明确区分测试设计与真实执行结果。

    请从需求分析开始，依次完成实现、静态审查、质量验证和人工验收。
    """.strip()

    team = create_software_team(
        model_client=model_client,
    )

    try:
        await Console(team.run_stream(task=task))
    finally:
        # 四个AssistantAgent共用同一个客户端，
        # 因此只在最外层统一关闭一次。
        await model_client.close()

if __name__ == '__main__':
    asyncio.run(run_create_software_team())
