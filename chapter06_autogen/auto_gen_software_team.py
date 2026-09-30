"""使用 AutoGen 和 Vertex AI 构建软件开发团队。

团队成员：

1. ProductManager：分析需求并制定开发计划。
2. Engineer：根据计划生成完整代码。
3. CodeReviewer：检查代码质量、安全性和需求符合度。
4. UserProxy：代表真实用户进行人工验收，并决定是否终止流程。

本案例使用 RoundRobinGroupChat，成员按照固定顺序依次发言。
"""

import asyncio

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
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
    列出用户代理应实际检查的项目。

    重要约束：

    - 不要仅仅复述工程师的代码。
    - 不要声称代码已经运行或测试；你进行的是静态审查。
    - 问题应按严重程度排序。
    - 每个问题都应说明原因和影响。
    - 即使代码存在问题，也要把问题清楚交给用户代理决定下一步。
    - 完成后，最后单独写一句：代码审查完成，请用户代理验收。
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
    # TODO 加一个新的身份完成 质量检验
    pass


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
) -> RoundRobinGroupChat:
    """创建按固定顺序协作的软件开发团队。

    发言顺序：

        ProductManager
            ↓
        Engineer
            ↓
        CodeReviewer
            ↓
        Quality Assurance
            ↓
        UserProxy
            ↓
        若未终止，则重新回到 ProductManager

    Args:
        model_client:
            ProductManager、Engineer 和 CodeReviewer 共享的模型客户端。

    Returns:
        配置完成的 RoundRobinGroupChat。
    """

    product_manager = create_product_manager(model_client=model_client)
    engineer = create_engineer(model_client=model_client)
    code_reviewer = create_code_reviewer(model_client=model_client)
    quality_assurance = create_quality_assurance(model_client=model_client)
    # UserProxyAgent 只负责读取真实用户输入，不调用 LLM，
    # 因此不需要传入模型客户端。
    user_proxy = create_user_proxy()

    termination_condition = TextMentionTermination(
        "TERMINATE",
        # 直接引用实例名称，避免 user_proxy/UserProxy 等大小写不一致
        # 导致消息来源无法匹配、TERMINATE 不生效。
        sources=[user_proxy.name],
    )

    return RoundRobinGroupChat(
        participants=[
            product_manager,
            engineer,
            code_reviewer,
            quality_assurance,
            user_proxy,
        ],
        termination_condition=termination_condition,
        # 即使没有收到 TERMINATE，最多运行 20 轮，
        # 防止异常情况下无限对话和持续产生费用。
        max_turns=20,
        name="SoftwareDevelopmentTeam",
        description="需求分析、编码、代码审查和人工验收软件开发团队。",
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

    请从需求分析开始，依次完成实现、审查和人工验收。
    """.strip()

    team = create_software_team(
        model_client=model_client,
    )

    try:
        await Console(team.run_stream(task=task))
    finally:
        # 三个 AssistantAgent 共用同一个客户端，
        # 因此只在最外层统一关闭一次。
        await model_client.close()

if __name__ == '__main__':
    asyncio.run(run_create_software_team())
