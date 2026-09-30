"""测试AutoGen软件团队的确定性角色路由。"""

from unittest.mock import Mock

import pytest
from autogen_agentchat.messages import TextMessage

from chapter06_autogen.auto_gen_software_team import create_software_team


@pytest.fixture
def selector_func():
    """创建团队并取得当前SelectorGroupChat使用的路由函数。"""
    team = create_software_team(Mock())

    # 当前selector_func定义在create_software_team内部，因此通过团队保存的
    # 私有属性测试。若未来路由规则继续变复杂，可将它提取为独立公共函数。
    return team._selector_func


@pytest.mark.parametrize(
    ("route", "expected_agent"),
    [
        ("ROUTE: REVIEW_PASSED", "quality_assurance"),
        ("ROUTE: IMPLEMENTATION_CHANGES_REQUIRED", "engineer"),
        ("ROUTE: REQUIREMENTS_REVIEW_REQUIRED", "product_manager"),
    ],
)
def test_code_reviewer_routes(selector_func, route, expected_agent):
    """验证CodeReviewer的三种主要路由。"""
    message = TextMessage(
        source="code_reviewer",
        content=f"## 审查结论\n测试结论\n\n{route}",
    )

    assert selector_func([message]) == expected_agent


def test_route_must_be_on_last_line(selector_func):
    """正文中的ROUTE示例不能被误判为最终路由状态。"""
    message = TextMessage(
        source="code_reviewer",
        content="ROUTE: REVIEW_PASSED\n最终结论尚未给出。",
    )

    assert selector_func([message]) == "user_proxy"


def test_missing_route_falls_back_to_user_proxy(selector_func):
    """缺少ROUTE时交给真人处理，不能擅自假设审查通过。"""
    message = TextMessage(
        source="code_reviewer",
        content="## 审查结论\n需要修改，但没有输出路由状态。",
    )

    assert selector_func([message]) == "user_proxy"


def test_product_manager_routes_to_engineer(selector_func):
    message = TextMessage(
        source="product_manager",
        content="需求和验收标准已经确认。",
    )

    assert selector_func([message]) == "engineer"


def test_engineer_routes_to_code_reviewer(selector_func):
    message = TextMessage(
        source="engineer",
        content="代码实现完成，请进行静态审查。",
    )

    assert selector_func([message]) == "code_reviewer"


def test_initial_user_task_routes_to_product_manager(selector_func):
    message = TextMessage(
        source="user",
        content="请开发一个比特币价格应用。",
    )

    assert selector_func([message]) == "product_manager"


@pytest.mark.parametrize(
    ("route", "expected_agent"),
    [
        ("ROUTE: QA_PASSED", "user_proxy"),
        ("ROUTE: QA_IMPLEMENTATION_CHANGES_REQUIRED", "engineer"),
        ("ROUTE: QA_REQUIREMENTS_REVIEW_REQUIRED", "product_manager"),
    ],
)
def test_quality_assurance_routes(selector_func, route, expected_agent):
    """验证QualityAssurance的三种主要路由。"""
    message = TextMessage(
        source="quality_assurance",
        content=f"## 质量结论\n测试结论\n\n{route}",
    )

    assert selector_func([message]) == expected_agent


def test_qa_missing_route_falls_back_to_user_proxy(selector_func):
    """QA缺少ROUTE时交给真人处理，不能擅自判定质量检查通过。"""
    message = TextMessage(
        source="quality_assurance",
        content="## 质量结论\n未提供路由状态。",
    )

    assert selector_func([message]) == "user_proxy"


def test_user_feedback_routes_to_product_manager(selector_func):
    """用户不接受结果并提出修改意见时，重新进入需求分析。"""
    message = TextMessage(
        source="user_proxy",
        content="请增加人民币价格显示。",
    )

    assert selector_func([message]) == "product_manager"
