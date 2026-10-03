"""多源网络搜索工具 (SearchTool) 实现。

继承自 BaseTool，优先使用专为 AI Agent 设计的 Tavily Search API，
负责为智能体提供实时互联网资讯、突发财经事件与宏观动态检索能力。
支持自动读取环境变量、强制过滤中文噪音、筛选一手英文权威信源，
并面向下游 Agent 决策提炼结构化摘要。
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

from chapter07_my_framework.tools.base import BaseTool

# 确保独立运行该模块时也能自动加载根目录 .env
# search.py -> builtin (0) -> tools (1) -> chapter07_my_framework (2) -> LearnAgents (3)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")


def _is_mainly_chinese(text: str, threshold: float = 0.08) -> bool:
    """检测一段文本是否主要是中文。

    通过统计 Unicode 汉字范围 [\\u4e00-\\u9fa5] 的字符占比。
    如果汉字比例超过阈值（默认 8%），则判定为中文网页。

    Args:
        text: 待检测文本。
        threshold: 汉字字符占总长度的比例阈值。

    Returns:
        bool: 是中文网页返回 True，纯英文/极少量汉字返回 False。
    """
    if not text:
        return False
    # 统计所有汉字
    chinese_chars = [c for c in text if "\u4e00" <= c <= "\u9fa5"]
    ratio = len(chinese_chars) / max(len(text), 1)
    return ratio > threshold


class SearchTool(BaseTool):
    """全球一手英文财经资讯搜索工具。

    支持对宏观货币政策、美联储决议、美股科技巨头/ETF与加密货币进行权威一手信源检索。
    自动过滤中文二次转载噪音，输出结构化的高信息密度英文摘要。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_results: int = 3,
        english_only: bool = True,
    ) -> None:
        """初始化搜索工具元数据。

        Args:
            api_key: Tavily API 密钥。若不传，自动从环境变量 TAVILY_API_KEY 读取。
            max_results: 最终保留的高质量英文结果条数。
            english_only: 是否强制过滤中文内容，仅保留英文一手信源（默认 True）。
        """
        super().__init__(
            name="search",
            description=(
                "全球一手互联网财经与新闻搜索引擎。用于检索权威英文一手信源（如 Reuters, Bloomberg, "
                "CNBC, CoinDesk, SEC 等）关于美联储政策、美股行情、加密货币的最新事实、财报数据与宏观走向。"
                "注意：本工具优先处理英文关键词（如 'Fed rate cut impact US stocks' 或 'USDT market cap'）。"
            ),
            parameters_description=(
                "query: 必填，搜索关键词字符串，必须且强烈推荐使用英文专业术语或标的代码（例如 'NVIDIA Q3 earnings report' "
                "或 'Bitcoin ETF inflows outflows'），以获取最权威的第一手市场数据。"
            ),
        )
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.default_max_results = max_results
        self.english_only = english_only

    def execute(self, query: str, **kwargs: Any) -> str:
        """执行网络搜索，执行中文噪音过滤，并提炼结构化摘要。

        Args:
            query: 搜索查询字符串（推荐英文）。
            **kwargs: 允许传入 max_results, english_only 等动态控制参数。

        Returns:
            str: 面向智能体决策的结构化 Markdown 摘要文本块。
        """
        if not query or not query.strip():
            return "❌ 搜索失败：搜索查询关键词不能为空。"

        cleaned_query = query.strip()
        target_results_count = kwargs.get("max_results", self.default_max_results)
        english_only = kwargs.get("english_only", self.english_only)

        # 检查是否配置了 Tavily 密钥
        if not self.api_key:
            return (
                f"⚠️ 搜索服务提示：未检测到 TAVILY_API_KEY 环境变量配置。\n"
                f"当前仅能返回本地模拟占位数据：已接收查询 '{cleaned_query}'，"
                f"请在项目根目录 .env 中配置 TAVILY_API_KEY 以启用真实的实时全球搜索能力。"
            )

        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=self.api_key)

            # 为了在过滤掉中文网页后仍能凑齐目标条数，向底层 Tavily 索取 2~3 倍的候选池
            fetch_count = target_results_count * 3 if english_only else target_results_count

            search_response = client.search(
                query=cleaned_query,
                search_depth="basic",
                max_results=min(fetch_count, 10),
            )

            raw_results: List[Dict[str, Any]] = search_response.get("results", [])
            if not raw_results:
                return f"🔍 未检索到关于 '{cleaned_query}' 的相关公开网络资讯。"

            # 筛选与清洗流
            filtered_results: List[Dict[str, Any]] = []
            for item in raw_results:
                title = item.get("title", "")
                content = item.get("content", "")

                # 中文网页过滤机制
                if english_only:
                    # 只要标题或正文中汉字超标，直接剔除二次转载的中文网页
                    if _is_mainly_chinese(title) or _is_mainly_chinese(content):
                        continue

                filtered_results.append(item)
                if len(filtered_results) >= target_results_count:
                    break

            if not filtered_results:
                return (
                    f"🔍 已检索到底层结果，但因开启了 english_only 过滤机制，"
                    f"所有返回结果均为中文或包含大量中文字符。建议尝试将搜索词 '{cleaned_query}' 改为纯英文关键词再次搜索。"
                )

            # 格式化输出为面向 Agent 决策的结构化洞察报告 (Executive Insights)
            formatted_outputs: List[str] = [
                f"### Global Market Insights for '{cleaned_query}' (Primary English Sources):"
            ]
            for idx, item in enumerate(filtered_results, start=1):
                title = item.get("title", "No Title").strip()
                content = item.get("content", "No summary available.").strip()
                url = item.get("url", "#")

                # 精炼摘要：清洗多余换行，并控制单条长度在合理决策窗口（约 200~350 字符）
                clean_summary = " ".join(content.split())
                if len(clean_summary) > 350:
                    clean_summary = clean_summary[:350] + "..."

                formatted_outputs.append(
                    f"{idx}. **{title}**\n"
                    f"   - **Key Facts / Takeaway**: {clean_summary}\n"
                    f"   - **Source**: {url}"
                )

            return "\n\n".join(formatted_outputs)

        except Exception as e:
            return f"❌ 搜索远程 API 执行异常: {str(e)}"


if __name__ == "__main__":
    # --- 模块自测逻辑：验证中文字符过滤与纯正英文一手信息提取 ---
    print("--- 正在验证 SearchTool 全球一手英文信源搜索与摘要提炼 ---")

    tool = SearchTool(max_results=3, english_only=True)
    print(f"✅ 成功实例化工具: {tool}")
    print("📋 工具格式化 Prompt 描述:\n" + tool.get_tool_info())

    # 测试案例 1：纯英文金融宏观查询（美联储利率与美股影响）
    test_query_en = "Federal Reserve interest rate decision stock market impact"
    print(f"\n🔍 [测试 1: 纯英文一手信源检索] '{test_query_en}'")
    search_result_en = tool.run(query=test_query_en)
    print("\n📄 搜索执行返回结果:")
    print(search_result_en)

    # 测试案例 2：输入包含中文关键词，验证中文网页过滤与英文保底
    test_query_mix = "USDT supply and risk 2026"
    print(f"\n🔍 [测试 2: 加密货币英文信源检索] '{test_query_mix}'")
    search_result_mix = tool.run(query=test_query_mix)
    print("\n📄 搜索执行返回结果:")
    print(search_result_mix)

    print("\n🎉 SearchTool 重构测试全部完成！")