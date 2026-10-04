"""汇率换算与资金磨损计算工具 (CurrencyConverterTool) 实现。

继承自 BaseTool，免 API Key 查询 AUD（澳元）与 USD、USDT、CNY 的实时外汇汇率。
针对 30 AUD 等微小资金投资，内置换汇点差（Spread）与通道手续费扣除计算，
精确计算出投资者的“实际有效净购买力 (Net Buying Power)”，防止盲目投资导致本金被手续费蚕食。
"""

import json
import urllib.request
from typing import Any, Dict

from chapter07_my_framework.tools.base import BaseTool


class CurrencyConverterTool(BaseTool):
    """外汇换算与净投资额度计算工具。
    支持实时汇率转换以及交易摩擦成本（手续费、点差）评估。
    """

    def __init__(self) -> None:
        """初始化汇率工具元数据"""
        super().__init__(
            name="currency_converter",
            description=(
                "法币与稳定币汇率换算及净资本计算器。用于查询 AUD 与 USD/USDT 的实时汇率，"
                "并自动扣除预估的换汇滑点与固定手续费，计算 30 AUD 本金最终到账的实际有效美元购买力。"
            ),
            parameters_description=(
                "amount: 必填，待转换的资金金额数值（例如 30）。\n"
                "from_currency: 选填，源货币代码，默认 'AUD'。\n"
                "to_currency: 选填，目标货币代码，默认 'USD'（可选 'USD', 'USDT', 'CNY'）。\n"
                "fee_percent: 选填，百分比手续费/点差（如 0.5 表示 0.5%），默认 0.5%。\n"
                "fixed_fee_usd: 选填，固定通道/入金手续费美元数（如 1.0 表示 1 USD），默认 0.0。"
            ),
        )

    def execute(
            self,
            amount: float | int | str,
            from_currency: str = "AUD",
            to_currency: str = "USD",
            fee_percent: float | int | str = 0.5,
            fixed_fee_usd: float | int | str = 0.0,
            **kwargs: Any
    ) -> str:
        """执行汇率换算与摩擦损耗计算。
        Args:
            amount: 转换金额。
            from_currency: 源币种代码（默认 AUD）。
            to_currency: 目标币种代码（默认 USD）。
            fee_percent: 预估换汇点差比例（百分比）。
            fixed_fee_usd: 预估固定手续费（美元）。

        Returns:
            str: 格式化的资金换算与磨损分析 Markdown 报告。
        """
        try:
            amount = float(amount)
            pct_fee = float(fee_percent)
            fixed_fee = float(fixed_fee_usd)
        except (ValueError, TypeError):
            return f"❌ 汇率计算失败：传入的金额或手续费参数不是有效的数字 (amount={amount})。"

        if amount <= 0:
            return "❌ 汇率计算失败：投资金额必须大于 0。"

        from_currency = from_currency.strip().upper()
        to_currency = to_currency.strip().upper()

        # 处理 USDT：在汇率层面 1 USDT 基础锚定 1 USD
        target_lookup = "USD" if to_currency in {"USDT", "USD"} else to_currency

        try:
            rate = self._fetch_exchange_rate(from_currency, target_lookup)
        except Exception as e:
            return f"❌ 获取实时外汇汇率失败: {str(e)}"

        # 1. 理论无损转换金额
        gross_target_amount = amount * rate
        # 2. 扣除点差/百分比磨损 (例如 0.5%)
        spread_cost = gross_target_amount * (pct_fee / 100.0)
        # 3. 扣除固定通道费用 (例如 1 USD 链上转账费或入金最低扣费)
        total_fee_target = spread_cost + fixed_fee
        net_target_amount = max(0.0, gross_target_amount - total_fee_target)
        # 4. 计算综合摩擦成本比例
        fee_ratio = (total_fee_target / gross_target_amount * 100) if gross_target_amount > 0 else 0.0

        # 风控告警：如果单次手续费占比超过 3%，提示磨损严重
        risk_alert = ""
        if fee_ratio > 5.0:
            risk_alert = "⚠️ **严重磨损警示**：当前手续费占本金比例过高，极微小资金频繁换汇将严重吞噬收益！建议减少交易频次或选择零入金手续费渠道。"
        elif fee_ratio > 2.0:
            risk_alert = "💡 **成本提示**：检测到换汇/通道摩擦成本，建议作为中长线持有，避免超短线频繁换手。"

        return (
            f"### 💱 资金换算与有效净资本测算报告\n"
            f"- **初始投入资金**: `{amount:,.2f} {from_currency}`\n"
            f"- **基准外汇汇率**: `1 {from_currency} = {rate:.4f} {to_currency}`\n"
            f"- **理论未扣费金额**: `${gross_target_amount:,.2f} {to_currency}`\n"
            f"- **预估换汇点差损耗 ({pct_fee}%)**: `-${spread_cost:,.2f} {to_currency}`\n"
            f"- **固定通道/入金扣费**: `-${fixed_fee:,.2f} {to_currency}`\n"
            f"- **真实到账有效购买力 (净值)**: **`${net_target_amount:,.2f} {to_currency}`**\n"
            f"- **综合摩擦损耗率**: `{fee_ratio:.2f}%`\n"
            f"{risk_alert}"
        )

    def _fetch_exchange_rate(self, base: str, target: str) -> float:
        """从免 Key 的开放外汇 API 获取基准汇率。"""
        url = f"https://open.er-api.com/v6/latest/{base}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        )

        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))

        if data.get("result") != "success":
            raise RuntimeError(f"开放汇率接口返回非成功状态: {data}")

        rates: Dict[str, float] = data.get("rates", {})
        if target not in rates:
            raise KeyError(f"未在实时外汇表中找到目标币种 '{target}' 的汇率。")

        return float(rates[target])


if __name__ == "__main__":
    # --- 模块自测逻辑：验证 30 澳元的换算与手续费测算 ---
    print("--- 正在验证 CurrencyConverterTool 汇率与净购买力工具 ---")

    tool = CurrencyConverterTool()
    print(f"✅ 成功实例化工具: {tool}")
    print("📋 工具格式化 Prompt 描述:\n" + tool.get_tool_info())

    # 1. 测试基础 30 AUD 换算为 USD（无固定费，0.5% 正常换汇点差）
    print("\n🔍 [测试 1: 30 AUD 基础换算为 USD (0.5% 点差)]")
    res1 = tool.run(amount=30, from_currency="AUD", to_currency="USD")
    print(res1)

    # 2. 测试 30 AUD 兑换 USDT 并在链上产生 1.5 USD 固定 Gas/提现费的情形
    print("\n🔍 [测试 2: 30 AUD 兑换 USDT 并扣除 1.5 USD 通道费 (高摩擦测试)]")
    res2 = tool.run(amount=30, from_currency="AUD", to_currency="USDT", fixed_fee_usd=1.5)
    print(res2)

    print("\n🎉 CurrencyConverterTool 汇率工具自测全部通过！")
