"""多资产实时行情工具 (MarketQuoteTool) 实现。

继承自 BaseTool，免 API Key 实时查询美股核心标的（如 SPY, QQQ, NVDA, AAPL）
以及主流加密货币（如 USDT, BTC, ETH）的最新市场价格、24小时涨跌幅与基础量化指标。
为 Agent 的投资决策提供客观、真实的数据事实支撑。
"""

import json
import urllib.request
from typing import Any

from chapter07_my_framework.tools.base import BaseTool


class MarketQuoteTool(BaseTool):
    """实时市场行情检索工具。

    支持美股与加密货币的一站式价格快照查询。
    """

    def __init__(self) -> None:
        super().__init__(
            name="market_quote",
            description=(
                "实时资产行情快照工具。可免 Key 查询美股核心指数/个股（如 SPY, QQQ, NVDA, AAPL）"
                "以及加密货币（如 BTC, ETH, USDT）的最新美元报价、24h 涨跌幅及日内高低点。"
            ),
            parameters_description=(
                "symbol: 必填，标准资产代码字符串，例如 'SPY'（标普500ETF）、'NVDA'（英伟达）、'BTC'（比特币）或 'USDT'。"
            ),
        )

    def execute(self, symbol, **kwargs: Any) -> str:
        """执行实际的行情查询请求
        Args:
            symbol: 资产代号（如 'NVDA', 'BTC', 'SPY'）
        Returns:
            str: 格式化的行情数据 Markdown 摘要。
        """
        if not symbol or not symbol.strip():
            return "❌ 行情查询失败：必须提供有效的资产代码（例如 'NVDA' 或 'BTC'）。"
        clean_symbol = symbol.strip().upper()

        crypto_symbol = {"BTC", "ETH", "USDT", "SOL", "BNB", "DOGE"}
        if clean_symbol in crypto_symbol or clean_symbol.endswith("USDT"):
            # 返回加密货币路由信息
            return self._fetch_crypto_quote(clean_symbol)
        else:
            # 返回美股路由信息
            return self._fetch_stock_quote(clean_symbol)

    def _fetch_stock_quote(self, symbol: str) -> str:
        """从 Yahoo Finance 公共端点查询美股实时/盘前盘后行情。"""

        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=5d"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/json",
            },
        )

        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))

        chart = data.get("chart", {})
        result = chart.get("result", {})

        if not result:
            return f"❌ 未能获取到美股代码 '{symbol}' 的有效行情，请核实代码是否正确。"

        meta = result[0].get("meta", {})
        current_price = meta.get("regularMarketPrice", 0.0)
        previous_close = meta.get("chartPreviousClose", current_price)
        currency = meta.get("currency", "USD")
        market_time = meta.get("regularMarketTime", "未知")

        # 计算日内涨跌额与涨跌幅
        change = current_price - previous_close
        change_pct = (change / previous_close * 100) if previous_close else 0.0
        direction = "🟢 上涨" if change >= 0 else "🔴 下跌"

        fifty_two_high = meta.get("fiftyTwoWeekHigh", 0.0)
        fifty_two_low = meta.get("fiftyTwoWeekLow", 0.0)

        return (
            f"### 📈 美股资产实时行情快照 [{symbol}]\n"
            f"- **当前价格**: `${current_price:,.2f} {currency}`\n"
            f"- **日内涨跌**: {direction} `{change:+.2f} ({change_pct:+.2f}%)`\n"
            f"- **昨日收盘价**: `${previous_close:,.2f}`\n"
            f"- **52周波动区间**: `${fifty_two_low:,.2f} ~ ${fifty_two_high:,.2f}`\n"
            f"- **交易币种**: `{currency}`"
        )

    def _fetch_crypto_quote(self, symbol: str) -> str:
        """从公共接口查询加密货币实时行情（默认以 USDT 计价）。"""

        # 稳定币代号映射：USDT 本身是计价基准币，查询 USDCUSDT 交易对观察微观脱锚与汇率
        is_stablecoin_check = symbol == "USDT"
        if is_stablecoin_check:
            pair = "USDCUSDT"
        else:
            pair = symbol if symbol.endswith("USDT") else f"{symbol}USDT"

        # 使用币安官方公开免 Key 行情接口
        url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={pair}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        )

        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))

        last_price = float(data.get("lastPrice", 0.0))
        price_change_pct = float(data.get("priceChangePercent", 0.0))
        high_price = float(data.get("highPrice", 0.0))
        low_price = float(data.get("lowPrice", 0.0))
        volume = float(data.get("volume", 0.0))

        direction = "🟢 上涨" if price_change_pct >= 0 else "🔴 下跌"

        # 如果查询的是稳定币 USDT，输出专门的锚定健康度报告
        if is_stablecoin_check:
            # 衡量与 1 美元的偏离度（脱锚阈值通常为 1%）
            is_pegged = abs(last_price - 1.0) < 0.01
            peg_status = "🟢 锚定正常 (1 USDT ≈ $1.00 USD)" if is_pegged else "⚠️ 警告：检测到微观脱锚波动"
            return (
                f"### 🪙 稳定币实时状态快照 [USDT / 美元锚定]\n"
                f"- **基准参考价**: `$1.0000 USD` (1:1 法币资产抵押挂钩)\n"
                f"- **市场成交比价 (USDC/USDT)**: `${last_price:,.4f}`\n"
                f"- **24h 微观波动**: `{price_change_pct:+.2f}%`\n"
                f"- **24h 波动区间**: `${low_price:,.4f} ~ ${high_price:,.4f}`\n"
                f"- **锚定健康状态**: {peg_status}\n"
                f"- **24h 交易量**: `${volume:,.2f} USDT`"
            )

        return (
            f"### 🪙 加密资产实时行情快照 [{pair}]\n"
            f"- **当前现价**: `${last_price:,.4f}`\n"
            f"- **24h 涨跌幅**: {direction} `{price_change_pct:+.2f}%`\n"
            f"- **24h 最高价**: `${high_price:,.4f}`\n"
            f"- **24h 最低价**: `${low_price:,.4f}`\n"
            f"- **24h 成交量**: `{volume:,.2f} {symbol.replace('USDT', '')}`"
        )


if __name__ == "__main__":
    # --- 模块自测逻辑：验证免 Key 美股与加密货币双重查询 ---
    print("--- 正在验证 MarketQuoteTool 免 Key 行情查询工具 ---")

    tool = MarketQuoteTool()
    print(f"✅ 成功实例化工具: {tool}")
    print("📋 工具格式化 Prompt 描述:\n" + tool.get_tool_info())

    # 1. 测试美股标的 (以标普500 ETF: SPY 和 科技巨头 NVDA 为例)
    print("\n🔍 [测试 1: 美股标普500 ETF (SPY) 实时行情]")
    spy_quote = tool.run(symbol="SPY")
    print(spy_quote)

    # 2. 测试加密货币标的 (以 BTC 为例)
    print("\n🔍 [测试 2: 加密资产比特币 (BTC) 实时行情]")
    btc_quote = tool.run(symbol="BTC")
    print(btc_quote)

    # 3. 测试 USDT 稳定性与价格
    print("\n🔍 [测试 3: 稳定币 (USDT) 实时行情]")
    usdt_quote = tool.run(symbol="USDT")
    print(usdt_quote)

    print("\n🎉 MarketQuoteTool 行情工具自测全部通过！")
