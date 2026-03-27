import os

from langchain_core.tools import tool
from typing import Annotated
from tradingagents.dataflows.interface import route_to_vendor


def _tpm_budget_enabled() -> bool:
    raw = os.getenv("TRADINGAGENTS_MAX_TPM", "").strip()
    return raw.isdigit() and int(raw) > 0


def _compact_tool_output(text: str, max_lines: int = 18, max_chars: int = 3500) -> str:
    if not _tpm_budget_enabled():
        return text

    lines = text.splitlines()
    if len(lines) > max_lines:
        head_count = min(4, max_lines // 2)
        tail_count = max_lines - head_count
        omitted = max(0, len(lines) - head_count - tail_count)
        lines = (
            lines[:head_count]
            + [f"... [{omitted} lines truncated for TRADINGAGENTS_MAX_TPM budget] ..."]
            + lines[-tail_count:]
        )
        text = "\n".join(lines)

    if len(text) > max_chars:
        text = text[: max_chars - 48].rstrip() + "\n... [truncated for TRADINGAGENTS_MAX_TPM budget]"

    return text


@tool
def get_stock_data(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """
    Retrieve stock price data (OHLCV) for a given ticker symbol.
    Uses the configured core_stock_apis vendor.
    Args:
        symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
        start_date (str): Start date in yyyy-mm-dd format
        end_date (str): End date in yyyy-mm-dd format
    Returns:
        str: A formatted dataframe containing the stock price data for the specified ticker symbol in the specified date range.
    """
    result = route_to_vendor("get_stock_data", symbol, start_date, end_date)
    return _compact_tool_output(result)
