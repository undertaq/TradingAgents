import os

from langchain_core.tools import tool
from typing import Annotated
from tradingagents.dataflows.interface import route_to_vendor


def _tpm_budget_enabled() -> bool:
    raw = os.getenv("TRADINGAGENTS_MAX_TPM", "").strip()
    return raw.isdigit() and int(raw) > 0


def _compact_tool_output(text: str, max_lines: int = 24, max_chars: int = 4500) -> str:
    if not _tpm_budget_enabled():
        return text

    lines = text.splitlines()
    if len(lines) > max_lines:
        head_count = min(5, max_lines // 2)
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


def _coerce_int(value, default: int) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        value = value.strip()
        if value.isdigit():
            return int(value)
    return default


@tool
def get_indicators(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
    look_back_days: Annotated[str, "how many days to look back"] = "30",
) -> str:
    """
    Retrieve a single technical indicator for a given ticker symbol.
    Uses the configured technical_indicators vendor.
    Args:
        symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
        indicator (str): A single technical indicator name, e.g. 'rsi', 'macd'. Call this tool once per indicator.
        curr_date (str): The current trading date you are trading on, YYYY-mm-dd
        look_back_days (int): How many days to look back, default is 30
    Returns:
        str: A formatted dataframe containing the technical indicators for the specified ticker symbol and indicator.
    """
    # LLMs sometimes pass multiple indicators as a comma-separated string;
    # split and process each individually.
    look_back_days_int = _coerce_int(look_back_days, 30)
    indicators = [i.strip() for i in indicator.split(",") if i.strip()]
    if len(indicators) > 1:
        results = []
        for ind in indicators:
            results.append(route_to_vendor("get_indicators", symbol, ind, curr_date, look_back_days_int))
        return _compact_tool_output("\n\n".join(results))
    result = route_to_vendor("get_indicators", symbol, indicator.strip(), curr_date, look_back_days_int)
    return _compact_tool_output(result)
