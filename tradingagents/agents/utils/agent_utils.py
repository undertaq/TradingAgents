import os

from langchain_core.messages import HumanMessage, RemoveMessage

# Import tools from separate utility files
from tradingagents.agents.utils.core_stock_tools import (
    get_stock_data
)
from tradingagents.agents.utils.technical_indicators_tools import (
    get_indicators
)
from tradingagents.agents.utils.fundamental_data_tools import (
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement
)
from tradingagents.agents.utils.news_data_tools import (
    get_news,
    get_insider_transactions,
    get_global_news
)
from tradingagents.dataflows.config import get_config


def get_tpm_budget() -> int | None:
    """Return the optional per-request TPM budget configured via env."""
    raw = os.getenv("TRADINGAGENTS_MAX_TPM", "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def is_tpm_constrained() -> bool:
    """Whether prompt/tool payloads should be compacted for low-TPM providers."""
    return get_tpm_budget() is not None


def get_output_language() -> str:
    config = get_config()
    language = config.get("language", "en")
    return language if language in {"en", "zh-TW"} else "en"


def get_language_instruction(keep_decision_keywords_english: bool = False) -> str:
    if get_output_language() == "zh-TW":
        if keep_decision_keywords_english:
            return (
                "Write all user-facing analysis in Traditional Chinese. "
                "Keep fixed decision keywords such as BUY, OVERWEIGHT, HOLD, UNDERWEIGHT, SELL, "
                "and FINAL TRANSACTION PROPOSAL exactly in English."
            )
        return "Write all user-facing analysis in Traditional Chinese."

    if keep_decision_keywords_english:
        return (
            "Write all user-facing analysis in English. "
            "Keep fixed decision keywords such as BUY, OVERWEIGHT, HOLD, UNDERWEIGHT, SELL, "
            "and FINAL TRANSACTION PROPOSAL exactly in English."
        )
    return "Write all user-facing analysis in English."


def build_instrument_context(ticker: str) -> str:
    """Describe the exact instrument so agents preserve exchange-qualified tickers."""
    return (
        f"The instrument to analyze is `{ticker}`. "
        "Use this exact ticker in every tool call, report, and recommendation, "
        "preserving any exchange suffix (e.g. `.TO`, `.L`, `.HK`, `.T`)."
    )

def create_msg_delete():
    def delete_messages(state):
        """Clear messages and add placeholder for Anthropic compatibility"""
        messages = state["messages"]

        # Remove all messages
        removal_operations = [RemoveMessage(id=m.id) for m in messages]

        # Add a minimal placeholder message
        placeholder = HumanMessage(content="Continue")

        return {"messages": removal_operations + [placeholder]}

    return delete_messages


        
