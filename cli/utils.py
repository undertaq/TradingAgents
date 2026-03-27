import questionary
from typing import List, Optional, Tuple, Dict

from rich.console import Console

from cli.i18n import set_language, t, tr_agent
from cli.models import AnalystType
from tradingagents.default_config import DEFAULT_CONFIG

console = Console()

TICKER_INPUT_EXAMPLES = "Examples: SPY, CNC.TO, 7203.T, 0700.HK"

ANALYST_ORDER = [
    ("Market Analyst", AnalystType.MARKET),
    ("Social Analyst", AnalystType.SOCIAL),
    ("News Analyst", AnalystType.NEWS),
    ("Fundamentals Analyst", AnalystType.FUNDAMENTALS),
]


def prioritize_default_option(
    options: list[tuple[str, str]], default_value: str | None
) -> list[tuple[str, str]]:
    if not default_value:
        return options
    prioritized = [option for option in options if option[1] == default_value]
    prioritized.extend(option for option in options if option[1] != default_value)
    return prioritized


def select_language() -> str:
    choice = questionary.select(
        t("language_prompt"),
        choices=[
            questionary.Choice(t("language_english"), value="en"),
            questionary.Choice(t("language_zh_tw"), value="zh-TW"),
        ],
    ).ask()
    if not choice:
        choice = "en"
    set_language(choice)
    return choice


def get_ticker() -> str:
    """Prompt the user to enter a ticker symbol."""
    ticker = questionary.text(
        f"{t('step_1_prompt')}:",
        validate=lambda x: len(x.strip()) > 0 or t("error_invalid_ticker"),
        style=questionary.Style(
            [
                ("text", "fg:green"),
                ("highlighted", "noinherit"),
            ]
        ),
    ).ask()

    if not ticker:
        console.print(f"\n[red]{t('error_no_ticker')}[/red]")
        exit(1)

    return normalize_ticker_symbol(ticker)


def normalize_ticker_symbol(ticker: str) -> str:
    """Normalize ticker input while preserving exchange suffixes."""
    return ticker.strip().upper()


def get_analysis_date() -> str:
    """Prompt the user to enter a date in YYYY-MM-DD format."""
    import re
    from datetime import datetime

    def validate_date(date_str: str) -> bool:
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
            return False
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            return False

    date = questionary.text(
        f"{t('step_2_prompt')}:",
        validate=lambda x: validate_date(x.strip()) or t("error_invalid_date"),
        style=questionary.Style(
            [
                ("text", "fg:green"),
                ("highlighted", "noinherit"),
            ]
        ),
    ).ask()

    if not date:
        console.print(f"\n[red]{t('error_no_date')}[/red]")
        exit(1)

    return date.strip()


def select_analysts() -> List[AnalystType]:
    """Select analysts using an interactive checkbox."""
    choices = questionary.checkbox(
        f"{t('step_3_prompt')}:",
        choices=[
            questionary.Choice(tr_agent(display), value=value)
            for display, value in ANALYST_ORDER
        ],
        instruction=f"\n- {t('instruction_checkbox')}",
        validate=lambda x: len(x) > 0 or t("error_no_analysts"),
        style=questionary.Style(
            [
                ("checkbox-selected", "fg:green"),
                ("selected", "fg:green noinherit"),
                ("highlighted", "noinherit"),
                ("pointer", "noinherit"),
            ]
        ),
    ).ask()

    if not choices:
        console.print(f"\n[red]{t('error_no_analysts')}[/red]")
        exit(1)

    return choices


def select_research_depth() -> int:
    """Select research depth using an interactive selection."""

    # Define research depth options with their corresponding values
    DEPTH_OPTIONS = [
        (t("research_depth_shallow"), 1),
        (t("research_depth_medium"), 3),
        (t("research_depth_deep"), 5),
    ]

    choice = questionary.select(
        f"{t('step_4_prompt')}:",
        choices=[
            questionary.Choice(display, value=value) for display, value in DEPTH_OPTIONS
        ],
        instruction=f"\n- {t('instruction_select')}",
        style=questionary.Style(
            [
                ("selected", "fg:yellow noinherit"),
                ("highlighted", "fg:yellow noinherit"),
                ("pointer", "fg:yellow noinherit"),
            ]
        ),
    ).ask()

    if choice is None:
        console.print(f"\n[red]{t('error_no_research_depth')}[/red]")
        exit(1)

    return choice


def select_shallow_thinking_agent(provider) -> str:
    """Select shallow thinking llm engine using an interactive selection."""

    # Define shallow thinking llm engine options with their corresponding model names
    # Ordering: medium → light → heavy (balanced first for quick tasks)
    # Within same tier, newer models first
    SHALLOW_AGENT_OPTIONS = {
        "openai": [
            ("GPT-5 Mini - Balanced speed, cost, and capability", "gpt-5-mini"),
            ("GPT-5 Nano - High-throughput, simple tasks", "gpt-5-nano"),
            ("GPT-5.4 - Latest frontier, 1M context", "gpt-5.4"),
            ("GPT-4.1 - Smartest non-reasoning model", "gpt-4.1"),
        ],
        "anthropic": [
            ("Claude Sonnet 4.6 - Best speed and intelligence balance", "claude-sonnet-4-6"),
            ("Claude Haiku 4.5 - Fast, near-instant responses", "claude-haiku-4-5"),
            ("Claude Sonnet 4.5 - Agents and coding", "claude-sonnet-4-5"),
        ],
        "google": [
            ("Gemini 3 Flash - Next-gen fast", "gemini-3-flash-preview"),
            ("Gemini 2.5 Flash - Balanced, stable", "gemini-2.5-flash"),
            ("Gemini 3.1 Flash Lite - Most cost-efficient", "gemini-3.1-flash-lite-preview"),
            ("Gemini 2.5 Flash Lite - Fast, low-cost", "gemini-2.5-flash-lite"),
        ],
        "groq": [
            ("Llama 3.1 8B Instant - Lowest latency", "llama-3.1-8b-instant"),
            ("GPT-OSS 20B - Fast open-weight reasoning", "openai/gpt-oss-20b"),
            ("Llama 3.3 70B Versatile - Strong general purpose", "llama-3.3-70b-versatile"),
        ],
        "xai": [
            ("Grok 4.1 Fast (Non-Reasoning) - Speed optimized, 2M ctx", "grok-4-1-fast-non-reasoning"),
            ("Grok 4 Fast (Non-Reasoning) - Speed optimized", "grok-4-fast-non-reasoning"),
            ("Grok 4.1 Fast (Reasoning) - High-performance, 2M ctx", "grok-4-1-fast-reasoning"),
        ],
        "openrouter": [
            ("NVIDIA Nemotron 3 Nano 30B (free)", "nvidia/nemotron-3-nano-30b-a3b:free"),
            ("Z.AI GLM 4.5 Air (free)", "z-ai/glm-4.5-air:free"),
        ],
        "ollama": [
            ("Qwen3.5 27B (local)", "qwen3.5:27b"),
            ("Qwen3:latest (8B, local)", "qwen3:latest"),
            ("GPT-OSS:latest (20B, local)", "gpt-oss:latest"),
            ("GLM-4.7-Flash:latest (30B, local)", "glm-4.7-flash:latest"),
        ],
    }
    default_model = DEFAULT_CONFIG.get("quick_think_llm")
    options = prioritize_default_option(
        SHALLOW_AGENT_OPTIONS[provider.lower()], default_model
    )

    choice = questionary.select(
        t("quick_thinking_prompt"),
        choices=[
            questionary.Choice(display, value=value) for display, value in options
        ],
        instruction=f"\n- {t('instruction_select')}",
        style=questionary.Style(
            [
                ("selected", "fg:magenta noinherit"),
                ("highlighted", "fg:magenta noinherit"),
                ("pointer", "fg:magenta noinherit"),
            ]
        ),
    ).ask()

    if choice is None:
        console.print(f"\n[red]{t('error_no_shallow')}[/red]")
        exit(1)

    return choice


def select_deep_thinking_agent(provider) -> str:
    """Select deep thinking llm engine using an interactive selection."""

    # Define deep thinking llm engine options with their corresponding model names
    # Ordering: heavy → medium → light (most capable first for deep tasks)
    # Within same tier, newer models first
    DEEP_AGENT_OPTIONS = {
        "openai": [
            ("GPT-5.4 - Latest frontier, 1M context", "gpt-5.4"),
            ("GPT-5.2 - Strong reasoning, cost-effective", "gpt-5.2"),
            ("GPT-5 Mini - Balanced speed, cost, and capability", "gpt-5-mini"),
            ("GPT-5.4 Pro - Most capable, expensive ($30/$180 per 1M tokens)", "gpt-5.4-pro"),
        ],
        "anthropic": [
            ("Claude Opus 4.6 - Most intelligent, agents and coding", "claude-opus-4-6"),
            ("Claude Opus 4.5 - Premium, max intelligence", "claude-opus-4-5"),
            ("Claude Sonnet 4.6 - Best speed and intelligence balance", "claude-sonnet-4-6"),
            ("Claude Sonnet 4.5 - Agents and coding", "claude-sonnet-4-5"),
        ],
        "google": [
            ("Gemini 3.1 Pro - Reasoning-first, complex workflows", "gemini-3.1-pro-preview"),
            ("Gemini 3 Flash - Next-gen fast", "gemini-3-flash-preview"),
            ("Gemini 2.5 Pro - Stable pro model", "gemini-2.5-pro"),
            ("Gemini 2.5 Flash - Balanced, stable", "gemini-2.5-flash"),
        ],
        "groq": [
            ("Llama 3.3 70B Versatile - Strongest default Groq option", "llama-3.3-70b-versatile"),
            ("Kimi K2 Instruct - Larger reasoning-oriented model", "moonshotai/kimi-k2-instruct-0905"),
            ("GPT-OSS 120B - Highest-capacity open-weight option", "openai/gpt-oss-120b"),
            ("GPT-OSS 20B - Faster open-weight reasoning", "openai/gpt-oss-20b"),
        ],
        "xai": [
            ("Grok 4 - Flagship model", "grok-4-0709"),
            ("Grok 4.1 Fast (Reasoning) - High-performance, 2M ctx", "grok-4-1-fast-reasoning"),
            ("Grok 4 Fast (Reasoning) - High-performance", "grok-4-fast-reasoning"),
            ("Grok 4.1 Fast (Non-Reasoning) - Speed optimized, 2M ctx", "grok-4-1-fast-non-reasoning"),
        ],
        "openrouter": [
            ("Z.AI GLM 4.5 Air (free)", "z-ai/glm-4.5-air:free"),
            ("NVIDIA Nemotron 3 Nano 30B (free)", "nvidia/nemotron-3-nano-30b-a3b:free"),
        ],
        "ollama": [
            ("Qwen3.5 27B (local)", "qwen3.5:27b"),
            ("GLM-4.7-Flash:latest (30B, local)", "glm-4.7-flash:latest"),
            ("GPT-OSS:latest (20B, local)", "gpt-oss:latest"),
            ("Qwen3:latest (8B, local)", "qwen3:latest"),
        ],
    }
    default_model = DEFAULT_CONFIG.get("deep_think_llm")
    options = prioritize_default_option(
        DEEP_AGENT_OPTIONS[provider.lower()], default_model
    )

    choice = questionary.select(
        t("deep_thinking_prompt"),
        choices=[
            questionary.Choice(display, value=value) for display, value in options
        ],
        instruction=f"\n- {t('instruction_select')}",
        style=questionary.Style(
            [
                ("selected", "fg:magenta noinherit"),
                ("highlighted", "fg:magenta noinherit"),
                ("pointer", "fg:magenta noinherit"),
            ]
        ),
    ).ask()

    if choice is None:
        console.print(f"\n[red]{t('error_no_deep')}[/red]")
        exit(1)

    return choice

def select_llm_provider() -> tuple[str, str]:
    """Select the OpenAI api url using interactive selection."""
    # Define OpenAI api options with their corresponding endpoints
    BASE_URLS = [
        ("OpenAI", "https://api.openai.com/v1"),
        ("Google", "https://generativelanguage.googleapis.com/v1"),
        ("Anthropic", "https://api.anthropic.com/"),
        ("Groq", "https://api.groq.com/openai/v1"),
        ("xAI", "https://api.x.ai/v1"),
        ("Openrouter", "https://openrouter.ai/api/v1"),
        ("Ollama", "http://localhost:11434/v1"),
    ]
    provider_defaults = {
        "openai": "OpenAI",
        "google": "Google",
        "anthropic": "Anthropic",
        "groq": "Groq",
        "xai": "xAI",
        "openrouter": "Openrouter",
        "ollama": "Ollama",
    }
    default_provider = provider_defaults.get(DEFAULT_CONFIG.get("llm_provider", "").lower())
    ordered_urls = BASE_URLS
    if default_provider:
        ordered_urls = [
            option for option in BASE_URLS if option[0] == default_provider
        ] + [option for option in BASE_URLS if option[0] != default_provider]
    
    choice = questionary.select(
        t("provider_prompt"),
        choices=[
            questionary.Choice(display, value=(display, value))
            for display, value in ordered_urls
        ],
        instruction=f"\n- {t('instruction_select')}",
        style=questionary.Style(
            [
                ("selected", "fg:magenta noinherit"),
                ("highlighted", "fg:magenta noinherit"),
                ("pointer", "fg:magenta noinherit"),
            ]
        ),
    ).ask()
    
    if choice is None:
        console.print(f"\n[red]{t('error_no_provider')}[/red]")
        exit(1)
    
    display_name, url = choice
    console.print(f"[green]{t('provider_selected')}:[/green] {display_name} ({url})")

    return display_name, url


def ask_openai_reasoning_effort() -> str:
    """Ask for OpenAI reasoning effort level."""
    choices = [
        questionary.Choice(t("effort_medium_default"), "medium"),
        questionary.Choice(t("effort_high_thorough"), "high"),
        questionary.Choice(t("effort_low_fast"), "low"),
    ]
    return questionary.select(
        t("reasoning_select"),
        choices=choices,
        style=questionary.Style([
            ("selected", "fg:cyan noinherit"),
            ("highlighted", "fg:cyan noinherit"),
            ("pointer", "fg:cyan noinherit"),
        ]),
    ).ask()


def ask_anthropic_effort() -> str | None:
    """Ask for Anthropic effort level.

    Controls token usage and response thoroughness on Claude 4.5+ and 4.6 models.
    """
    return questionary.select(
        t("effort_select"),
        choices=[
            questionary.Choice(t("effort_high_recommended"), "high"),
            questionary.Choice(t("effort_medium_balanced"), "medium"),
            questionary.Choice(t("effort_low_cheaper"), "low"),
        ],
        style=questionary.Style([
            ("selected", "fg:cyan noinherit"),
            ("highlighted", "fg:cyan noinherit"),
            ("pointer", "fg:cyan noinherit"),
        ]),
    ).ask()


def ask_gemini_thinking_config() -> str | None:
    """Ask for Gemini thinking configuration.

    Returns thinking_level: "high" or "minimal".
    Client maps to appropriate API param based on model series.
    """
    return questionary.select(
        t("thinking_mode_select"),
        choices=[
            questionary.Choice(t("thinking_enabled"), "high"),
            questionary.Choice(t("thinking_minimal"), "minimal"),
        ],
        style=questionary.Style([
            ("selected", "fg:green noinherit"),
            ("highlighted", "fg:green noinherit"),
            ("pointer", "fg:green noinherit"),
        ]),
    ).ask()
