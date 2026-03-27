import os
import json
import re
import uuid
from copy import deepcopy
from typing import Any, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model


_MAX_TPM_ENV = "TRADINGAGENTS_MAX_TPM"
_FAILED_GENERATION_PATTERN = re.compile(
    r"""failed_generation['"]?\s*[:=]\s*['"](?P<generation><function=(?P<name>[a-zA-Z0-9_/-]+)\s*(?P<args>\{.*?\})</function>)['"]""",
    re.DOTALL,
)


def _get_positive_int_env(name: str) -> Optional[int]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _get_nonempty_env(name: str) -> Optional[str]:
    value = os.getenv(name, "").strip()
    return value or None


def _trim_text(text: str, limit: int = 120) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _strip_schema_metadata(schema: Any) -> Any:
    if isinstance(schema, dict):
        compact = {}
        for key, value in schema.items():
            if key in {"description", "default", "title", "examples"}:
                continue
            compact[key] = _strip_schema_metadata(value)
        return compact
    if isinstance(schema, list):
        return [_strip_schema_metadata(item) for item in schema]
    return schema


def _compact_tool_definition(tool: Any) -> dict[str, Any]:
    tool_def = deepcopy(convert_to_openai_tool(tool))
    function_def = tool_def.get("function", {})
    compact = {
        "type": "function",
        "function": {
            "name": function_def.get("name", "tool"),
        },
    }

    description = function_def.get("description")
    if description:
        compact["function"]["description"] = _trim_text(description)

    parameters = function_def.get("parameters")
    if parameters:
        compact["function"]["parameters"] = _strip_schema_metadata(parameters)

    return compact


def _estimate_token_count(payload: Any) -> int:
    if payload is None:
        return 0
    if isinstance(payload, str):
        return max(1, len(payload) // 4)
    if isinstance(payload, (int, float, bool)):
        return 1
    if isinstance(payload, list):
        return sum(_estimate_token_count(item) for item in payload)
    if isinstance(payload, dict):
        return sum(_estimate_token_count(key) + _estimate_token_count(value) for key, value in payload.items())

    content = getattr(payload, "content", None)
    if content is not None:
        return _estimate_token_count(content) + _estimate_token_count(getattr(payload, "tool_calls", None))

    return max(1, len(str(payload)) // 4)


def _recover_tool_call_from_error(err: Exception) -> Optional[AIMessage]:
    match = _FAILED_GENERATION_PATTERN.search(str(err))
    if not match:
        return None

    tool_name = match.group("name")
    raw_args = match.group("args")
    try:
        parsed_args = json.loads(raw_args)
    except json.JSONDecodeError:
        return None

    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": tool_name,
                "args": parsed_args,
                "id": f"recovered_{uuid.uuid4().hex[:12]}",
                "type": "tool_call",
            }
        ],
    )


class NormalizedChatOpenAI(ChatOpenAI):
    """ChatOpenAI with normalized content output.

    The Responses API returns content as a list of typed blocks
    (reasoning, text, etc.). This normalizes to string for consistent
    downstream handling.
    """

    def bind_tools(self, tools, **kwargs):
        if _get_positive_int_env(_MAX_TPM_ENV):
            tools = [_compact_tool_definition(tool) for tool in tools]
            kwargs.setdefault("parallel_tool_calls", False)
        return super().bind_tools(tools, **kwargs)

    def invoke(self, input, config=None, **kwargs):
        budget = _get_positive_int_env(_MAX_TPM_ENV)
        if budget:
            prompt_estimate = _estimate_token_count(input) + _estimate_token_count(kwargs.get("tools"))
            safety_margin = min(512, max(128, budget // 12))
            min_completion_tokens = min(256, max(64, budget // 20))
            max_completion_tokens = max(min_completion_tokens, budget // 3)
            remaining_tokens = budget - prompt_estimate - safety_margin
            capped_tokens = max(min_completion_tokens, min(max_completion_tokens, remaining_tokens))

            existing_limit = kwargs.get("max_tokens")
            existing_completion_limit = kwargs.get("max_completion_tokens")
            model_limit = getattr(self, "max_tokens", None)
            model_completion_limit = getattr(self, "max_completion_tokens", None)
            candidates = [
                limit
                for limit in (
                    existing_limit,
                    existing_completion_limit,
                    model_limit,
                    model_completion_limit,
                    capped_tokens,
                )
                if isinstance(limit, int) and limit > 0
            ]
            final_limit = min(candidates) if candidates else capped_tokens
            kwargs["max_tokens"] = final_limit
            kwargs["max_completion_tokens"] = final_limit

        try:
            return normalize_content(super().invoke(input, config, **kwargs))
        except Exception as err:
            recovered = _recover_tool_call_from_error(err)
            if recovered is not None:
                return recovered
            raise

# Kwargs forwarded from user config to ChatOpenAI
_PASSTHROUGH_KWARGS = (
    "timeout", "max_retries", "reasoning_effort",
    "api_key", "callbacks", "http_client", "http_async_client",
)

# Provider base URLs and API key env vars
_PROVIDER_CONFIG = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "xai": ("https://api.x.ai/v1", "XAI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "ollama": ("http://localhost:11434/v1", None),
}


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI-compatible providers.

    For native OpenAI models, uses the Responses API (/v1/responses) which
    supports reasoning_effort with function tools across all model families
    (GPT-4.1, GPT-5). Third-party compatible providers (Groq, xAI,
    OpenRouter, Ollama) use standard Chat Completions.
    """

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        provider: str = "openai",
        **kwargs,
    ):
        super().__init__(model, base_url, **kwargs)
        self.provider = provider.lower()

    def get_llm(self) -> Any:
        """Return configured ChatOpenAI instance."""
        llm_kwargs = {"model": self.model}

        # Provider-specific base URL and auth
        if self.provider in _PROVIDER_CONFIG:
            base_url, api_key_env = _PROVIDER_CONFIG[self.provider]
            llm_kwargs["base_url"] = base_url
            if api_key_env:
                api_key = os.environ.get(api_key_env)
                if api_key:
                    llm_kwargs["api_key"] = api_key
            else:
                llm_kwargs["api_key"] = "ollama"
        elif self.base_url:
            llm_kwargs["base_url"] = self.base_url

        # Forward user-provided kwargs
        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        # Native OpenAI: use Responses API for consistent behavior across
        # all model families. Third-party providers use Chat Completions.
        if self.provider == "openai":
            llm_kwargs["use_responses_api"] = True
        elif self.provider == "groq":
            constrained_quick_model = _get_nonempty_env("TRADINGAGENTS_GROQ_CONSTRAINED_QUICK_MODEL")
            if _get_positive_int_env(_MAX_TPM_ENV) and self.model == "llama-3.1-8b-instant":
                llm_kwargs["model"] = constrained_quick_model or "llama-3.3-70b-versatile"

        return NormalizedChatOpenAI(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for the provider."""
        return validate_model(self.provider, self.model)
