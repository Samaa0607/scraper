"""
Thin wrapper around the Anthropic Claude API.

The LLM is optional: if no credentials are configured, `is_available()` returns
False and the RAG/agent layers fall back to returning retrieved passages only.

Credentials are read by the SDK from the ANTHROPIC_API_KEY environment variable
(or ANTHROPIC_AUTH_TOKEN). The key is never stored in code, logged or returned.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from config import LLM_MAX_TOKENS, LLM_MODEL, LLM_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

_client = None


class LLMError(RuntimeError):
    pass


def is_available() -> bool:
    """True if the anthropic package is installed and credentials are configured."""
    key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN") or ""
    if not key.strip():
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


def _get_client():
    global _client
    if _client is None:
        import anthropic
        # A web request should not hang for the SDK's 10-minute default
        _client = anthropic.Anthropic(timeout=LLM_TIMEOUT_SECONDS, max_retries=2)
    return _client


def create_message(
    system: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
):
    """
    Send one Messages API request and return the response.
    Raises LLMError on API errors or a safety refusal.
    """
    import anthropic

    kwargs: Dict[str, Any] = dict(
        model=LLM_MODEL,
        max_tokens=LLM_MAX_TOKENS,
        system=system,
        messages=messages,
        output_config={"effort": "medium"},
        # If the model declines a request, let the API retry it on Anthropic's
        # recommended fallback model instead of returning a refusal.
        betas=["server-side-fallback-2026-07-01"],
        extra_body={"fallbacks": "default"},
    )
    if tools:
        kwargs["tools"] = tools

    try:
        response = _get_client().beta.messages.create(**kwargs)
    except anthropic.AuthenticationError as e:
        raise LLMError("Invalid or missing Anthropic API credentials") from e
    except anthropic.RateLimitError as e:
        raise LLMError("LLM rate limit reached, try again later") from e
    except anthropic.APIStatusError as e:
        raise LLMError(f"LLM API error ({e.status_code}): {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise LLMError("Could not connect to the LLM API") from e

    if response.stop_reason == "refusal":
        raise LLMError("The model declined to answer this request")
    if response.stop_reason == "max_tokens":
        logger.warning("LLM response was truncated (max_tokens reached)")
    return response


def response_text(response) -> str:
    """Concatenate the text blocks of a response."""
    return "\n".join(b.text for b in response.content if b.type == "text").strip()
