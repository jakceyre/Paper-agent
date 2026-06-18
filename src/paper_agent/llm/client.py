"""LLM client: wraps the Anthropic SDK for synchronous and async use."""

from __future__ import annotations

import logging
import threading

from paper_agent.config import Config, load_config

logger = logging.getLogger(__name__)


class LLMClient:
    """Thin wrapper around the Anthropic Messages API."""

    def __init__(self, config: Config | None = None):
        self.config = config or load_config()
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the Anthropic client if an API key is available."""
        if not self.config.llm.api_key:
            self._client = None
            return
        try:
            import anthropic
            self._client = anthropic.AsyncAnthropic(api_key=self.config.llm.api_key)
        except ImportError:
            self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    async def generate(
        self,
        system: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        if not self._client:
            raise RuntimeError(
                "LLM client not available. Set ANTHROPIC_API_KEY in .env or config.toml."
            )

        anthropic_messages = []
        for m in messages:
            anthropic_messages.append({"role": m["role"], "content": m["content"]})

        response = await self._client.messages.create(
            model=self.config.llm.model,
            max_tokens=max_tokens or self.config.llm.max_tokens,
            temperature=temperature or self.config.llm.temperature,
            system=system,
            messages=anthropic_messages,
        )

        for block in response.content:
            if hasattr(block, "text"):
                return block.text

        return ""

    async def generate_with_json(
        self,
        system: str,
        prompt: str,
        *,
        max_tokens: int | None = None,
    ) -> dict:
        import json

        messages = [{"role": "user", "content": prompt}]
        text = await self.generate(system, messages, max_tokens=max_tokens)

        if not text:
            logger.warning("generate_with_json: LLM returned empty response")
            return {"_parse_error": True, "_raw_text": "", "_reason": "empty response"}

        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            return json.loads(text[start:end])
        except ValueError:
            logger.warning(
                "generate_with_json: No JSON object found in LLM response. "
                "Response preview: %s...",
                text[:200],
            )
            return {
                "_parse_error": True,
                "_raw_text": text[:500],
                "_reason": "no JSON object found in response",
            }
        except json.JSONDecodeError as e:
            logger.warning(
                "generate_with_json: Invalid JSON in LLM response: %s. "
                "Response preview: %s...",
                e,
                text[:200],
            )
            return {
                "_parse_error": True,
                "_raw_text": text[:500],
                "_reason": f"JSON decode error: {e}",
            }

    async def generate_stub(self, prompt: str) -> str:
        if self.available:
            return await self.generate("", [{"role": "user", "content": prompt}])
        return f"[STUB] LLM not configured. Would respond to: {prompt[:100]}..."


# Global singleton (thread-safe)
_client: LLMClient | None = None
_lock = threading.Lock()


def get_llm() -> LLMClient:
    """Get or create the global LLM client singleton (thread-safe)."""
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = LLMClient()
    return _client
