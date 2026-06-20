"""LLM client: supports Anthropic, DeepSeek, and OpenAI-compatible APIs."""

from __future__ import annotations

import logging
import threading

from paper_agent.config import Config, load_config

logger = logging.getLogger(__name__)


class LLMClient:
    """Multi-provider LLM client.

    Supports:
        - Anthropic (Claude) via anthropic SDK
        - DeepSeek (deepseek-chat / deepseek-reasoner) via OpenAI-compatible SDK
        - Any OpenAI-compatible provider via openai SDK + base_url
    """

    def __init__(self, config: Config | None = None):
        self.config = config or load_config()
        self._provider: str | None = None
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the appropriate client based on provider."""
        api_key = self.config.llm.api_key
        provider = self.config.llm.provider

        if not api_key:
            self._client = None
            logger.info("No API key configured — LLM calls will be unavailable")
            return

        if provider == "deepseek" or provider == "openai":
            self._init_openai()
        else:
            self._init_anthropic()

    def _init_anthropic(self) -> None:
        """Initialize Anthropic client."""
        try:
            import anthropic

            self._provider = "anthropic"
            self._client = anthropic.AsyncAnthropic(api_key=self.config.llm.api_key)
            logger.info("LLM: Anthropic (%s)", self.config.llm.model)
        except ImportError:
            self._client = None
            logger.warning("anthropic SDK not installed")

    def _init_openai(self) -> None:
        """Initialize OpenAI-compatible client (DeepSeek, OpenAI, etc.)."""
        try:
            from openai import AsyncOpenAI

            base_url = self.config.llm.base_url or "https://api.deepseek.com/v1"
            self._provider = "openai"
            self._client = AsyncOpenAI(
                api_key=self.config.llm.api_key,
                base_url=base_url,
            )
            logger.info("LLM: OpenAI-compatible (%s) @ %s", self.config.llm.model, base_url)
        except ImportError:
            self._client = None
            logger.warning("openai SDK not installed — run: pip install openai")

    @property
    def available(self) -> bool:
        """Whether the LLM client is configured and ready."""
        return self._client is not None

    @property
    def provider(self) -> str:
        """Return the active provider name."""
        return self._provider or "none"

    async def generate(
        self,
        system: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate a completion from the LLM.

        Args:
            system: System prompt.
            messages: List of {'role': 'user'|'assistant', 'content': '...'} dicts.
            max_tokens: Override config max_tokens.
            temperature: Override config temperature.

        Returns:
            The generated text response.

        Raises:
            RuntimeError: If no API key is configured.
        """
        if not self._client:
            raise RuntimeError(
                "LLM client not available. Set ANTHROPIC_API_KEY or DEEPSEEK_API_KEY "
                "in .env or config.toml."
            )

        if self._provider == "openai":
            return await self._generate_openai(system, messages, max_tokens, temperature)
        else:
            return await self._generate_anthropic(system, messages, max_tokens, temperature)

    async def _generate_anthropic(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int | None,
        temperature: float | None,
    ) -> str:
        """Anthropic Messages API call."""
        max_tok = max_tokens or self.config.llm.max_tokens
        temp = temperature if temperature is not None else self.config.llm.temperature

        anthropic_messages = []
        for m in messages:
            anthropic_messages.append({"role": m["role"], "content": m["content"]})

        response = await self._client.messages.create(
            model=self.config.llm.model,
            max_tokens=max_tok,
            temperature=temp,
            system=system,
            messages=anthropic_messages,
        )

        for block in response.content:
            if hasattr(block, "text"):
                return block.text

        return ""

    async def _generate_openai(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int | None,
        temperature: float | None,
    ) -> str:
        """OpenAI-compatible Chat Completions API call."""
        max_tok = max_tokens or self.config.llm.max_tokens
        temp = temperature if temperature is not None else self.config.llm.temperature

        # Build messages list with system prompt
        openai_messages = [{"role": "system", "content": system}]
        for m in messages:
            openai_messages.append({"role": m["role"], "content": m["content"]})

        response = await self._client.chat.completions.create(
            model=self.config.llm.model,
            max_tokens=max_tok,
            temperature=temp,
            messages=openai_messages,
        )

        choice = response.choices[0]
        return choice.message.content or ""

    async def generate_with_json(
        self,
        system: str,
        prompt: str,
        *,
        max_tokens: int | None = None,
    ) -> dict:
        """Generate a completion and parse it as JSON.

        Used for structured extraction (claims, comparisons, etc.).

        Args:
            system: System prompt.
            prompt: User prompt (should instruct JSON output).
            max_tokens: Override config max_tokens.

        Returns:
            Parsed JSON dict. On parse failure, returns a dict with
            `_parse_error: true` and `_raw_text` for debugging.
        """
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
        """Return a placeholder response when no API key is set."""
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


def reset_llm() -> None:
    """Reset the global LLM client (useful for testing or config changes)."""
    global _client
    with _lock:
        _client = None
