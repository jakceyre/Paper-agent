"""Configuration management: load from TOML file, override with env vars.

Priority: env vars > config.toml > dataclass defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # fallback for older Python


@dataclass
class LLMConfig:
    """LLM provider configuration. Supports Anthropic and DeepSeek."""

    provider: str = "anthropic"
    """'anthropic' or 'deepseek'."""

    model: str = "claude-sonnet-4-20250514"
    """Model name. DeepSeek example: 'deepseek-chat'."""

    max_tokens: int = 4096
    temperature: float = 0.3
    base_url: str = ""
    """Override API base URL. DeepSeek default: https://api.deepseek.com/v1."""

    api_key: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "")
        or os.getenv("OPENAI_API_KEY", "")  # DeepSeek uses OPENAI_API_KEY too
        or os.getenv("DEEPSEEK_API_KEY", "")
    )


@dataclass
class SearchConfig:
    """Paper search configuration."""

    arxiv_max_results: int = 50
    semantic_scholar_max_results: int = 50
    default_year_range: tuple[int, int] = (2020, 2025)
    max_papers: int = 20
    citation_expansion_depth: int = 1


@dataclass
class PDFConfig:
    """PDF download and parsing configuration."""

    download_dir: str = "outputs/paper-agent/pdf_cache"
    download_timeout_sec: int = 30
    parse_max_pages: int = 30
    parse_chunk_size: int = 3000  # characters per chunk for LLM context


@dataclass
class OutputConfig:
    """Output generation configuration."""

    base_dir: str = "outputs/paper-agent"
    include_trace: bool = True
    include_evidence: bool = True


@dataclass
class Config:
    """Top-level configuration aggregating all sub-configs."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    pdf: PDFConfig = field(default_factory=PDFConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def load_config(config_path: str | Path = "config.toml") -> Config:
    """Load configuration from a TOML file, with env var overrides.

    Args:
        config_path: Path to the TOML configuration file.

    Returns:
        A Config instance with values loaded and overridden.
    """
    path = Path(config_path)
    if not path.exists():
        return Config()

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    cfg = Config()

    # Apply TOML values for each section
    _apply_section(raw, "llm", cfg.llm)
    _apply_section(raw, "search", cfg.search)
    _apply_section(raw, "pdf", cfg.pdf)
    _apply_section(raw, "output", cfg.output)

    # Env var overrides (highest priority)
    if api_key := os.getenv("ANTHROPIC_API_KEY"):
        cfg.llm.api_key = api_key
    elif api_key := os.getenv("DEEPSEEK_API_KEY"):
        cfg.llm.api_key = api_key
    elif api_key := os.getenv("OPENAI_API_KEY"):
        cfg.llm.api_key = api_key

    return cfg


def _apply_section(raw: dict, section: str, target: object) -> None:
    """Apply TOML section values to a dataclass instance."""
    if section not in raw:
        return
    for k, v in raw[section].items():
        if hasattr(target, k):
            if k == "default_year_range" and isinstance(v, list):
                v = tuple(v)
            if k == "base_url":
                v = str(v)
            setattr(target, k, v)
