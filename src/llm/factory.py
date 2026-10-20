from __future__ import annotations

from typing import Any


def build_client(backend: str = "openai", **kwargs: Any):
    if backend == "openai":
        from src.llm.client import LLMClient

        return LLMClient(**kwargs)
    if backend == "hf":
        from src.llm.hf_client import HFClient

        return HFClient(**kwargs)
    raise ValueError(f"unknown backend: {backend}")
