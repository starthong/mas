from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from openai import OpenAI


class LLMClient:
    """OpenAI 兼容对话接口的轻量封装。

    base 模型和 LoRA 模型共用同一客户端，只改变模型名，
    以保证系统实验的其他变量不变。
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        default_model: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url or os.getenv(
            "OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"
        )
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "EMPTY")
        self.default_model = default_model or os.getenv("FT_MODEL", "toolcall")
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=timeout,
        )
        self.usage = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
        self.last_usage = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        if temperature is None:
            temperature = float(os.getenv("TEMPERATURE", "0"))
        if max_tokens is None:
            max_tokens = int(os.getenv("MAX_TOKENS", "512"))

        response = self.client.chat.completions.create(
            model=model or self.default_model,
            messages=list(messages),  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        self.last_usage = {
            "calls": 1,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
        self.usage["calls"] += 1
        self.usage["prompt_tokens"] += prompt_tokens
        self.usage["completion_tokens"] += completion_tokens
        return response.choices[0].message.content or ""
