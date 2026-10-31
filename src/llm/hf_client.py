from __future__ import annotations

from collections.abc import Sequence
from typing import Any


class HFClient:
    """本地 transformers/peft 后端，主要用于 Kaggle 评测。

    对外提供与 ``LLMClient`` 一致的 ``chat`` 接口，
    因此评测和 Agent 代码不需要感知后端差异。
    """

    def __init__(
        self,
        base_model: str,
        adapter: str | None = None,
        load_in_4bit: bool = True,
        max_new_tokens: int = 512,
        device_map: str = "auto",
    ) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "device_map": device_map,
        }
        if load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs["torch_dtype"] = torch.float16

        model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
        if adapter:
            model = PeftModel.from_pretrained(model, adapter)
        model.eval()

        self.model = model
        self.max_new_tokens = max_new_tokens
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
        # 为保持接口兼容，接受 ``model`` 参数但忽略它。
        del model
        text = self.tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        do_sample = bool(temperature and temperature > 0)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens or self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = float(temperature)

        with self.torch.no_grad():
            output = self.model.generate(**inputs, **generation_kwargs)

        generated = output[0][inputs["input_ids"].shape[-1] :]
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        completion_tokens = int(generated.shape[-1])
        self.last_usage = {
            "calls": 1,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
        self.usage["calls"] += 1
        self.usage["prompt_tokens"] += prompt_tokens
        self.usage["completion_tokens"] += completion_tokens
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
