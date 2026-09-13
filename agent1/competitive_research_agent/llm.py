from __future__ import annotations

import json
import re
from typing import Any, Callable

from zhipuai import ZhipuAI

from .observability import current_collector, measure_current


def _extract_json_block(text: str) -> str:
    fenced = re.findall(r"```json\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced[0].strip()

    generic = re.findall(r"```(.*?)```", text, re.DOTALL)
    if generic:
        return generic[0].strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    return text.strip()


class GLMClient:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key.strip()
        self.model = model
        self._client = ZhipuAI(api_key=self.api_key) if self.api_key else None

    @property
    def is_available(self) -> bool:
        return self._client is not None

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        if not self._client:
            raise ValueError("Missing Zhipu API key.")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        with measure_current("llm_api"):
            if on_token is None:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                )
                content = response.choices[0].message.content.strip()
                usage = getattr(response, "usage", None)
            else:
                chunks: list[str] = []
                usage = None
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    stream=True,
                )
                for chunk in response:
                    usage = getattr(chunk, "usage", None) or usage
                    token = chunk.choices[0].delta.content if chunk.choices else ""
                    if token:
                        chunks.append(token)
                        on_token(token)
                content = "".join(chunks).strip()
        collector = current_collector()
        if collector is not None:
            collector.record_llm_call(_usage_to_dict(usage))
        return content

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        default: Any,
        temperature: float = 0.1,
    ) -> Any:
        payload, _ = self.complete_json_with_meta(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            default=default,
            temperature=temperature,
        )
        return payload

    def complete_json_with_meta(
        self,
        system_prompt: str,
        user_prompt: str,
        default: Any,
        temperature: float = 0.1,
    ) -> tuple[Any, dict[str, Any]]:
        if not self._client:
            return default, {"used_default": True, "parse_success": False, "client_available": False}

        raw = self.complete(
            system_prompt=system_prompt,
            user_prompt=f"{user_prompt}\n\nReturn valid JSON only.",
            temperature=temperature,
        )
        try:
            return json.loads(_extract_json_block(raw)), {
                "used_default": False,
                "parse_success": True,
                "client_available": True,
            }
        except json.JSONDecodeError:
            return default, {
                "used_default": True,
                "parse_success": False,
                "client_available": True,
            }


def _usage_to_dict(usage: Any) -> dict[str, int]:
    if usage is None:
        return {}
    if isinstance(usage, dict):
        raw = usage
    elif hasattr(usage, "model_dump"):
        raw = usage.model_dump()
    else:
        raw = {
            key: getattr(usage, key, 0)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        }
    return {
        key: int(raw.get(key, 0) or 0)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
