from __future__ import annotations

import json
import re
from typing import Any

from zhipuai import ZhipuAI


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

    def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        if not self._client:
            raise ValueError("Missing Zhipu API key.")

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        default: Any,
        temperature: float = 0.1,
    ) -> Any:
        if not self._client:
            return default

        raw = self.complete(
            system_prompt=system_prompt,
            user_prompt=f"{user_prompt}\n\nReturn valid JSON only.",
            temperature=temperature,
        )
        try:
            return json.loads(_extract_json_block(raw))
        except json.JSONDecodeError:
            return default
