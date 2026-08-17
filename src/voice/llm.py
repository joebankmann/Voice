from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx


class LlmClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self._client = httpx.Client(transport=transport, timeout=None)
        self._active: httpx.Response | None = None

    def cancel(self) -> None:
        if self._active is not None:
            self._active.close()
            self._active = None

    def stream_chat(self, messages: list[dict[str, str]], system_prompt: str) -> Iterator[str]:
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "stream": True,
            "messages": [{"role": "system", "content": system_prompt}, *messages],
        }
        with self._client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            json=payload,
            headers={"Accept": "text/event-stream"},
        ) as resp:
            self._active = resp
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    obj = json.loads(data)
                    delta = obj["choices"][0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield content
            self._active = None

    def complete(
        self,
        messages: list[dict[str, str]],
        system_prompt: str,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "stream": False,
            "messages": [{"role": "system", "content": system_prompt}, *messages],
        }
        request = self._client.build_request(
            "POST",
            f"{self.base_url}/chat/completions",
            json=payload,
        )
        response = self._client.send(request, stream=True)
        self._active = response
        try:
            response.raise_for_status()
            body = response.read()
            obj = json.loads(body)
            return str(obj["choices"][0]["message"]["content"])
        finally:
            self._active = None
            response.close()
