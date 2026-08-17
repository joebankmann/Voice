import json

import httpx

from voice.llm import LlmClient


def test_stream_chat_yields_content_tokens():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        chunks = [
            {"choices": [{"delta": {"content": "Hi"}}]},
            {"choices": [{"delta": {"content": "!"}}]},
        ]
        body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    transport = httpx.MockTransport(handler)
    client = LlmClient(
        base_url="http://test/v1",
        model="qwen3-8b",
        temperature=0.7,
        transport=transport,
    )
    tokens = list(client.stream_chat([{"role": "user", "content": "hey"}], system_prompt="Be brief."))
    assert tokens == ["Hi", "!"]


def test_complete_returns_message_content():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        payload = json.loads(request.content)
        assert payload["stream"] is False
        body = json.dumps(
            {"choices": [{"message": {"role": "assistant", "content": "Noted."}}]}
        )
        return httpx.Response(200, text=body)

    client = LlmClient(
        base_url="http://test/v1",
        model="qwen3-8b",
        temperature=0.7,
        transport=httpx.MockTransport(handler),
    )
    text = client.complete(
        [{"role": "user", "content": "summarize"}],
        system_prompt="One sentence.",
    )
    assert text == "Noted."
