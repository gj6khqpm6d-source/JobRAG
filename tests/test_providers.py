from dataclasses import replace

from app.config import settings
from app.rag.providers import CloudflareBGEProvider, DeepSeekProvider


class FakeResponse:
    ok = True
    status_code = 200

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


def test_cloudflare_provider_parses_vectors(monkeypatch):
    config = replace(
        settings,
        cloudflare_account_id="account",
        cloudflare_api_token="token",
        embedding_dimensions=3,
    )
    monkeypatch.setattr(
        "app.rag.providers.requests.post",
        lambda *args, **kwargs: FakeResponse({"success": True, "result": {"data": [[0.1, 0.2, 0.3]]}}),
    )
    assert CloudflareBGEProvider(config).embed(["job requirements"]) == [[0.1, 0.2, 0.3]]


def test_deepseek_provider_parses_answer(monkeypatch):
    config = replace(settings, deepseek_api_key="token", deepseek_model="deepseek-chat")
    monkeypatch.setattr(
        "app.rag.providers.requests.post",
        lambda *args, **kwargs: FakeResponse({"choices": [{"message": {"content": "answer [1]"}}]}),
    )
    assert DeepSeekProvider(config).generate(system="grounded", user="question") == "answer [1]"


def test_deepseek_provider_exposes_token_usage(monkeypatch):
    config = replace(settings, deepseek_api_key="token", deepseek_model="deepseek-chat")
    monkeypatch.setattr(
        "app.rag.providers.requests.post",
        lambda *args, **kwargs: FakeResponse(
            {
                "model": "deepseek-chat",
                "choices": [{"message": {"content": "answer [1]"}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
            }
        ),
    )
    result = DeepSeekProvider(config).generate_with_metadata(system="grounded", user="question")
    assert result.text == "answer [1]"
    assert result.usage["total_tokens"] == 25
    assert result.model == "deepseek-chat"
