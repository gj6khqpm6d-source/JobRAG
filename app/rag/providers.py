from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path
from threading import Lock

import requests

from app.config import PROJECT_ROOT, Settings, settings


class EmbeddingProviderError(RuntimeError):
    pass


class LLMProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class GenerationResult:
    text: str
    usage: dict[str, int] = field(default_factory=dict)
    model: str = ""


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class LocalBGEProvider(EmbeddingProvider):
    """Lazy, in-process SentenceTransformers provider for offline embeddings."""

    def __init__(self, config: Settings = settings) -> None:
        self.config = config
        self._model = None
        self._load_lock = Lock()

    def _load_model(self):
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:
                return self._model
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingProviderError("本地 Embedding 依赖未安装，请重新运行 ./run.sh") from exc

            cache_path = Path(self.config.model_cache_dir).expanduser()
            if not cache_path.is_absolute():
                cache_path = PROJECT_ROOT / cache_path
            cache_path.mkdir(parents=True, exist_ok=True)
            device = None if self.config.embedding_device == "auto" else self.config.embedding_device
            try:
                self._model = SentenceTransformer(
                    self.config.embedding_model,
                    revision=self.config.embedding_revision or None,
                    device=device,
                cache_folder=str(cache_path),
                model_kwargs={"use_safetensors": True},
            )
            except Exception as exc:
                raise EmbeddingProviderError(f"无法加载本地 Embedding 模型：{exc}") from exc
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            vectors = self._load_model().encode(
                texts,
                batch_size=self.config.embedding_batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except EmbeddingProviderError:
            raise
        except Exception as exc:
            raise EmbeddingProviderError(f"本地 Embedding 生成失败：{exc}") from exc
        output = vectors.tolist()
        expected = self.config.embedding_dimensions
        if any(len(vector) != expected for vector in output):
            raise EmbeddingProviderError(f"Embedding 维度与配置不一致，预期 {expected}")
        return output


class CloudflareBGEProvider(EmbeddingProvider):
    def __init__(self, config: Settings = settings, timeout: int = 60) -> None:
        self.config = config
        self.timeout = timeout

    @property
    def endpoint(self) -> str:
        return (
            "https://api.cloudflare.com/client/v4/accounts/"
            f"{self.config.cloudflare_account_id}/ai/run/{self.config.embedding_model}"
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self.config.cloudflare_account_id or not self.config.cloudflare_api_token:
            raise EmbeddingProviderError("尚未配置 CLOUDFLARE_ACCOUNT_ID 和 CLOUDFLARE_API_TOKEN")
        try:
            response = requests.post(
                self.endpoint,
                headers={"Authorization": f"Bearer {self.config.cloudflare_api_token}"},
                json={"text": texts},
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise EmbeddingProviderError(f"无法连接 Cloudflare Embedding 服务：{exc}") from exc
        if not response.ok:
            raise EmbeddingProviderError(f"Cloudflare Embedding 请求失败（HTTP {response.status_code}）")
        try:
            payload = response.json()
        except ValueError as exc:
            raise EmbeddingProviderError("Cloudflare Embedding 返回了非 JSON 响应") from exc
        if not payload.get("success", True):
            messages = "; ".join(error.get("message", "unknown error") for error in payload.get("errors", []))
            raise EmbeddingProviderError(f"Cloudflare Embedding 返回错误：{messages}")
        vectors = payload.get("result", {}).get("data")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise EmbeddingProviderError("Cloudflare Embedding 返回了无法识别的数据格式")
        expected = self.config.embedding_dimensions
        if any(not isinstance(vector, list) or len(vector) != expected for vector in vectors):
            raise EmbeddingProviderError(f"Embedding 维度与配置不一致，预期 {expected}")
        return vectors


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, *, system: str, user: str) -> str:
        raise NotImplementedError

    def generate_with_metadata(self, *, system: str, user: str) -> GenerationResult:
        return GenerationResult(text=self.generate(system=system, user=user))


class DeepSeekProvider(LLMProvider):
    def __init__(self, config: Settings = settings, timeout: int = 120) -> None:
        self.config = config
        self.timeout = timeout

    def generate(self, *, system: str, user: str) -> str:
        return self.generate_with_metadata(system=system, user=user).text

    def generate_with_metadata(self, *, system: str, user: str) -> GenerationResult:
        if not self.config.deepseek_api_key or not self.config.deepseek_model:
            raise LLMProviderError("尚未配置 DEEPSEEK_API_KEY 和 DEEPSEEK_MODEL")
        try:
            response = requests.post(
                f"{self.config.deepseek_base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {self.config.deepseek_api_key}"},
                json={
                    "model": self.config.deepseek_model,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    "temperature": 0.1,
                    "thinking": {"type": "disabled"},
                    "max_tokens": self.config.deepseek_max_tokens,
                    "stream": False,
                },
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise LLMProviderError(f"无法连接 DeepSeek 服务：{exc}") from exc
        if not response.ok:
            raise LLMProviderError(f"DeepSeek 请求失败（HTTP {response.status_code}）")
        try:
            payload: dict[str, Any] = response.json()
            usage = payload.get("usage") or {}
            return GenerationResult(
                text=payload["choices"][0]["message"]["content"],
                usage={
                    "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                    "completion_tokens": int(usage.get("completion_tokens") or 0),
                    "total_tokens": int(usage.get("total_tokens") or 0),
                },
                model=str(payload.get("model") or self.config.deepseek_model),
            )
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError("DeepSeek 返回了无法识别的数据格式") from exc


_local_provider: LocalBGEProvider | None = None
_provider_lock = Lock()


def get_embedding_provider(config: Settings = settings) -> EmbeddingProvider:
    global _local_provider
    if config.embedding_provider == "cloudflare":
        return CloudflareBGEProvider(config)
    if config.embedding_provider != "local":
        raise EmbeddingProviderError(f"不支持的 Embedding Provider：{config.embedding_provider}")
    with _provider_lock:
        if _local_provider is None or _local_provider.config != config:
            _local_provider = LocalBGEProvider(config)
        return _local_provider
