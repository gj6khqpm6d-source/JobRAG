from __future__ import annotations

import os
from pathlib import Path
from threading import Lock

import torch

from app.config import PROJECT_ROOT, Settings, settings


class RerankerError(RuntimeError):
    pass


class LocalReranker:
    """Lazy local cross-encoder reranker with no network inference calls."""

    def __init__(self, config: Settings = settings) -> None:
        self.config = config
        self._tokenizer = None
        self._model = None
        self._device = None
        self._load_lock = Lock()

    def _resolve_device(self) -> torch.device:
        requested = self.config.reranker_device
        if requested != "auto":
            return torch.device(requested)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load(self) -> tuple[object, object, torch.device]:
        if self._model is not None:
            return self._tokenizer, self._model, self._device
        with self._load_lock:
            if self._model is not None:
                return self._tokenizer, self._model, self._device
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                cache_path = Path(self.config.model_cache_dir).expanduser()
                if not cache_path.is_absolute():
                    cache_path = PROJECT_ROOT / cache_path
                cache_path.mkdir(parents=True, exist_ok=True)
                device = self._resolve_device()
                if device.type == "cpu":
                    torch.set_num_threads(max(torch.get_num_threads(), min(8, os.cpu_count() or 2)))
                tokenizer = AutoTokenizer.from_pretrained(self.config.reranker_model, cache_dir=str(cache_path))
                model = AutoModelForSequenceClassification.from_pretrained(
                    self.config.reranker_model,
                    cache_dir=str(cache_path),
                )
                model.to(device)
                model.eval()
            except Exception as exc:
                raise RerankerError(f"无法加载本地 Reranker 模型：{exc}") from exc
            self._tokenizer, self._model, self._device = tokenizer, model, device
        return self._tokenizer, self._model, self._device

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        tokenizer, model, device = self._load()
        scores: list[float] = []
        try:
            for start in range(0, len(passages), self.config.reranker_batch_size):
                batch = passages[start : start + self.config.reranker_batch_size]
                encoded = tokenizer(
                    [query] * len(batch),
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.config.reranker_max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                with torch.inference_mode():
                    logits = model(**encoded).logits.reshape(-1).float()
                    scores.extend(torch.sigmoid(logits).cpu().tolist())
        except Exception as exc:
            raise RerankerError(f"本地 Reranker 推理失败：{exc}") from exc
        return scores


_local_reranker: LocalReranker | None = None
_provider_lock = Lock()


def get_reranker(config: Settings = settings) -> LocalReranker:
    global _local_reranker
    with _provider_lock:
        if _local_reranker is None or _local_reranker.config != config:
            _local_reranker = LocalReranker(config)
        return _local_reranker
