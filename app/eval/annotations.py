from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from threading import RLock
from typing import Any


DEFAULT_ANNOTATION_PATH = Path(__file__).resolve().parents[2] / "data" / "eval" / "annotations.json"
_lock = RLock()


def _read(path: Path = DEFAULT_ANNOTATION_PATH) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "annotations": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取评估标注文件：{path}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("annotations", {}), dict):
        raise ValueError("评估标注文件格式无效")
    return payload


def all_annotations(path: Path = DEFAULT_ANNOTATION_PATH) -> dict[str, dict[str, Any]]:
    with _lock:
        return dict(_read(path).get("annotations", {}))


def get_annotation(question_id: str, path: Path = DEFAULT_ANNOTATION_PATH) -> dict[str, Any] | None:
    return all_annotations(path).get(question_id)


def save_annotation(question_id: str, annotation: dict[str, Any], path: Path = DEFAULT_ANNOTATION_PATH) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        payload = _read(path)
        payload.setdefault("version", 1)
        payload.setdefault("annotations", {})[question_id] = annotation
        fd, temp_name = tempfile.mkstemp(prefix="annotations-", suffix=".json", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
    return annotation
