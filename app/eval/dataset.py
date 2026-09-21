from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


EVAL_CATEGORIES = {"skills", "experience", "comparison", "filters", "insufficient_evidence"}


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    category: str
    query: str
    filters: dict[str, str] = field(default_factory=dict)
    expected_terms: tuple[str, ...] = ()
    relevant_job_ids: tuple[str, ...] = ()
    should_refuse: bool = False
    notes: str = ""


def load_questions(path: str | Path) -> list[EvalQuestion]:
    questions: list[EvalQuestion] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            raw: dict[str, Any] = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"评估问题第 {line_number} 行不是合法 JSON") from exc
        question_id = str(raw.get("id", "")).strip()
        category = str(raw.get("category", "")).strip()
        query = str(raw.get("query", "")).strip()
        if not question_id or question_id in seen_ids:
            raise ValueError(f"评估问题 id 缺失或重复：{question_id or line_number}")
        if category not in EVAL_CATEGORIES:
            raise ValueError(f"不支持的评估类别：{category}")
        if not query:
            raise ValueError(f"评估问题 {question_id} 缺少 query")
        filters = raw.get("filters") or {}
        if not isinstance(filters, dict):
            raise ValueError(f"评估问题 {question_id} 的 filters 必须是对象")
        seen_ids.add(question_id)
        questions.append(
            EvalQuestion(
                id=question_id,
                category=category,
                query=query,
                filters={str(key): str(value) for key, value in filters.items() if value},
                expected_terms=tuple(str(term).casefold() for term in raw.get("expected_terms", [])),
                relevant_job_ids=tuple(str(job_id) for job_id in raw.get("relevant_job_ids", [])),
                should_refuse=bool(raw.get("should_refuse", False)),
                notes=str(raw.get("notes", "")),
            )
        )
    if not questions:
        raise ValueError("评估问题集为空")
    return questions

