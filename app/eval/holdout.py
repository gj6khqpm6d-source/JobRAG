from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.db.session import SessionLocal
from app.eval.annotations import DEFAULT_ANNOTATION_PATH, all_annotations
from app.eval.dataset import load_questions
from app.eval.tuning import RetrievalConfig, evaluate_config, load_splits
from app.rag.providers import get_embedding_provider


def _relevance(annotation: dict[str, Any]) -> dict[str, int]:
    return {
        str(item["job_id"]): int(item["relevance"])
        for item in annotation.get("job_judgments", [])
        if item.get("job_id") and item.get("relevance") in (0, 1, 2)
    }


def run_holdout(
    question_path: str | Path,
    annotation_path: str | Path,
    split_path: str | Path,
    locked_config_path: str | Path,
    *,
    min_recall_at_8: float = 0.75,
    min_ndcg_at_8: float = 0.75,
) -> dict[str, Any]:
    questions = {question.id: question for question in load_questions(question_path)}
    annotations = all_annotations(Path(annotation_path))
    splits = load_splits(split_path)
    locked_payload = json.loads(Path(locked_config_path).read_text(encoding="utf-8"))
    raw_config = locked_payload["config"]
    config = RetrievalConfig(**raw_config)
    missing = [question_id for question_id in splits["test"] if question_id not in annotations]
    invalid = []
    answerable = []
    refusal = []
    for question_id in splits["test"]:
        annotation = annotations.get(question_id)
        if annotation is None:
            continue
        if annotation.get("should_refuse", False):
            refusal.append(question_id)
            continue
        if not any(grade > 0 for grade in _relevance(annotation).values()):
            invalid.append(question_id)
            continue
        answerable.append(question_id)
    if missing or invalid:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "invalid_holdout_annotations",
            "locked_config": raw_config,
            "missing_questions": missing,
            "answerable_questions": answerable,
            "refusal_questions": refusal,
            "invalid_questions": invalid,
            "message": "每道非拒答测试题必须至少有一个 relevance=1 或 2 的岗位；证据不足的题目应标记为拒答。",
            "passed": False,
        }
    answerable_questions = [questions[question_id] for question_id in answerable]
    vectors_list = get_embedding_provider().embed([question.query for question in answerable_questions])
    vectors = {question.id: vector for question, vector in zip(answerable_questions, vectors_list)}
    evaluated = evaluate_config(config, answerable_questions, annotations, vectors)
    summary = evaluated["summary"]
    gates = {
        "recall_at_8": summary["recall_at_8"] >= min_recall_at_8,
        "graded_ndcg_at_8": summary["graded_ndcg_at_8"] >= min_ndcg_at_8,
        "filter_accuracy": summary["filter_accuracy"] >= 1.0,
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "locked_config": raw_config,
        "answerable_questions": answerable,
        "refusal_questions": refusal,
        "summary": summary,
        "gates": gates,
        "passed": all(gates.values()),
        "retrieval_rows": evaluated["rows"],
        "refusal_evaluation": {
            "cases": len(refusal),
            "note": "本阶段只确认人工拒答标签；DeepSeek 是否实际拒答将在答案评估阶段验证。",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the locked JobRAG retrieval config on the holdout set")
    parser.add_argument("--questions", default="data/eval/questions.jsonl")
    parser.add_argument("--annotations", default=str(DEFAULT_ANNOTATION_PATH))
    parser.add_argument("--splits", default="data/eval/splits.json")
    parser.add_argument("--locked-config", default="data/eval/locked-retrieval-config.json")
    parser.add_argument("--output", default="data/eval/reports/retrieval-holdout.json")
    args = parser.parse_args()
    report = run_holdout(args.questions, args.annotations, args.splits, args.locked_config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "summary", "gates", "passed", "message") if key in report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
