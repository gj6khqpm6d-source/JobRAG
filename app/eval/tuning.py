from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.db.session import SessionLocal
from app.eval.annotations import DEFAULT_ANNOTATION_PATH, all_annotations
from app.eval.dataset import EvalQuestion, load_questions
from app.eval.metrics import graded_ndcg_at_k, recall_at_k, reciprocal_rank
from app.eval.runner import question_filters, unique_in_order
from app.rag.providers import get_embedding_provider
from app.rag.retrieval import hybrid_search


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RetrievalConfig:
    mode: str
    candidate_k: int
    max_chunks_per_job: int
    rrf_k: int = 60
    vector_weight: float = 1.0
    keyword_weight: float = 1.0

    @property
    def id(self) -> str:
        return (
            f"{self.mode}-c{self.candidate_k}-m{self.max_chunks_per_job}"
            f"-r{self.rrf_k}-vw{self.vector_weight:g}-kw{self.keyword_weight:g}"
        )


def tuning_configs() -> list[RetrievalConfig]:
    configs: list[RetrievalConfig] = []
    for mode in ("keyword", "vector"):
        for candidate_k in (50, 100):
            for max_chunks in (1, 2):
                configs.append(RetrievalConfig(mode, candidate_k, max_chunks))
    weights = ((1.0, 1.0), (1.5, 1.0), (2.0, 1.0), (1.0, 1.5))
    for candidate_k in (50, 100):
        for max_chunks in (1, 2):
            for rrf_k in (20, 60):
                for vector_weight, keyword_weight in weights:
                    configs.append(
                        RetrievalConfig(
                            "hybrid",
                            candidate_k,
                            max_chunks,
                            rrf_k,
                            vector_weight,
                            keyword_weight,
                        )
                    )
    return configs


def load_splits(path: str | Path) -> dict[str, list[str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"development", "test", "refusal"}
    if set(payload) != required or any(not isinstance(payload[key], list) for key in required):
        raise ValueError("评估集划分必须包含 development、test 和 refusal")
    ids = [str(question_id) for values in payload.values() for question_id in values]
    if len(ids) != len(set(ids)):
        raise ValueError("评估集划分中存在重复问题 ID")
    return {key: [str(value) for value in payload[key]] for key in required}


def annotation_relevance(annotation: dict[str, Any]) -> dict[str, int]:
    return {
        str(item["job_id"]): int(item["relevance"])
        for item in annotation.get("job_judgments", [])
        if item.get("job_id") and item.get("relevance") in (0, 1, 2)
    }


def _filter_match(item: dict[str, Any], question: EvalQuestion) -> bool:
    filters = question.filters
    if filters.get("source") and str(item.get("source", "")).casefold() != filters["source"].casefold():
        return False
    for key in ("location", "title", "job_type"):
        expected = filters.get(key, "").casefold()
        if expected and expected not in str(item.get(key, "")).casefold():
            return False
    return True


def evaluate_config(
    config: RetrievalConfig,
    questions: list[EvalQuestion],
    annotations: dict[str, dict[str, Any]],
    vectors: dict[str, list[float]],
    metric_k: int = 8,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for question in questions:
        relevance = annotation_relevance(annotations[question.id])
        relevant_ids = [job_id for job_id, grade in relevance.items() if grade > 0]
        results = hybrid_search(
            SessionLocal,
            query=question.query,
            query_vector=vectors[question.id],
            filters=question_filters(question),
            top_k=metric_k * config.max_chunks_per_job,
            candidate_k=config.candidate_k,
            max_chunks_per_job=config.max_chunks_per_job,
            mode=config.mode,
            rrf_k=config.rrf_k,
            vector_weight=config.vector_weight,
            keyword_weight=config.keyword_weight,
        )
        result_ids = unique_in_order([item["job_id"] for item in results])[:metric_k]
        rows.append(
            {
                "id": question.id,
                "retrieved_job_ids": result_ids,
                "recall_at_5": recall_at_k(result_ids, relevant_ids, 5),
                "recall_at_8": recall_at_k(result_ids, relevant_ids, 8),
                "mrr_at_8": reciprocal_rank(result_ids, relevant_ids, 8),
                "graded_ndcg_at_8": graded_ndcg_at_k(result_ids, relevance, 8),
                "filter_accuracy": (
                    sum(_filter_match(item, question) for item in results) / len(results)
                    if results
                    else 1.0
                ),
            }
        )
    metric_names = ("recall_at_5", "recall_at_8", "mrr_at_8", "graded_ndcg_at_8", "filter_accuracy")
    summary = {
        name: round(sum(row[name] for row in rows) / len(rows), 4) if rows else 0.0
        for name in metric_names
    }
    return {"config": {"id": config.id, **asdict(config)}, "summary": summary, "rows": rows}


def run_tuning(
    question_path: str | Path,
    annotation_path: str | Path,
    split_path: str | Path,
    *,
    minimum_development_questions: int = 5,
) -> dict[str, Any]:
    questions = {question.id: question for question in load_questions(question_path)}
    annotations = all_annotations(Path(annotation_path))
    splits = load_splits(split_path)
    development = [
        questions[question_id]
        for question_id in splits["development"]
        if question_id in questions
        and question_id in annotations
        and not annotations[question_id].get("should_refuse", False)
        and any(grade > 0 for grade in annotation_relevance(annotations[question_id]).values())
    ]
    base = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "label_policy": "human annotations only; relevance 1 and 2 are relevant, graded nDCG preserves grades",
        "development_questions_available": len(development),
        "minimum_development_questions": minimum_development_questions,
        "annotated_questions_total": len(annotations),
        "test_questions_annotated": sum(question_id in annotations for question_id in splits["test"]),
        "test_set_evaluated": False,
    }
    if len(development) < minimum_development_questions:
        return {
            **base,
            "status": "insufficient_annotations",
            "message": f"至少需要 {minimum_development_questions} 道 development 问题具有正相关岗位标注",
            "experiments": [],
            "best_config": None,
        }

    vectors_list = get_embedding_provider().embed([question.query for question in development])
    vectors = {question.id: vector for question, vector in zip(development, vectors_list)}
    experiments = [
        evaluate_config(config, development, annotations, vectors)
        for config in tuning_configs()
    ]
    experiments.sort(
        key=lambda item: (
            item["summary"]["graded_ndcg_at_8"],
            item["summary"]["recall_at_8"],
            item["summary"]["mrr_at_8"],
        ),
        reverse=True,
    )
    return {
        **base,
        "status": "complete",
        "selection_metric": "graded_ndcg_at_8, then recall_at_8, then mrr_at_8",
        "experiments": experiments,
        "best_config": experiments[0]["config"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune JobRAG retrieval using human relevance judgments")
    parser.add_argument("--questions", default="data/eval/questions.jsonl")
    parser.add_argument("--annotations", default=str(DEFAULT_ANNOTATION_PATH))
    parser.add_argument("--splits", default="data/eval/splits.json")
    parser.add_argument("--output", default="data/eval/reports/retrieval-tuning.json")
    args = parser.parse_args()
    report = run_tuning(args.questions, args.annotations, args.splits)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "message", "best_config") if key in report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
