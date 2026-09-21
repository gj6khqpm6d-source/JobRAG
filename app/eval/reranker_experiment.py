from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.db.session import SessionLocal
from app.eval.annotations import DEFAULT_ANNOTATION_PATH, all_annotations
from app.eval.dataset import load_questions
from app.eval.metrics import graded_ndcg_at_k, recall_at_k, reciprocal_rank
from app.eval.runner import question_filters, unique_in_order
from app.eval.tuning import RetrievalConfig, annotation_relevance, load_splits
from app.rag.providers import get_embedding_provider
from app.rag.reranker import RerankerError, get_reranker
from app.rag.retrieval import hybrid_search


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_locked_config(path: str | Path) -> RetrievalConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return RetrievalConfig(**payload["config"])


def _candidate_pool(question, query_vector: list[float], pool_k: int = 20) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    seen_chunks: set[int] = set()
    for mode in ("hybrid", "vector", "keyword"):
        results = hybrid_search(
            SessionLocal,
            query=question.query,
            query_vector=query_vector,
            filters=question_filters(question),
            top_k=pool_k,
            candidate_k=50,
            max_chunks_per_job=2,
            mode=mode,
        )
        for item in results:
            if item["chunk_id"] in seen_chunks:
                continue
            seen_chunks.add(item["chunk_id"])
            pool.append({**item, "retrieved_by": [mode]})
    return pool


def _job_ranking(results: list[dict[str, Any]]) -> list[str]:
    return unique_in_order([item["job_id"] for item in results])


def _rank_metrics(ranking: list[str], relevance: dict[str, int], k: int = 8) -> dict[str, float]:
    relevant = [job_id for job_id, grade in relevance.items() if grade > 0]
    return {
        "recall_at_5": recall_at_k(ranking, relevant, 5),
        "recall_at_8": recall_at_k(ranking, relevant, 8),
        "mrr_at_8": reciprocal_rank(ranking, relevant, 8),
        "graded_ndcg_at_8": graded_ndcg_at_k(ranking, relevance, k),
    }


def _average(rows: list[dict[str, float]]) -> dict[str, float]:
    names = ("recall_at_5", "recall_at_8", "mrr_at_8", "graded_ndcg_at_8")
    return {name: round(sum(row[name] for row in rows) / len(rows), 4) if rows else 0.0 for name in names}


def _macro_f1(labels: list[int], predictions: list[int]) -> float:
    values: list[float] = []
    for label in (0, 1, 2):
        tp = sum(actual == label and predicted == label for actual, predicted in zip(labels, predictions))
        fp = sum(actual != label and predicted == label for actual, predicted in zip(labels, predictions))
        fn = sum(actual == label and predicted != label for actual, predicted in zip(labels, predictions))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(values) / len(values)


def _predict_grade(score: float, thresholds: tuple[float, float]) -> int:
    low, high = thresholds
    return 0 if score < low else 1 if score < high else 2


def _calibrate_thresholds(samples: list[tuple[float, int]]) -> tuple[float, float] | None:
    if not samples or len({label for _, label in samples}) < 2:
        return None
    values = sorted({score for score, _ in samples})
    candidates = [values[0]] + [values[min(len(values) - 1, int(index * (len(values) - 1) / 12))] for index in range(1, 13)]
    best: tuple[float, float, float] | None = None
    for low in candidates:
        for high in candidates:
            if low >= high:
                continue
            predictions = [_predict_grade(score, (low, high)) for score, _ in samples]
            labels = [label for _, label in samples]
            score = _macro_f1(labels, predictions)
            if best is None or score > best[0]:
                best = (score, low, high)
    return (best[1], best[2]) if best else None


def _classification_metrics(samples: list[tuple[float, int]], thresholds: tuple[float, float] | None) -> dict[str, float | int | None]:
    if not samples or thresholds is None:
        return {"samples": len(samples), "accuracy": None, "macro_f1": None, "disagreements": None}
    labels = [label for _, label in samples]
    predictions = [_predict_grade(score, thresholds) for score, _ in samples]
    return {
        "samples": len(samples),
        "accuracy": round(sum(actual == predicted for actual, predicted in zip(labels, predictions)) / len(labels), 4),
        "macro_f1": round(_macro_f1(labels, predictions), 4),
        "disagreements": sum(actual != predicted for actual, predicted in zip(labels, predictions)),
    }


def _evaluate_split(
    questions,
    annotations: dict[str, dict[str, Any]],
    vectors: dict[str, list[float]],
    reranker,
    locked: RetrievalConfig,
    thresholds: tuple[float, float] | None,
    pool_k: int,
) -> dict[str, Any]:
    baseline_rows: list[dict[str, Any]] = []
    reranked_rows: list[dict[str, Any]] = []
    disagreements: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    for question in questions:
        relevance = annotation_relevance(annotations[question.id])
        query_vector = vectors[question.id]
        baseline_results = hybrid_search(
            SessionLocal,
            query=question.query,
            query_vector=query_vector,
            filters=question_filters(question),
            top_k=pool_k,
            candidate_k=locked.candidate_k,
            max_chunks_per_job=locked.max_chunks_per_job,
            mode=locked.mode,
            rrf_k=locked.rrf_k,
            vector_weight=locked.vector_weight,
            keyword_weight=locked.keyword_weight,
        )
        pool = _candidate_pool(question, query_vector, pool_k=pool_k)
        started = time.perf_counter()
        scores = reranker.score(question.query, [item["content"] for item in pool])
        rerank_ms = round((time.perf_counter() - started) * 1000, 2)
        for item, score in zip(pool, scores):
            item["reranker_score"] = round(float(score), 8)
        groups: dict[str, dict[str, Any]] = {}
        for item in pool:
            group = groups.setdefault(item["job_id"], {"job_id": item["job_id"], "score": -1.0, "chunk_id": item["chunk_id"], "title": item["title"]})
            if item["reranker_score"] > group["score"]:
                group.update(score=item["reranker_score"], chunk_id=item["chunk_id"])
        reranked_ids = [item["job_id"] for item in sorted(groups.values(), key=lambda value: value["score"], reverse=True)]
        baseline_ids = _job_ranking(baseline_results)[:pool_k]
        baseline_rows.append({"id": question.id, **_rank_metrics(baseline_ids, relevance)})
        reranked_rows.append({"id": question.id, "rerank_ms": rerank_ms, **_rank_metrics(reranked_ids, relevance)})
        relevant_jobs = [job_id for job_id, grade in relevance.items() if grade > 0]
        coverage_rows.append({"id": question.id, "relevant_jobs": relevant_jobs, "covered_by_pool": [job_id for job_id in relevant_jobs if job_id in groups]})
        if thresholds:
            for item in groups.values():
                if item["job_id"] not in relevance:
                    continue
                predicted = _predict_grade(item["score"], thresholds)
                actual = relevance[item["job_id"]]
                if predicted != actual:
                    disagreements.append({"id": question.id, "job_id": item["job_id"], "chunk_id": item["chunk_id"], "score": item["score"], "predicted": predicted, "actual": actual, "title": item["title"]})
    coverage = [len(row["covered_by_pool"]) / len(row["relevant_jobs"]) if row["relevant_jobs"] else 1.0 for row in coverage_rows]
    return {
        "baseline": {"summary": _average(baseline_rows), "rows": baseline_rows},
        "reranked": {"summary": _average(reranked_rows), "rows": reranked_rows},
        "candidate_pool_coverage": round(sum(coverage) / len(coverage), 4) if coverage else 0.0,
        "coverage_rows": coverage_rows,
        "disagreements": disagreements,
    }


def run_experiment(
    question_path: str | Path,
    annotation_path: str | Path,
    split_path: str | Path,
    locked_config_path: str | Path,
    *,
    pool_k: int = 20,
) -> dict[str, Any]:
    questions = {question.id: question for question in load_questions(question_path)}
    annotations = all_annotations(Path(annotation_path))
    splits = load_splits(split_path)
    locked = _load_locked_config(locked_config_path)
    valid_ids = [question_id for question_id in splits["development"] + splits["test"] if question_id in annotations and not annotations[question_id].get("should_refuse", False) and any(grade > 0 for grade in annotation_relevance(annotations[question_id]).values())]
    selected = [questions[question_id] for question_id in valid_ids]
    vectors_list = get_embedding_provider().embed([question.query for question in selected]) if selected else []
    vectors = {question.id: vector for question, vector in zip(selected, vectors_list)}
    reranker = get_reranker()
    started = time.perf_counter()
    pools: dict[str, list[dict[str, Any]]] = {}
    samples: list[tuple[float, int]] = []
    for question in selected:
        pool = _candidate_pool(question, vectors[question.id], pool_k=pool_k)
        scores = reranker.score(question.query, [item["content"] for item in pool])
        pools[question.id] = []
        relevance = annotation_relevance(annotations[question.id])
        for item, score in zip(pool, scores):
            item = {**item, "reranker_score": float(score)}
            pools[question.id].append(item)
        max_scores: dict[str, float] = {}
        for item in pools[question.id]:
            max_scores[item["job_id"]] = max(max_scores.get(item["job_id"], 0.0), item["reranker_score"])
        for job_id, score in max_scores.items():
            if job_id in relevance:
                samples.append((score, relevance[job_id]))
    thresholds = _calibrate_thresholds(samples)
    reranker_elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

    def split_result(ids: list[str]) -> dict[str, Any]:
        split_questions = [questions[question_id] for question_id in ids if question_id in pools]
        split_vectors = {question.id: vectors[question.id] for question in split_questions}
        # Reuse the pooled scores to avoid a second model pass.
        class CachedReranker:
            def score(self, query: str, passages: list[str]) -> list[float]:
                question = next(item for item in split_questions if item.query == query)
                return [item["reranker_score"] for item in pools[question.id]][: len(passages)]

        return _evaluate_split(split_questions, annotations, split_vectors, CachedReranker(), locked, thresholds, pool_k)

    development = split_result(splits["development"])
    test = split_result(splits["test"])
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "reranker_model": get_reranker().config.reranker_model,
        "reranker_device": str(get_reranker()._device) if get_reranker()._device else "auto",
        "pool_k": pool_k,
        "locked_config": {"id": locked.id, **locked.__dict__},
        "calibration": {**_classification_metrics(samples, thresholds), "thresholds": thresholds, "method": "development human judgments, job-level max chunk score, macro F1 over grades 0/1/2"},
        "runtime_ms": reranker_elapsed_ms,
        "development": development,
        "holdout": test,
        "disagreement_queue": (development["disagreements"] + test["disagreements"])[:100],
        "interpretation": "Reranker scores and silver labels require human audit before automatic labels are used in production.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a local reranker against JobRAG human judgments")
    parser.add_argument("--questions", default="data/eval/questions.jsonl")
    parser.add_argument("--annotations", default=str(DEFAULT_ANNOTATION_PATH))
    parser.add_argument("--splits", default="data/eval/splits.json")
    parser.add_argument("--locked-config", default="data/eval/locked-retrieval-config.json")
    parser.add_argument("--pool-k", type=int, default=10)
    parser.add_argument("--output", default="data/eval/reports/reranker-experiment.json")
    args = parser.parse_args()
    try:
        report = run_experiment(args.questions, args.annotations, args.splits, args.locked_config, pool_k=args.pool_k)
    except RerankerError as exc:
        raise SystemExit(str(exc)) from exc
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "calibration": report["calibration"], "development": report["development"]["reranked"]["summary"], "holdout": report["holdout"]["reranked"]["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
