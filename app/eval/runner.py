from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select

from app.db.models import Job
from app.db.session import SessionLocal
from app.eval.dataset import EvalQuestion, load_questions
from app.eval.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from app.rag.providers import get_embedding_provider
from app.rag.retrieval import RetrievalFilters, hybrid_search


def question_filters(question: EvalQuestion) -> RetrievalFilters:
    return RetrievalFilters(
        source=question.filters.get("source", ""),
        location=question.filters.get("location", ""),
        title=question.filters.get("title", ""),
        job_type=question.filters.get("job_type", ""),
    )


def silver_relevant_job_ids(question: EvalQuestion, limit: int = 10) -> list[str]:
    """Build provisional labels from expected terms; these require later human review."""
    if question.should_refuse or not question.expected_terms:
        return []
    filters = question_filters(question)
    with SessionLocal() as session:
        statement = select(Job)
        if filters.source:
            statement = statement.where(Job.source.ilike(f"%{filters.source}%"))
        if filters.location:
            statement = statement.where(Job.location.ilike(f"%{filters.location}%"))
        if filters.title:
            statement = statement.where(Job.title.ilike(f"%{filters.title}%"))
        if filters.job_type:
            statement = statement.where(Job.job_type.ilike(f"%{filters.job_type}%"))
        jobs = session.scalars(statement).all()
    terms = tuple(term.casefold() for term in question.expected_terms)
    scored: list[tuple[int, str]] = []
    for job in jobs:
        haystack = f"{job.title}\n{job.company or ''}\n{job.description or ''}".casefold()
        score = sum(haystack.count(term) for term in terms)
        if score:
            scored.append((score, job.id))
    return [job_id for _, job_id in sorted(scored, key=lambda item: (-item[0], item[1]))[:limit]]


def unique_in_order(values: list[str]) -> list[str]:
    """Collapse chunk-level results to a job-level ranking without reordering."""
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def run_benchmark(question_path: str | Path, top_k: int = 8) -> dict[str, Any]:
    questions = load_questions(question_path)
    embeddable = [question for question in questions if not question.should_refuse]
    vectors = get_embedding_provider().embed([question.query for question in embeddable]) if embeddable else []
    vector_by_id = {question.id: vector for question, vector in zip(embeddable, vectors)}
    rows: list[dict[str, Any]] = []
    for question in questions:
        gold_ids = list(question.relevant_job_ids) or silver_relevant_job_ids(question)
        if question.should_refuse:
            rows.append({"id": question.id, "category": question.category, "should_refuse": True, "gold_ids": [], "modes": {}})
            continue
        query_vector = vector_by_id[question.id]
        modes: dict[str, Any] = {}
        for mode in ("keyword", "vector", "hybrid"):
            results = hybrid_search(
                SessionLocal,
                query=question.query,
                query_vector=query_vector,
                filters=question_filters(question),
                top_k=top_k,
                mode=mode,
            )
            raw_result_ids = [item["job_id"] for item in results]
            result_ids = unique_in_order(raw_result_ids)
            modes[mode] = {
                "retrieved_ids": raw_result_ids,
                "retrieved_job_ids": result_ids,
                "recall_at_k": recall_at_k(result_ids, gold_ids, top_k),
                "mrr": reciprocal_rank(result_ids, gold_ids, top_k),
                "ndcg_at_k": ndcg_at_k(result_ids, gold_ids, top_k),
            }
        rows.append({
            "id": question.id,
            "category": question.category,
            "should_refuse": False,
            "label_source": "gold" if question.relevant_job_ids else "silver_expected_terms",
            "gold_ids": gold_ids,
            "modes": modes,
        })

    summary: dict[str, Any] = {}
    for mode in ("keyword", "vector", "hybrid"):
        values = [row["modes"][mode] for row in rows if row["modes"]]
        summary[mode] = {
            "questions": len(values),
            "recall_at_k": round(sum(value["recall_at_k"] for value in values) / len(values), 4) if values else 0.0,
            "mrr": round(sum(value["mrr"] for value in values) / len(values), 4) if values else 0.0,
            "ndcg_at_k": round(sum(value["ndcg_at_k"] for value in values) / len(values), 4) if values else 0.0,
        }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "question_file": str(question_path),
        "top_k": top_k,
        "label_policy": "gold IDs when present; otherwise provisional expected-term silver labels",
        "summary": summary,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the offline JobRAG retrieval benchmark")
    parser.add_argument("--questions", default="data/eval/questions.jsonl")
    parser.add_argument("--output", default="data/eval/reports/retrieval-baseline.json")
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()
    report = run_benchmark(args.questions, top_k=args.top_k)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
