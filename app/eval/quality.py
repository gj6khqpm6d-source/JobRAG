from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import sessionmaker

from app.db.models import Job, JobChunk
from app.db.session import SessionLocal
from app.eval.annotations import DEFAULT_ANNOTATION_PATH, all_annotations
from app.eval.dataset import EvalQuestion, load_questions
from app.eval.metrics import condensed_graded_ndcg_at_k, recall_at_k, reciprocal_rank
from app.eval.runner import question_filters
from app.eval.tuning import load_splits
from app.rag.cache import corpus_version
from app.rag.lifecycle import DEFAULT_RETENTION_DAYS
from app.rag.providers import EmbeddingProvider, LLMProvider, get_embedding_provider
from app.rag.retrieval import ranked_job_search, scope_query
from app.rag.service import answer_question


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUESTIONS_PATH = PROJECT_ROOT / "data" / "eval" / "questions.jsonl"
DEFAULT_GATES_PATH = PROJECT_ROOT / "data" / "eval" / "quality-gates.json"
DEFAULT_SPLITS_PATH = PROJECT_ROOT / "data" / "eval" / "splits.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "data" / "eval" / "reports" / "quality-latest.json"


@dataclass(frozen=True)
class QualityGates:
    version: int = 1
    minimum_questions: int = 25
    minimum_annotated_answerable_questions: int = 8
    minimum_recall_at_8: float = 0.7
    minimum_graded_ndcg_at_8: float = 0.65
    minimum_filter_accuracy: float = 1.0
    minimum_analysis_invariant_rate: float = 1.0
    minimum_citation_validity_rate: float = 1.0
    minimum_refusal_accuracy: float = 1.0
    minimum_index_coverage: float = 1.0
    maximum_active_stale_jobs: int = 0
    maximum_inactive_chunks: int = 0


class DeterministicEvaluationLLM(LLMProvider):
    """Exercise the answer pipeline without paying for an external judge."""

    def generate(self, *, system: str, user: str) -> str:
        if "[统计]" in user and "最相关岗位上下文" not in user:
            return "## 分析范围\n仅总结提供的知识库统计。[统计]"
        return (
            "## 分析范围与样本数量\n本次回答仅依据检索到的岗位样本。[1]\n\n"
            "## 核心技能和硬性要求\n具体要求以岗位证据为准。[1]\n\n"
            "## 常见但非必需的技能\n未超出证据范围作额外推断。[1]\n\n"
            "## 加分项\n请查看对应岗位原文。[1]\n\n"
            "## 职责和技术栈归纳\n内容来自当前岗位证据。[1]\n\n"
            "## 结论限制与不确定性\n该结果不代表整个招聘市场。[1]"
        )


def load_quality_gates(path: str | Path = DEFAULT_GATES_PATH) -> QualityGates:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return QualityGates(**payload)


def _relevance(annotation: dict[str, Any]) -> dict[str, int]:
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


def _average(rows: list[dict[str, Any]], key: str) -> float:
    return round(sum(float(row[key]) for row in rows) / len(rows), 4) if rows else 0.0


def _retrieval_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "evaluated_questions": len(rows),
        "recall_at_8": _average(rows, "recall_at_8"),
        "mrr_at_8": _average(rows, "mrr_at_8"),
        "graded_ndcg_at_8": _average(rows, "graded_ndcg_at_8"),
        "filter_accuracy": _average(rows, "filter_accuracy"),
    }


def _gate(name: str, actual: Any, expected: str, passed: bool) -> dict[str, Any]:
    return {"name": name, "actual": actual, "expected": expected, "passed": bool(passed)}


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix="quality-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def load_latest_quality_report(path: str | Path = DEFAULT_REPORT_PATH) -> dict[str, Any] | None:
    report_path = Path(path)
    if not report_path.exists():
        return None
    return json.loads(report_path.read_text(encoding="utf-8"))


def run_quality_evaluation(
    *,
    session_factory: sessionmaker = SessionLocal,
    embedding_provider: EmbeddingProvider | None = None,
    question_path: str | Path = DEFAULT_QUESTIONS_PATH,
    annotation_path: str | Path = DEFAULT_ANNOTATION_PATH,
    gates_path: str | Path = DEFAULT_GATES_PATH,
    split_path: str | Path | None = DEFAULT_SPLITS_PATH,
    report_path: str | Path | None = DEFAULT_REPORT_PATH,
    top_k: int = 8,
) -> dict[str, Any]:
    started = time.perf_counter()
    gates_config = load_quality_gates(gates_path)
    questions = load_questions(question_path)
    annotations = all_annotations(Path(annotation_path))
    provider = embedding_provider or get_embedding_provider()
    split_by_id: dict[str, str] = {}
    if split_path is not None and Path(split_path).exists():
        for split_name, question_ids in load_splits(split_path).items():
            split_by_id.update({question_id: split_name for question_id in question_ids})

    with session_factory() as session:
        jobs = session.scalars(select(Job)).all()
        known_job_ids = {job.id for job in jobs}
        active_job_ids = {job.id for job in jobs if job.is_active}
        known_chunk_ids = set(session.scalars(select(JobChunk.id)).all())

    orphaned_judgments = 0
    inactive_positive_labels = 0
    missing_evidence_chunks = 0
    annotated_answerable = 0
    for annotation in annotations.values():
        relevance = _relevance(annotation)
        orphaned_judgments += sum(job_id not in known_job_ids for job_id in relevance)
        inactive_positive_labels += sum(
            grade > 0 and job_id in known_job_ids and job_id not in active_job_ids
            for job_id, grade in relevance.items()
        )
        missing_evidence_chunks += sum(
            int(chunk_id) not in known_chunk_ids for chunk_id in annotation.get("evidence_chunk_ids", [])
        )
        if not annotation.get("should_refuse", False) and any(
            grade > 0 and job_id in active_job_ids for job_id, grade in relevance.items()
        ):
            annotated_answerable += 1

    dataset_layer = {
        "questions": len(questions),
        "annotations": len(annotations),
        "annotated_answerable_questions": annotated_answerable,
        "orphaned_job_judgments": orphaned_judgments,
        "inactive_positive_labels": inactive_positive_labels,
        "missing_evidence_chunks": missing_evidence_chunks,
    }

    answerable_for_retrieval: list[tuple[EvalQuestion, dict[str, int]]] = []
    for question in questions:
        annotation = annotations.get(question.id)
        if not annotation or annotation.get("should_refuse", False):
            continue
        relevance = _relevance(annotation)
        active_relevance = {
            job_id: grade for job_id, grade in relevance.items() if job_id in active_job_ids
        }
        if any(grade > 0 for grade in active_relevance.values()):
            answerable_for_retrieval.append((question, active_relevance))

    vectors = provider.embed(
        [scope_query(question.query, question_filters(question)) for question, _ in answerable_for_retrieval]
    ) if answerable_for_retrieval else []
    retrieval_rows: list[dict[str, Any]] = []
    for (question, relevance), vector in zip(answerable_for_retrieval, vectors):
        result = ranked_job_search(
            session_factory,
            query=question.query,
            query_vector=vector,
            filters=question_filters(question),
            top_jobs=top_k,
            candidate_jobs=50,
        )
        ranked_ids = [job["job_id"] for job in result["jobs"]]
        relevant_ids = [job_id for job_id, grade in relevance.items() if grade > 0]
        retrieval_rows.append(
            {
                "id": question.id,
                "split": split_by_id.get(question.id, "evaluation"),
                "retrieved_job_ids": ranked_ids,
                "recall_at_8": recall_at_k(ranked_ids, relevant_ids, top_k),
                "mrr_at_8": reciprocal_rank(ranked_ids, relevant_ids, top_k),
                "graded_ndcg_at_8": condensed_graded_ndcg_at_k(ranked_ids, relevance, top_k),
                "filter_accuracy": (
                    sum(_filter_match(job, question) for job in result["jobs"]) / len(result["jobs"])
                    if result["jobs"] else 1.0
                ),
            }
        )
    overall_summary = _retrieval_summary(retrieval_rows)
    development_summary = _retrieval_summary(
        [row for row in retrieval_rows if row["split"] == "development"]
    )
    holdout_summary = _retrieval_summary(
        [row for row in retrieval_rows if row["split"] == "test"]
    )
    gate_summary = holdout_summary if holdout_summary["evaluated_questions"] else overall_summary
    retrieval_layer = {
        **gate_summary,
        "gate_split": "test" if holdout_summary["evaluated_questions"] else "all",
        "metric_policy": "Recall@8 plus positive-judgment condensed graded NDCG@8",
        "overall": overall_summary,
        "development": development_summary,
        "test": holdout_summary,
        "rows": retrieval_rows,
    }

    end_to_end_rows: list[dict[str, Any]] = []
    evaluation_llm = DeterministicEvaluationLLM()
    for question in questions:
        annotation = annotations.get(question.id, {})
        expected_refusal = bool(annotation.get("should_refuse", question.should_refuse))
        result = answer_question(
            session_factory,
            question=question.query,
            embedding_provider=provider,
            llm_provider=evaluation_llm,
            filters=question_filters(question),
            top_k=top_k,
            cache_enabled=False,
        )
        actual_refusal = result.get("intent") == "refusal" or not result.get("sources") and "证据不足" in result.get("answer", "")
        sources = result.get("sources", [])
        source_ids = [source.get("job_id") for source in sources]
        citation = result.get("answer_validation") or {}
        citation_valid = actual_refusal or citation.get("status") == "passed"
        unique_sources = len(source_ids) == len(set(source_ids))
        filters_valid = all(_filter_match(source, question) for source in sources)
        statistics = result.get("statistics") or {}
        statistics_valid = True
        if statistics.get("denominator") == "top_relevant_jobs":
            total = int(statistics.get("total_jobs", 0))
            statistics_valid = total == len(sources) and all(
                0 <= int(item.get("jobs", -1)) <= total
                and float(item.get("percentage", -1)) == round(int(item.get("jobs", 0)) / total * 100, 1)
                for item in statistics.get("skill_counts", [])
            ) if total else not statistics.get("skill_counts")
        analysis_valid = unique_sources and filters_valid and statistics_valid
        refusal_correct = actual_refusal == expected_refusal
        end_to_end_rows.append(
            {
                "id": question.id,
                "expected_refusal": expected_refusal,
                "actual_refusal": actual_refusal,
                "refusal_correct": refusal_correct,
                "citation_valid": citation_valid,
                "analysis_invariants_valid": analysis_valid,
                "source_count": len(sources),
                "intent": result.get("intent"),
            }
        )

    non_refusal_rows = [row for row in end_to_end_rows if not row["expected_refusal"]]
    refusal_rows = [row for row in end_to_end_rows if row["expected_refusal"]]
    analysis_layer = {
        "evaluated_questions": len(non_refusal_rows),
        "invariant_pass_rate": _average(non_refusal_rows, "analysis_invariants_valid"),
    }
    answer_layer = {
        "evaluated_questions": len(end_to_end_rows),
        "citation_validity_rate": _average(non_refusal_rows, "citation_valid"),
        "refusal_cases": len(refusal_rows),
        "refusal_accuracy": _average(refusal_rows, "refusal_correct"),
        "rows": end_to_end_rows,
        "generation_policy": "deterministic_no_external_llm",
    }

    now = datetime.now(timezone.utc)
    cutoff_date = now.date() - timedelta(days=DEFAULT_RETENTION_DAYS)
    cutoff_datetime = datetime.combine(cutoff_date, datetime.min.time(), tzinfo=timezone.utc)
    with session_factory() as session:
        active_jobs = int(session.scalar(select(func.count()).select_from(Job).where(Job.is_active.is_(True))) or 0)
        active_stale_jobs = int(
            session.scalar(
                select(func.count()).select_from(Job).where(
                    Job.is_active.is_(True),
                    or_(
                        and_(Job.date_posted.is_not(None), Job.date_posted < cutoff_date),
                        and_(Job.date_posted.is_(None), Job.last_seen_at < cutoff_datetime),
                    ),
                )
            ) or 0
        )
        active_chunks = int(
            session.scalar(
                select(func.count()).select_from(JobChunk).join(Job).where(Job.is_active.is_(True))
            ) or 0
        )
        indexed_chunks = int(
            session.scalar(
                select(func.count()).select_from(JobChunk).join(Job).where(
                    Job.is_active.is_(True),
                    JobChunk.embedding.is_not(None),
                )
            ) or 0
        )
        inactive_chunks = int(
            session.scalar(
                select(func.count()).select_from(JobChunk).join(Job).where(Job.is_active.is_(False))
            ) or 0
        )
    index_coverage = round(indexed_chunks / active_chunks, 4) if active_chunks else 1.0
    operations_layer = {
        "active_jobs": active_jobs,
        "active_stale_jobs": active_stale_jobs,
        "active_chunks": active_chunks,
        "indexed_chunks": indexed_chunks,
        "index_coverage": index_coverage,
        "inactive_chunks": inactive_chunks,
    }

    gates = [
        _gate("评估问题数量", len(questions), f">={gates_config.minimum_questions}", len(questions) >= gates_config.minimum_questions),
        _gate("人工校准题数量", annotated_answerable, f">={gates_config.minimum_annotated_answerable_questions}", annotated_answerable >= gates_config.minimum_annotated_answerable_questions),
        _gate("Recall@8", retrieval_layer["recall_at_8"], f">={gates_config.minimum_recall_at_8}", retrieval_layer["recall_at_8"] >= gates_config.minimum_recall_at_8),
        _gate("NDCG@8", retrieval_layer["graded_ndcg_at_8"], f">={gates_config.minimum_graded_ndcg_at_8}", retrieval_layer["graded_ndcg_at_8"] >= gates_config.minimum_graded_ndcg_at_8),
        _gate("元数据过滤准确率", retrieval_layer["filter_accuracy"], f">={gates_config.minimum_filter_accuracy}", retrieval_layer["filter_accuracy"] >= gates_config.minimum_filter_accuracy),
        _gate("分析不变量通过率", analysis_layer["invariant_pass_rate"], f">={gates_config.minimum_analysis_invariant_rate}", analysis_layer["invariant_pass_rate"] >= gates_config.minimum_analysis_invariant_rate),
        _gate("引用有效率", answer_layer["citation_validity_rate"], f">={gates_config.minimum_citation_validity_rate}", answer_layer["citation_validity_rate"] >= gates_config.minimum_citation_validity_rate),
        _gate("拒答准确率", answer_layer["refusal_accuracy"], f">={gates_config.minimum_refusal_accuracy}", answer_layer["refusal_accuracy"] >= gates_config.minimum_refusal_accuracy),
        _gate("向量索引覆盖率", index_coverage, f">={gates_config.minimum_index_coverage}", index_coverage >= gates_config.minimum_index_coverage),
        _gate("有效但过期岗位", active_stale_jobs, f"<={gates_config.maximum_active_stale_jobs}", active_stale_jobs <= gates_config.maximum_active_stale_jobs),
        _gate("失效岗位残留片段", inactive_chunks, f"<={gates_config.maximum_inactive_chunks}", inactive_chunks <= gates_config.maximum_inactive_chunks),
    ]
    report = {
        "report_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(time.perf_counter() - started, 2),
        "corpus_version": corpus_version(session_factory),
        "top_k": top_k,
        "thresholds": asdict(gates_config),
        "layers": {
            "dataset": dataset_layer,
            "retrieval": retrieval_layer,
            "analysis": analysis_layer,
            "answer": answer_layer,
            "operations": operations_layer,
        },
        "gates": gates,
        "passed": all(gate["passed"] for gate in gates),
    }
    if report_path is not None:
        _write_report(Path(report_path), report)
    return report
