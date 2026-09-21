from __future__ import annotations

import re
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from app.db.models import Job, JobChunk
from app.rag.retrieval import RetrievalFilters


# A small transparent vocabulary is preferable to asking an LLM to infer
# corpus-wide frequencies from the top-k retrieved chunks.  It can grow as the
# user adds a recurring skill to the job corpus.
SKILL_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Python", (r"\bpython\b",)),
    ("SQL", (r"\bsql\b", r"\bsqlite\b", r"postgres(?:ql)?")),
    ("RAG", (r"\brag\b", r"retrieval[- ]augmented generation")),
    ("LLM", (r"\bllm\b", r"large language model")),
    ("AI agents", (r"\bagentic\b", r"\bai agents?\b", r"\bllm agents?\b")),
    ("Machine learning", (r"machine learning", r"\bml\b")),
    ("Deep learning", (r"deep learning",)),
    ("Vector databases", (r"vector databases?", r"\bqdrant\b", r"\bmilvus\b", r"\bweaviate\b")),
    ("LangChain", (r"\blangchain\b",)),
    ("PyTorch", (r"\bpytorch\b",)),
    ("TensorFlow", (r"\btensorflow\b",)),
    ("Docker", (r"\bdocker\b",)),
    ("Kubernetes", (r"\bkubernetes\b", r"\bk8s\b")),
    ("JavaScript/TypeScript", (r"\bjavascript\b", r"\btypescript\b", r"\bnode(?:\.js)?\b")),
    ("React", (r"\breact(?:\.js)?\b",)),
)


def _skill_counts(
    job_texts: list[str],
    limit: int,
    job_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    total_jobs = len(job_texts)
    counts: list[dict[str, Any]] = []
    for label, patterns in SKILL_PATTERNS:
        compiled = [re.compile(pattern, re.I) for pattern in patterns]
        matching_indexes = [
            index
            for index, text in enumerate(job_texts)
            if any(pattern.search(text) for pattern in compiled)
        ]
        matches = len(matching_indexes)
        if matches:
            item = {
                "skill": label,
                "jobs": matches,
                "percentage": round(matches / total_jobs * 100, 1) if total_jobs else 0.0,
            }
            if job_ids is not None:
                item["job_ids"] = [job_ids[index] for index in matching_indexes]
            counts.append(item)
    counts.sort(key=lambda item: (-item["jobs"], item["skill"]))
    return counts[:limit]


def aggregate_job_evidence(
    session_factory: sessionmaker,
    *,
    filters: RetrievalFilters,
    limit: int = 20,
) -> dict[str, Any]:
    with session_factory() as session:
        statement = select(Job).where(Job.is_active.is_(True))
        if filters.source:
            statement = statement.where(Job.source == filters.source)
        if filters.location:
            statement = statement.where(Job.location.ilike(f"%{filters.location}%"))
        if filters.title:
            statement = statement.where(Job.title.ilike(f"%{filters.title}%"))
        if filters.job_type:
            statement = statement.where(Job.job_type.ilike(f"%{filters.job_type}%"))
        if filters.date_after:
            statement = statement.where(Job.date_posted >= filters.date_after)
        jobs = session.scalars(statement).all()

    job_texts = [f"{job.title}\n{job.description or ''}" for job in jobs]
    return {
        "total_jobs": len(job_texts),
        "skill_counts": _skill_counts(job_texts, limit),
        "filters": {
            "source": filters.source,
            "location": filters.location,
            "title": filters.title,
            "job_type": filters.job_type,
            "date_after": filters.date_after.isoformat() if filters.date_after else None,
        },
    }


def aggregate_ranked_job_evidence(
    session_factory: sessionmaker,
    *,
    job_ids: list[str],
    limit: int = 20,
) -> dict[str, Any]:
    """Count each skill once per selected relevant job using filtered chunks."""
    ordered_ids = list(dict.fromkeys(job_ids))
    if not ordered_ids:
        return {"total_jobs": 0, "skill_counts": [], "job_ids": []}
    with session_factory() as session:
        rows = session.execute(
            select(JobChunk.job_id, JobChunk.content)
            .join(Job, Job.id == JobChunk.job_id)
            .where(JobChunk.job_id.in_(ordered_ids))
            .where(Job.is_active.is_(True))
            .order_by(JobChunk.job_id, JobChunk.chunk_index)
        ).all()
    by_job: dict[str, list[str]] = {job_id: [] for job_id in ordered_ids}
    for job_id, content in rows:
        by_job.setdefault(job_id, []).append(content)
    available_ids = [job_id for job_id in ordered_ids if by_job.get(job_id)]
    job_texts = ["\n".join(by_job[job_id]) for job_id in available_ids]
    return {
        "total_jobs": len(available_ids),
        "skill_counts": _skill_counts(job_texts, limit, available_ids),
        "job_ids": available_ids,
        "denominator": "top_relevant_jobs",
    }
