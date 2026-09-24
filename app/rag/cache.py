from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.db.models import Job, JobChunk, RAGQueryCache
from app.rag.retrieval import RetrievalFilters


def corpus_version(session_factory: sessionmaker) -> str:
    """Return a cheap version token that invalidates cache after data changes."""
    with session_factory() as session:
        job_count = int(session.scalar(select(func.count()).select_from(Job)) or 0)
        active_job_count = int(
            session.scalar(select(func.count()).select_from(Job).where(Job.is_active.is_(True))) or 0
        )
        chunk_count = int(session.scalar(select(func.count()).select_from(JobChunk)) or 0)
        embedded_count = int(
            session.scalar(
                select(func.count())
                .select_from(JobChunk)
                .where(
                    JobChunk.embedding_provider == settings.embedding_provider,
                    JobChunk.embedding_model == settings.embedding_model_id,
                    JobChunk.embedding.is_not(None),
                )
            )
            or 0
        )
        active_content = session.execute(
            select(Job.id, Job.content_hash)
            .where(Job.is_active.is_(True))
            .order_by(Job.id)
        ).all()
        latest_chunk_hash = session.scalar(select(func.max(JobChunk.content_hash))) or ""
    corpus_content_hash = hashlib.sha256(
        "\n".join(f"{job_id}:{content_hash}" for job_id, content_hash in active_content).encode("utf-8")
    ).hexdigest()
    return "|".join(
        [
            str(job_count),
            str(active_job_count),
            str(chunk_count),
            str(embedded_count),
            corpus_content_hash,
            str(latest_chunk_hash),
            settings.embedding_model_id,
        ]
    )


def request_cache_key(
    *,
    kind: str,
    query: str,
    filters: RetrievalFilters,
    top_k: int,
    llm_model: str = "",
) -> str:
    payload = {
        "kind": kind,
        "query": query.strip().casefold(),
        "filters": asdict(filters),
        "top_k": top_k,
        "embedding_model": settings.embedding_model_id,
        "llm_model": llm_model,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def get_cached(
    session_factory: sessionmaker,
    *,
    cache_key: str,
    kind: str,
    version: str,
) -> dict[str, Any] | None:
    now = datetime.now(timezone.utc)
    with session_factory() as session:
        row = session.scalar(
            select(RAGQueryCache).where(
                RAGQueryCache.cache_key == cache_key,
                RAGQueryCache.kind == kind,
                RAGQueryCache.corpus_version == version,
            )
        )
        if row is None:
            return None
        expires_at = row.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if expires_at <= now:
            session.delete(row)
            session.commit()
            return None
        return dict(row.payload)


def put_cached(
    session_factory: sessionmaker,
    *,
    cache_key: str,
    kind: str,
    version: str,
    payload: dict[str, Any],
    ttl_seconds: int,
) -> None:
    now = datetime.now(timezone.utc)
    with session_factory() as session:
        existing = session.get(RAGQueryCache, cache_key)
        if existing is None:
            session.add(
                RAGQueryCache(
                    cache_key=cache_key,
                    kind=kind,
                    corpus_version=version,
                    payload=payload,
                    created_at=now,
                    expires_at=now + timedelta(seconds=ttl_seconds),
                )
            )
        else:
            existing.kind = kind
            existing.corpus_version = version
            existing.payload = payload
            existing.created_at = now
            existing.expires_at = now + timedelta(seconds=ttl_seconds)
        session.commit()
