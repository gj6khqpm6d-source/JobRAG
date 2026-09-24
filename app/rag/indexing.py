from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from sqlalchemy import delete, func, or_, select
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.db.models import Job, JobChunk
from app.rag.chunking import build_job_chunks
from app.rag.providers import EmbeddingProvider
from app.rag.retrieval import rebuild_sqlite_keyword_index

MAX_AUTOMATIC_EMBEDDING_ATTEMPTS = 3


@dataclass
class ChunkingResult:
    jobs_processed: int = 0
    chunks_created: int = 0
    chunks_removed: int = 0
    jobs_without_description: int = 0

    def to_dict(self):
        return asdict(self)


@dataclass
class IndexingResult:
    chunks_indexed: int = 0
    batches: int = 0
    chunks_failed: int = 0
    error_summary: str | None = None

    def to_dict(self):
        return asdict(self)


def prepare_chunks(session_factory: sessionmaker, job_ids: list[str] | None = None) -> ChunkingResult:
    result = ChunkingResult()
    with session_factory() as session:
        statement = select(Job).where(Job.is_active.is_(True))
        if job_ids is not None:
            if not job_ids:
                return result
            statement = statement.where(Job.id.in_(job_ids))
        jobs = session.scalars(statement).all()
        for job in jobs:
            chunks = build_job_chunks(
                title=job.title,
                company=job.company,
                location=job.location,
                description=job.description,
                job_type=job.job_type,
                is_remote=job.is_remote,
            )
            if not chunks:
                deletion = session.execute(delete(JobChunk).where(JobChunk.job_id == job.id))
                result.chunks_removed += int(deletion.rowcount or 0)
                result.jobs_without_description += 1
                continue
            current_hashes = set(session.scalars(select(JobChunk.content_hash).where(JobChunk.job_id == job.id)).all())
            new_hashes = {chunk.content_hash for chunk in chunks}
            if current_hashes == new_hashes:
                continue
            deletion = session.execute(delete(JobChunk).where(JobChunk.job_id == job.id))
            result.chunks_removed += int(deletion.rowcount or 0)
            for index, chunk in enumerate(chunks):
                session.add(
                    JobChunk(
                        job_id=job.id,
                        section_type=chunk.section_type,
                        chunk_index=index,
                        content=chunk.content,
                        content_hash=chunk.content_hash,
                        token_count=chunk.token_count,
                    )
                )
                result.chunks_created += 1
            result.jobs_processed += 1
        # Keep the local SQLite lexical index in sync with the chunk content.
        # PostgreSQL uses its native GIN index and simply returns False here.
        rebuild_sqlite_keyword_index(session)
        session.commit()
    return result


def index_pending_chunks(
    session_factory: sessionmaker,
    provider: EmbeddingProvider,
    *,
    limit: int = 500,
    batch_size: int = 32,
    job_ids: list[str] | None = None,
    retry_failed: bool = False,
) -> IndexingResult:
    result = IndexingResult()
    if job_ids is not None and not job_ids:
        return result
    pending = or_(
        JobChunk.embedding.is_(None),
        JobChunk.embedding_provider.is_(None),
        JobChunk.embedding_model.is_(None),
        JobChunk.embedding_provider != settings.embedding_provider,
        JobChunk.embedding_model != settings.embedding_model_id,
    )
    with session_factory() as session:
        chunk_ids = session.scalars(
        select(JobChunk.id)
        .join(Job, Job.id == JobChunk.job_id)
        .where(
            pending,
            Job.is_active.is_(True),
            *(
                [JobChunk.embedding_attempts > 0]
                if retry_failed
                else [JobChunk.embedding_attempts < MAX_AUTOMATIC_EMBEDDING_ATTEMPTS]
            ),
            *( [JobChunk.job_id.in_(job_ids)] if job_ids is not None else [] ),
            )
            .order_by(JobChunk.id)
            .limit(limit)
        ).all()
    for start in range(0, len(chunk_ids), batch_size):
        ids = chunk_ids[start : start + batch_size]
        with session_factory() as session:
            chunks = session.scalars(select(JobChunk).where(JobChunk.id.in_(ids)).order_by(JobChunk.id)).all()
            if retry_failed:
                # An explicit operator retry starts a fresh automatic retry window.
                for chunk in chunks:
                    chunk.embedding_attempts = 0
                    chunk.embedding_failed_at = None
                    chunk.embedding_error = None
                session.commit()
            try:
                result.batches += 1
                vectors = provider.embed([chunk.content for chunk in chunks])
                now = datetime.now(timezone.utc)
                for chunk, vector in zip(chunks, vectors, strict=True):
                    chunk.embedding = vector
                    chunk.embedding_provider = settings.embedding_provider
                    chunk.embedding_model = settings.embedding_model_id
                    chunk.embedded_at = now
                    chunk.embedding_attempts = 0
                    chunk.embedding_failed_at = None
                    chunk.embedding_error = None
                session.commit()
                result.chunks_indexed += len(chunks)
            except Exception as exc:
                session.rollback()
                failure_time = datetime.now(timezone.utc)
                error = f"{type(exc).__name__}: {exc}"[:1000]
                failed_chunks = session.scalars(select(JobChunk).where(JobChunk.id.in_(ids))).all()
                for chunk in failed_chunks:
                    chunk.embedding_attempts += 1
                    chunk.embedding_failed_at = failure_time
                    chunk.embedding_error = error
                session.commit()
                result.chunks_failed += len(failed_chunks)
                result.error_summary = error
                # Stop after the first failing batch. A provider outage should not
                # trigger repeated requests for every remaining batch in this run.
                break
    return result


def index_stats(session_factory: sessionmaker) -> dict[str, int | str]:
    pending = or_(
        JobChunk.embedding.is_(None),
        JobChunk.embedding_provider.is_(None),
        JobChunk.embedding_model.is_(None),
        JobChunk.embedding_provider != settings.embedding_provider,
        JobChunk.embedding_model != settings.embedding_model_id,
    )
    with session_factory() as session:
        total = session.scalar(
            select(func.count()).select_from(JobChunk).join(Job).where(Job.is_active.is_(True))
        ) or 0
        indexed = session.scalar(
            select(func.count()).select_from(JobChunk).join(Job).where(
                Job.is_active.is_(True),
                JobChunk.embedding.is_not(None),
                JobChunk.embedding_provider == settings.embedding_provider,
                JobChunk.embedding_model == settings.embedding_model_id,
            )
        ) or 0
        oldest_pending = session.scalar(
            select(func.min(JobChunk.created_at))
            .join(Job)
            .where(Job.is_active.is_(True), pending)
        )
        failed = session.scalar(
            select(func.count()).select_from(JobChunk).join(Job).where(
                Job.is_active.is_(True), pending, JobChunk.embedding_attempts > 0
            )
        ) or 0
        exhausted = session.scalar(
            select(func.count()).select_from(JobChunk).join(Job).where(
                Job.is_active.is_(True), pending,
                JobChunk.embedding_attempts >= MAX_AUTOMATIC_EMBEDDING_ATTEMPTS,
            )
        ) or 0
    if oldest_pending is not None:
        if oldest_pending.tzinfo is None:
            oldest_pending = oldest_pending.replace(tzinfo=timezone.utc)
        oldest_pending_hours = max(0, int((datetime.now(timezone.utc) - oldest_pending).total_seconds() // 3600))
    else:
        oldest_pending_hours = 0
    return {
        "total_chunks": total,
        "indexed_chunks": indexed,
        "pending_chunks": total - indexed,
        "failed_chunks": failed,
        "exhausted_chunks": exhausted,
        "oldest_pending_hours": oldest_pending_hours,
        "provider": settings.embedding_provider,
        "model": settings.embedding_model_id,
    }
