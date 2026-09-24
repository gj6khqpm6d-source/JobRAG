from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, Boolean, Date, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.config import settings


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: uuid.uuid4().hex)
    fingerprint: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    source_job_id: Mapped[str | None] = mapped_column(String(255), index=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False, index=True)
    normalized_title: Mapped[str] = mapped_column(String(500), nullable=False, index=True)
    company: Mapped[str | None] = mapped_column(String(500), index=True)
    location: Mapped[str | None] = mapped_column(String(500), index=True)
    job_type: Mapped[str | None] = mapped_column(String(100), index=True)
    min_amount: Mapped[float | None] = mapped_column(Float)
    max_amount: Mapped[float | None] = mapped_column(Float)
    currency: Mapped[str | None] = mapped_column(String(16))
    interval: Mapped[str | None] = mapped_column(String(32))
    is_remote: Mapped[bool | None] = mapped_column(Boolean, index=True)
    description: Mapped[str | None] = mapped_column(Text)
    job_url: Mapped[str] = mapped_column(Text, nullable=False)
    job_url_direct: Mapped[str | None] = mapped_column(Text)
    date_posted: Mapped[date | None] = mapped_column(Date, index=True)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    raw_data: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False, index=True)

    snapshots: Mapped[list["JobSnapshot"]] = relationship(back_populates="job", cascade="all, delete-orphan")
    chunks: Mapped[list["JobChunk"]] = relationship(back_populates="job", cascade="all, delete-orphan")

    __table_args__ = (Index("ix_jobs_source_date", "source", "date_posted"),)


class JobSnapshot(Base):
    __tablename__ = "job_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    raw_data: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    job: Mapped[Job] = relationship(back_populates="snapshots")

    __table_args__ = (UniqueConstraint("job_id", "content_hash", name="uq_snapshot_job_content"),)


class JobChunk(Base):
    __tablename__ = "job_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    section_type: Mapped[str] = mapped_column(String(64), default="general", nullable=False, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    token_count: Mapped[int | None] = mapped_column(Integer)
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(settings.embedding_dimensions).with_variant(JSON(none_as_null=True), "sqlite")
    )
    embedding_provider: Mapped[str | None] = mapped_column(String(64))
    embedding_model: Mapped[str | None] = mapped_column(String(255))
    embedded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    embedding_attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    embedding_failed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    embedding_error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    job: Mapped[Job] = relationship(back_populates="chunks")

    __table_args__ = (UniqueConstraint("job_id", "content_hash", name="uq_chunk_job_content"),)


class RAGQueryCache(Base):
    """Small persistent cache for retrieval and answer responses.

    The cache is keyed by a request fingerprint and guarded by a corpus
    version, so new scrapes, chunk changes, or newly generated embeddings do
    not serve stale results.
    """

    __tablename__ = "rag_query_cache"

    cache_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    kind: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    corpus_version: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)


class RAGRequestLog(Base):
    """Privacy-safe operational trace for one knowledge-base question."""

    __tablename__ = "rag_request_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(32), unique=True, nullable=False, index=True)
    query_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    query_length: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False, index=True)
    intent: Mapped[str | None] = mapped_column(String(40), index=True)
    error_type: Mapped[str | None] = mapped_column(String(80), index=True)
    total_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    retrieval_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    llm_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    cache_hit: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
    retrieval_cache_hit: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    candidate_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    analyzed_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    source_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    refused: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
    no_result: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    estimated_cost_usd: Mapped[float | None] = mapped_column(Float)
    llm_model: Mapped[str | None] = mapped_column(String(255))
    embedding_model: Mapped[str | None] = mapped_column(String(255))
    pipeline_version: Mapped[str | None] = mapped_column(String(80), index=True)
    corpus_version: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)


class AutoScrapeSchedule(Base):
    """Persistent settings for the single local daily scrape schedule."""

    __tablename__ = "auto_scrape_schedule"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    sites: Mapped[list[str]] = mapped_column(JSON, default=lambda: ["linkedin"], nullable=False)
    search_term: Mapped[str] = mapped_column(String(255), default="AI Agent", nullable=False)
    location: Mapped[str] = mapped_column(String(255), default="Singapore", nullable=False)
    job_type: Mapped[str] = mapped_column(String(32), default="internship", nullable=False)
    results_per_site: Mapped[int] = mapped_column(Integer, default=10, nullable=False)
    lookback_hours: Mapped[int] = mapped_column(Integer, default=72, nullable=False)
    max_index_chunks: Mapped[int] = mapped_column(Integer, default=100, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class AutoScrapeRun(Base):
    """Durable status and bounded ingestion metrics for scheduled crawls."""

    __tablename__ = "auto_scrape_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    scheduled_for: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(24), nullable=False, index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    settings_snapshot: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    results_found: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    inserted: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    updated: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    unchanged: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    chunks_indexed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    chunks_pending: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_summary: Mapped[str | None] = mapped_column(Text)
