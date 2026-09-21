from __future__ import annotations

import hashlib
import math
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.db.models import Job, JobChunk, RAGRequestLog
from app.db.repository import knowledge_stats
from app.rag.indexing import index_stats


def _utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 2)
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower), 2)


def estimate_cost_usd(prompt_tokens: int, completion_tokens: int) -> float | None:
    input_rate = settings.deepseek_input_cost_per_million
    output_rate = settings.deepseek_output_cost_per_million
    if input_rate <= 0 and output_rate <= 0:
        return None
    return round((prompt_tokens * input_rate + completion_tokens * output_rate) / 1_000_000, 8)


def record_rag_request(
    session_factory: sessionmaker,
    *,
    query: str,
    status: str,
    total_ms: float,
    result: dict[str, Any] | None = None,
    error_type: str | None = None,
) -> str:
    """Persist a privacy-safe trace without raw question or prompt text."""
    result = result or {}
    telemetry = result.get("telemetry") or {}
    usage = telemetry.get("usage") or {}
    scope = result.get("analysis_scope") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    request_id = uuid.uuid4().hex
    with session_factory() as session:
        session.add(
            RAGRequestLog(
                request_id=request_id,
                query_hash=hashlib.sha256(query.strip().casefold().encode("utf-8")).hexdigest(),
                query_length=len(query),
                status=status,
                intent=result.get("intent"),
                error_type=error_type,
                total_ms=round(total_ms, 2),
                retrieval_ms=float(telemetry.get("retrieval_ms") or 0),
                llm_ms=float(telemetry.get("llm_ms") or 0),
                cache_hit=bool(result.get("cache_hit")),
                retrieval_cache_hit=bool(result.get("retrieval_cache_hit")),
                candidate_count=int(scope.get("candidate_job_count") or 0),
                analyzed_count=int(scope.get("analyzed_job_count") or 0),
                source_count=len(result.get("sources") or []),
                refused=result.get("intent") == "refusal",
                no_result=(
                    not bool(result.get("sources"))
                    and not bool((result.get("statistics") or {}).get("total_jobs"))
                    and result.get("intent") != "refusal"
                ),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=int(usage.get("total_tokens") or prompt_tokens + completion_tokens),
                estimated_cost_usd=estimate_cost_usd(prompt_tokens, completion_tokens),
                llm_model=telemetry.get("llm_model") or settings.deepseek_model,
                embedding_model=settings.embedding_model_id,
                pipeline_version=telemetry.get("pipeline_version"),
                corpus_version=result.get("corpus_version"),
            )
        )
        session.commit()
    return request_id


def prune_request_logs(session_factory: sessionmaker, retention_days: int | None = None) -> int:
    days = retention_days if retention_days is not None else settings.ops_log_retention_days
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, days))
    with session_factory() as session:
        result = session.execute(delete(RAGRequestLog).where(RAGRequestLog.created_at < cutoff))
        session.commit()
        return int(result.rowcount or 0)


def operations_summary(session_factory: sessionmaker, *, days: int = 7) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)
    with session_factory() as session:
        logs = list(
            session.scalars(
                select(RAGRequestLog)
                .where(RAGRequestLog.created_at >= cutoff)
                .order_by(RAGRequestLog.created_at.desc())
            ).all()
        )
        kb = knowledge_stats(session)
        latest_seen = _utc(session.scalar(select(func.max(Job.last_seen_at)).where(Job.is_active.is_(True))))
        latest_posted = session.scalar(select(func.max(Job.date_posted)).where(Job.is_active.is_(True)))
        latest_embedded = _utc(session.scalar(select(func.max(JobChunk.embedded_at))) )

    request_count = len(logs)
    successful = [row for row in logs if row.status == "success"]
    llm_calls = [row for row in logs if row.prompt_tokens or row.completion_tokens or row.llm_ms > 0]
    priced_costs = [row.estimated_cost_usd for row in logs if row.estimated_cost_usd is not None]
    recent = [
        {
            "request_id": row.request_id,
            "created_at": _utc(row.created_at).isoformat() if _utc(row.created_at) else None,
            "status": row.status,
            "intent": row.intent,
            "total_ms": row.total_ms,
            "cache_hit": row.cache_hit,
            "source_count": row.source_count,
            "total_tokens": row.total_tokens,
            "error_type": row.error_type,
        }
        for row in logs[:20]
    ]
    return {
        "window_days": days,
        "generated_at": now.isoformat(),
        "requests": {
            "total": request_count,
            "successful": len(successful),
            "errors": sum(row.status == "error" for row in logs),
            "error_rate": round(sum(row.status == "error" for row in logs) / request_count, 4) if request_count else 0,
            "cache_hits": sum(row.cache_hit for row in logs),
            "cache_hit_rate": round(sum(row.cache_hit for row in logs) / request_count, 4) if request_count else 0,
            "refusals": sum(row.refused for row in logs),
            "no_results": sum(row.no_result for row in logs),
        },
        "latency_ms": {
            "p50": _percentile([row.total_ms for row in logs], 0.5),
            "p95": _percentile([row.total_ms for row in logs], 0.95),
            "retrieval_average": round(sum(row.retrieval_ms for row in logs) / request_count, 2) if request_count else 0,
            "llm_average": round(sum(row.llm_ms for row in llm_calls) / len(llm_calls), 2) if llm_calls else 0,
        },
        "llm": {
            "calls": len(llm_calls),
            "prompt_tokens": sum(row.prompt_tokens for row in logs),
            "completion_tokens": sum(row.completion_tokens for row in logs),
            "total_tokens": sum(row.total_tokens for row in logs),
            "estimated_cost_usd": round(sum(priced_costs), 8) if priced_costs else None,
            "cost_configured": bool(priced_costs),
            "model": settings.deepseek_model,
        },
        "knowledge_base": {
            **kb,
            "latest_job_seen_at": latest_seen.isoformat() if latest_seen else None,
            "latest_job_posted": latest_posted.isoformat() if latest_posted else None,
            "latest_embedding_at": latest_embedded.isoformat() if latest_embedded else None,
            "retention_days": 60,
        },
        "index": index_stats(session_factory),
        "versions": {
            "embedding_model": settings.embedding_model_id,
            "llm_model": settings.deepseek_model,
            "pipelines": sorted({row.pipeline_version for row in logs if row.pipeline_version}),
            "latest_corpus_version": logs[0].corpus_version if logs else None,
        },
        "privacy": {
            "raw_queries_stored": False,
            "raw_prompts_stored": False,
            "query_identifier": "sha256",
            "log_retention_days": settings.ops_log_retention_days,
        },
        "recent_requests": recent,
    }
