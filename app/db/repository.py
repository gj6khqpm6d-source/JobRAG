from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Iterable
from urllib.parse import urlsplit, urlunsplit

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.db.models import Job, JobChunk, JobSnapshot


@dataclass
class IngestResult:
    received: int = 0
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    reactivated: int = 0
    snapshots_created: int = 0
    job_ids_to_sync: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, int]:
        values = asdict(self)
        values.pop("job_ids_to_sync")
        return values


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalized_title(value: Any) -> str:
    return normalize_text(value).casefold()


def canonical_url(value: Any) -> str:
    url = normalize_text(value)
    if not url:
        return ""
    parts = urlsplit(url)
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), "", ""))


def job_fingerprint(record: dict[str, Any]) -> str:
    source = normalize_text(record.get("site")).casefold()
    source_id = normalize_text(record.get("id"))
    if source_id:
        identity = f"{source}:id:{source_id}"
    elif canonical_url(record.get("job_url")):
        identity = f"{source}:url:{canonical_url(record.get('job_url'))}"
    else:
        identity = "|".join(
            [source, normalized_title(record.get("title")), normalize_text(record.get("company")).casefold(), normalize_text(record.get("location")).casefold()]
        )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def record_content_hash(record: dict[str, Any]) -> str:
    relevant = {
        key: record.get(key)
        for key in (
            "title", "company", "location", "description", "job_type", "min_amount", "max_amount",
            "currency", "interval", "is_remote", "date_posted", "skills", "experience_range"
        )
    }
    encoded = json.dumps(relevant, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def parse_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def job_values(record: dict[str, Any], content_hash: str) -> dict[str, Any]:
    return {
        "source": normalize_text(record.get("site")) or "unknown",
        "source_job_id": normalize_text(record.get("id")) or None,
        "title": normalize_text(record.get("title")) or "Untitled",
        "normalized_title": normalized_title(record.get("title")) or "untitled",
        "company": normalize_text(record.get("company")) or None,
        "location": normalize_text(record.get("location")) or None,
        "job_type": normalize_text(record.get("job_type")) or None,
        "min_amount": record.get("min_amount"),
        "max_amount": record.get("max_amount"),
        "currency": normalize_text(record.get("currency")) or None,
        "interval": normalize_text(record.get("interval")) or None,
        "is_remote": record.get("is_remote"),
        "description": record.get("description"),
        "job_url": canonical_url(record.get("job_url")),
        "job_url_direct": normalize_text(record.get("job_url_direct")) or None,
        "date_posted": parse_date(record.get("date_posted")),
        "content_hash": content_hash,
        "raw_data": record,
        "last_seen_at": datetime.now(timezone.utc),
        "is_active": True,
    }


def upsert_jobs(session_factory: sessionmaker, records: Iterable[dict[str, Any]]) -> IngestResult:
    result = IngestResult()
    with session_factory() as session:
        for record in records:
            result.received += 1
            fingerprint = job_fingerprint(record)
            content_hash = record_content_hash(record)
            job = session.scalar(select(Job).where(Job.fingerprint == fingerprint))
            values = job_values(record, content_hash)
            if job is None and values["job_url"]:
                job = session.scalar(
                    select(Job).where(
                        Job.source == values["source"],
                        Job.job_url == values["job_url"],
                    )
                )
            if job is None:
                job = Job(fingerprint=fingerprint, **values)
                session.add(job)
                session.flush()
                session.add(JobSnapshot(job_id=job.id, content_hash=content_hash, description=job.description, raw_data=record))
                result.inserted += 1
                result.snapshots_created += 1
                result.job_ids_to_sync.append(job.id)
            elif job.content_hash != content_hash:
                was_inactive = not job.is_active
                for key, value in values.items():
                    setattr(job, key, value)
                session.add(JobSnapshot(job_id=job.id, content_hash=content_hash, description=job.description, raw_data=record))
                result.updated += 1
                result.snapshots_created += 1
                result.reactivated += int(was_inactive)
                result.job_ids_to_sync.append(job.id)
            else:
                was_inactive = not job.is_active
                job.last_seen_at = values["last_seen_at"]
                job.is_active = True
                result.unchanged += 1
                if was_inactive:
                    result.reactivated += 1
                    result.job_ids_to_sync.append(job.id)
        session.commit()
    return result


def knowledge_stats(session: Session) -> dict[str, Any]:
    stored_jobs = session.scalar(select(func.count()).select_from(Job)) or 0
    total_snapshots = session.scalar(select(func.count()).select_from(JobSnapshot)) or 0
    active_jobs = session.scalar(select(func.count()).select_from(Job).where(Job.is_active.is_(True))) or 0
    expired_jobs = stored_jobs - active_jobs
    jobs_with_description = session.scalar(
        select(func.count()).select_from(Job).where(
            Job.is_active.is_(True),
            Job.description.is_not(None),
            func.length(func.trim(Job.description)) > 0,
        )
    ) or 0
    searchable_jobs = session.scalar(
        select(func.count(func.distinct(JobChunk.job_id))).where(
            JobChunk.embedding.is_not(None),
            JobChunk.embedding_provider == settings.embedding_provider,
            JobChunk.embedding_model == settings.embedding_model_id,
        )
    ) or 0
    sources = dict(
        session.execute(
            select(Job.source, func.count(Job.id)).where(Job.is_active.is_(True)).group_by(Job.source)
        ).all()
    )
    coverage = round(jobs_with_description / active_jobs * 100, 1) if active_jobs else 0.0
    return {
        "total_jobs": active_jobs,
        "stored_jobs": stored_jobs,
        "active_jobs": active_jobs,
        "expired_jobs": expired_jobs,
        "total_snapshots": total_snapshots,
        "jobs_with_description": jobs_with_description,
        "jobs_missing_description": active_jobs - jobs_with_description,
        "searchable_jobs": searchable_jobs,
        "description_coverage_percent": coverage,
        "sources": sources,
    }


def list_jobs(session: Session, *, query: str = "", source: str = "", limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
    statement = select(Job).where(Job.is_active.is_(True))
    if query:
        pattern = f"%{query}%"
        statement = statement.where(or_(Job.title.ilike(pattern), Job.company.ilike(pattern), Job.location.ilike(pattern)))
    if source:
        statement = statement.where(Job.source == source)
    jobs = session.scalars(statement.order_by(Job.date_posted.desc().nullslast(), Job.last_seen_at.desc()).offset(offset).limit(limit)).all()
    return [serialize_job(job) for job in jobs]


def serialize_job(job: Job) -> dict[str, Any]:
    return {
        "id": job.id,
        "source": job.source,
        "source_job_id": job.source_job_id,
        "title": job.title,
        "company": job.company,
        "location": job.location,
        "job_type": job.job_type,
        "min_amount": job.min_amount,
        "max_amount": job.max_amount,
        "currency": job.currency,
        "interval": job.interval,
        "is_remote": job.is_remote,
        "description": job.description,
        "job_url": job.job_url,
        "job_url_direct": job.job_url_direct,
        "date_posted": job.date_posted.isoformat() if job.date_posted else None,
        "first_seen_at": job.first_seen_at.isoformat(),
        "last_seen_at": job.last_seen_at.isoformat(),
        "is_active": job.is_active,
    }
