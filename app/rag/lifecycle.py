from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone

from sqlalchemy import and_, delete, or_, select
from sqlalchemy.orm import sessionmaker

from app.db.models import Job, JobChunk
from app.rag.retrieval import rebuild_sqlite_keyword_index


DEFAULT_RETENTION_DAYS = 60


@dataclass
class LifecycleResult:
    retention_days: int
    cutoff_date: str
    expired_jobs: int = 0
    chunks_removed: int = 0

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


def expire_stale_jobs(
    session_factory: sessionmaker,
    *,
    retention_days: int = DEFAULT_RETENTION_DAYS,
    as_of: date | None = None,
) -> LifecycleResult:
    """Deactivate stale jobs and remove their derived retrieval index data.

    Source records and snapshots remain available for audit or reactivation.
    Jobs with a posting date use that date; jobs without one use last_seen_at.
    """
    if retention_days < 1:
        raise ValueError("retention_days must be positive")
    today = as_of or datetime.now(timezone.utc).date()
    cutoff_date = today - timedelta(days=retention_days)
    cutoff_datetime = datetime.combine(cutoff_date, time.min, tzinfo=timezone.utc)
    result = LifecycleResult(retention_days=retention_days, cutoff_date=cutoff_date.isoformat())

    with session_factory() as session:
        stale_ids = list(
            session.scalars(
                select(Job.id).where(
                    Job.is_active.is_(True),
                    or_(
                        and_(Job.date_posted.is_not(None), Job.date_posted < cutoff_date),
                        and_(Job.date_posted.is_(None), Job.last_seen_at < cutoff_datetime),
                    ),
                )
            ).all()
        )
        if not stale_ids:
            return result

        jobs = session.scalars(select(Job).where(Job.id.in_(stale_ids))).all()
        for job in jobs:
            job.is_active = False
        deletion = session.execute(delete(JobChunk).where(JobChunk.job_id.in_(stale_ids)))
        result.expired_jobs = len(jobs)
        result.chunks_removed = int(deletion.rowcount or 0)
        rebuild_sqlite_keyword_index(session)
        session.commit()
    return result
