from datetime import date, datetime, timezone

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import Base, Job, JobChunk
from app.db.repository import upsert_jobs
from app.rag.indexing import prepare_chunks
from app.rag.lifecycle import expire_stale_jobs


def sessions():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def job(job_id: str, posted: str | None):
    return {
        "id": job_id,
        "site": "linkedin",
        "title": f"AI Engineer {job_id}",
        "company": "Example",
        "location": "Singapore",
        "job_url": f"https://example/{job_id}",
        "description": "Requirements\n\nPython and RAG.",
        "date_posted": posted,
    }


def test_expiry_deactivates_old_jobs_and_removes_only_derived_chunks():
    factory = sessions()
    ingestion = upsert_jobs(
        factory,
        [job("old", "2026-07-01"), job("current", "2026-09-01"), job("unknown", None)],
    )
    prepare_chunks(factory, job_ids=ingestion.job_ids_to_sync)
    with factory() as session:
        unknown = session.scalar(select(Job).where(Job.source_job_id == "unknown"))
        unknown.last_seen_at = datetime(2026, 6, 1, tzinfo=timezone.utc)
        session.commit()

    result = expire_stale_jobs(factory, retention_days=60, as_of=date(2026, 9, 20))

    assert result.expired_jobs == 2
    assert result.chunks_removed > 0
    with factory() as session:
        jobs = {row.source_job_id: row for row in session.scalars(select(Job)).all()}
        assert jobs["old"].is_active is False
        assert jobs["unknown"].is_active is False
        assert jobs["current"].is_active is True
        assert session.scalar(select(func.count()).select_from(Job)) == 3
        assert session.scalar(select(func.count()).select_from(JobChunk)) > 0
        assert session.scalar(
            select(func.count()).select_from(JobChunk).where(JobChunk.job_id == jobs["old"].id)
        ) == 0


def test_seen_again_reactivates_job_and_schedules_index_sync():
    factory = sessions()
    ingestion = upsert_jobs(factory, [job("unknown", None)])
    prepare_chunks(factory, job_ids=ingestion.job_ids_to_sync)
    with factory() as session:
        stored = session.scalar(select(Job))
        stored.last_seen_at = datetime(2026, 6, 1, tzinfo=timezone.utc)
        session.commit()
    expire_stale_jobs(factory, retention_days=60, as_of=date(2026, 9, 20))

    seen_again = upsert_jobs(factory, [job("unknown", None)])

    assert seen_again.reactivated == 1
    assert len(seen_again.job_ids_to_sync) == 1
    with factory() as session:
        assert session.scalar(select(Job.is_active)) is True
