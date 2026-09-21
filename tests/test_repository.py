from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import Base, Job, JobSnapshot
from app.db.repository import job_fingerprint, upsert_jobs


def make_sessions():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def record(description="Build production RAG systems"):
    return {
        "id": "li-123",
        "site": "linkedin",
        "title": "AI Engineer",
        "company": "Example",
        "location": "Singapore",
        "job_url": "https://linkedin.com/jobs/view/123?tracking=abc",
        "description": description,
        "date_posted": "2026-09-01",
    }


def test_fingerprint_is_stable_for_same_source_id():
    first = record()
    second = {**record(), "job_url": "https://different.example/job"}
    assert job_fingerprint(first) == job_fingerprint(second)


def test_upsert_deduplicates_and_creates_change_snapshots():
    sessions = make_sessions()
    first = upsert_jobs(sessions, [record()])
    duplicate = upsert_jobs(sessions, [record()])
    changed = upsert_jobs(sessions, [record("Build and evaluate hybrid RAG systems")])

    assert first.inserted == 1
    assert duplicate.unchanged == 1
    assert changed.updated == 1
    assert changed.snapshots_created == 1
    assert len(first.job_ids_to_sync) == 1
    assert duplicate.job_ids_to_sync == []
    assert changed.job_ids_to_sync == first.job_ids_to_sync
    with sessions() as session:
        assert session.scalar(select(func.count()).select_from(Job)) == 1
        assert session.scalar(select(func.count()).select_from(JobSnapshot)) == 2


def test_upsert_deduplicates_same_canonical_url_with_changed_source_id():
    sessions = make_sessions()
    upsert_jobs(sessions, [record()])
    duplicate = upsert_jobs(
        sessions,
        [{**record(), "id": "li-new-id", "job_url": "https://linkedin.com/jobs/view/123/"}],
    )

    assert duplicate.unchanged == 1
    with sessions() as session:
        assert session.scalar(select(func.count()).select_from(Job)) == 1
