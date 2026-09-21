from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import Base, Job, JobChunk
from app.db.repository import knowledge_stats, upsert_jobs
from app.rag.backfill import backfill_linkedin_descriptions


def make_sessions():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def test_backfill_adds_description_snapshot_and_chunks():
    sessions = make_sessions()
    upsert_jobs(
        sessions,
        [{"id": "li-123", "site": "linkedin", "title": "RAG Engineer", "company": "Example", "location": "Singapore", "job_url": "https://linkedin.com/jobs/view/123", "description": None}],
    )

    result = backfill_linkedin_descriptions(
        sessions,
        fetch_details=lambda job_id: {"description": f"Requirements\n\nPython and vector search for job {job_id}."},
        retries=1,
    )

    assert result.descriptions_added == 1
    assert result.failed == 0
    assert result.chunks_created > 0
    with sessions() as session:
        assert session.scalar(select(Job.description)) == "Requirements\n\nPython and vector search for job 123."
        assert session.scalar(select(func.count()).select_from(JobChunk)) > 0
        stats = knowledge_stats(session)
        assert stats["jobs_with_description"] == 1
        assert stats["jobs_missing_description"] == 0
