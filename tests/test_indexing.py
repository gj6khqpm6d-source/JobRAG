from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import Base, Job, JobChunk
from app.db.repository import upsert_jobs
from app.rag.indexing import index_pending_chunks, index_stats, prepare_chunks
from app.rag.providers import EmbeddingProvider


class FakeEmbeddingProvider(EmbeddingProvider):
    def embed(self, texts):
        return [[float(len(text) % 7)] * 1024 for text in texts]


def sessions():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def test_prepare_and_index_chunks():
    factory = sessions()
    upsert_jobs(factory, [{"id":"li-1","site":"linkedin","title":"RAG Engineer","company":"Example","location":"Singapore","job_url":"https://example/1","description":"Responsibilities\n\nBuild retrieval.\n\nRequirements\n\nPython and vector databases."}])
    prepared = prepare_chunks(factory)
    indexed = index_pending_chunks(factory, FakeEmbeddingProvider(), batch_size=1)
    assert prepared.chunks_created == 2
    assert indexed.chunks_indexed == 2
    assert index_stats(factory)["pending_chunks"] == 0
    with factory() as session:
        assert len(session.scalar(select(JobChunk)).embedding) == 1024


def test_description_change_only_rebuilds_and_reindexes_changed_job():
    factory = sessions()
    first = upsert_jobs(
        factory,
        [
            {"id":"1","site":"linkedin","title":"RAG Engineer","company":"A","location":"Singapore","job_url":"https://example/1","description":"Requirements\n\nPython and RAG."},
            {"id":"2","site":"linkedin","title":"ML Engineer","company":"B","location":"Singapore","job_url":"https://example/2","description":"Requirements\n\nPython and PyTorch."},
        ],
    )
    prepare_chunks(factory, job_ids=first.job_ids_to_sync)
    index_pending_chunks(factory, FakeEmbeddingProvider())
    with factory() as session:
        second_job_id = session.scalar(select(Job.id).where(Job.source_job_id == "2"))
        preserved_ids = list(
            session.scalars(select(JobChunk.id).where(JobChunk.job_id == second_job_id)).all()
        )

    changed = upsert_jobs(
        factory,
        [{"id":"1","site":"linkedin","title":"RAG Engineer","company":"A","location":"Singapore","job_url":"https://example/1","description":"Requirements\n\nPython, RAG, and vector databases."}],
    )
    prepared = prepare_chunks(factory, job_ids=changed.job_ids_to_sync)
    indexed = index_pending_chunks(
        factory,
        FakeEmbeddingProvider(),
        job_ids=changed.job_ids_to_sync,
    )

    assert prepared.jobs_processed == 1
    assert indexed.chunks_indexed == prepared.chunks_created
    with factory() as session:
        assert list(
            session.scalars(select(JobChunk.id).where(JobChunk.job_id == second_job_id)).all()
        ) == preserved_ids
        assert all(
            embedding is not None
            for embedding in session.scalars(
                select(JobChunk.embedding).where(JobChunk.job_id == second_job_id)
            ).all()
        )
