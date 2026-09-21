from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import Base, JobChunk
from app.db.repository import upsert_jobs
from app.rag.indexing import prepare_chunks
from app.rag.retrieval import RetrievalFilters, _title_family, hybrid_search, ranked_job_search, scope_query


def setup_index():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    records = [
        {"id":"1","site":"linkedin","title":"RAG Engineer","company":"A","location":"Singapore","job_url":"https://x/1","description":"Requirements\n\nBuild retrieval pipelines with Python and vector databases."},
        {"id":"2","site":"indeed","title":"Frontend Engineer","company":"B","location":"London","job_url":"https://x/2","description":"Requirements\n\nBuild React design systems."},
    ]
    upsert_jobs(factory, records)
    prepare_chunks(factory)
    with factory() as session:
        chunks = session.scalars(select(JobChunk).order_by(JobChunk.id)).all()
        for chunk in chunks:
            chunk.embedding = ([1.0] + [0.0] * 1023) if "retrieval" in chunk.content else ([0.0, 1.0] + [0.0] * 1022)
        session.commit()
    return factory


def test_hybrid_retrieval_and_metadata_filter():
    factory = setup_index()
    results = hybrid_search(
        factory,
        query="Python RAG retrieval",
        query_vector=[1.0] + [0.0] * 1023,
        filters=RetrievalFilters(location="Singapore"),
        top_k=3,
    )
    assert results
    assert results[0]["title"] == "RAG Engineer"
    assert results[0]["location"] == "Singapore"
    assert results[0]["vector_score"] == 1.0


def test_job_type_filter_and_retrieval_modes():
    factory = setup_index()
    results = hybrid_search(
        factory,
        query="Python RAG retrieval",
        query_vector=[1.0] + [0.0] * 1023,
        filters=RetrievalFilters(location="Singapore", job_type="internship"),
        top_k=3,
        mode="vector",
    )
    assert results == []


def test_chinese_query_uses_job_aliases_with_local_fts():
    factory = setup_index()
    results = hybrid_search(
        factory,
        query="这个岗位需要哪些RAG技能？",
        query_vector=[1.0] + [0.0] * 1023,
        top_k=3,
        mode="keyword",
    )
    assert results
    assert results[0]["title"] == "RAG Engineer"
    assert results[0]["keyword_score"] is not None


def test_keyword_retrieval_does_not_wait_for_embeddings():
    factory = setup_index()
    with factory() as session:
        for chunk in session.scalars(select(JobChunk)).all():
            chunk.embedding = None
        session.commit()
    results = hybrid_search(
        factory,
        query="RAG skills",
        query_vector=[0.0] * 1024,
        top_k=3,
        mode="keyword",
    )
    assert results
    assert results[0]["title"] == "RAG Engineer"


def test_scope_query_removes_analysis_wording_but_keeps_role():
    query = scope_query("新加坡 RAG Engineer 岗位最常见的技能是什么？")

    assert "RAG Engineer" in query
    assert "最常见" not in query
    assert "技能" not in query


def test_ranked_job_search_groups_chunks_and_returns_top_jobs():
    factory = setup_index()
    result = ranked_job_search(
        factory,
        query="RAG Engineer 最常见的技术要求是什么？",
        query_vector=[1.0] + [0.0] * 1023,
        top_jobs=1,
    )

    assert result["candidate_job_count"] >= 1
    assert result["analyzed_job_count"] == 1
    assert result["jobs"][0]["title"] == "RAG Engineer"
    assert result["jobs"][0]["chunks"]
    assert result["jobs"][0]["rank_score"] > 0


def test_title_family_collapses_phd_and_year_repost_variants():
    phd = _title_family(
        "AI Agent Engineer Intern (Commerce) - 2027 Start (PhD)",
        "Example",
    )
    general = _title_family(
        "AI Agent Engineer Intern (Commerce) - 2027 Start",
        "Example",
    )

    assert phd == general
