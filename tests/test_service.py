from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import Base, JobChunk
from app.db.repository import upsert_jobs
from app.rag.cache import corpus_version
from app.rag.providers import EmbeddingProvider, LLMProvider
from app.rag.service import answer_question
from app.rag.retrieval import RetrievalFilters


class FakeEmbeddingProvider(EmbeddingProvider):
    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return [[1.0] + [0.0] * 1023 for _ in texts]


class FakeLLMProvider(LLMProvider):
    def __init__(self, answer="grounded answer [1]"):
        self.calls = 0
        self.prompts = []
        self.answer = answer

    def generate(self, *, system, user):
        self.calls += 1
        self.prompts.append((system, user))
        return self.answer


def sessions():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    upsert_jobs(
        factory,
        [
            {
                "id": "1",
                "site": "linkedin",
                "title": "RAG Engineer",
                "company": "Example",
                "location": "Singapore",
                "job_url": "https://example/1",
                "description": "Requirements\n\nPython, RAG, SQL and vector databases.",
            },
            {
                "id": "2",
                "site": "linkedin",
                "title": "AI Intern",
                "company": "Example",
                "location": "Singapore",
                "job_url": "https://example/2",
                "description": "Skills\n\nPython and machine learning.",
            },
        ],
    )
    from app.rag.indexing import prepare_chunks

    prepare_chunks(factory)
    with factory() as session:
        for chunk in session.scalars(select(JobChunk)).all():
            chunk.embedding = [1.0] + [0.0] * 1023
        session.commit()
    return factory


def test_answer_cache_avoids_second_embedding_and_llm_call():
    factory = sessions()
    embedding = FakeEmbeddingProvider()
    llm = FakeLLMProvider()
    first = answer_question(
        factory,
        question="What skills does the RAG Engineer role require?",
        embedding_provider=embedding,
        llm_provider=llm,
        filters=RetrievalFilters(location="Singapore"),
    )
    second = answer_question(
        factory,
        question="What skills does the RAG Engineer role require?",
        embedding_provider=embedding,
        llm_provider=llm,
        filters=RetrievalFilters(location="Singapore"),
    )
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert embedding.calls == 1
    assert llm.calls == 1


def test_unchanged_scrape_does_not_invalidate_corpus_cache_version():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    record = {
        "id": "stable-1",
        "site": "linkedin",
        "title": "AI Agent Intern",
        "company": "Example",
        "location": "Singapore",
        "job_type": "internship",
        "job_url": "https://example/stable-1",
        "description": "Requirements\n\nPython and retrieval systems.",
        "date_posted": "2026-09-20",
    }
    upsert_jobs(factory, [record])
    from app.rag.indexing import prepare_chunks

    prepare_chunks(factory)
    before_refresh = corpus_version(factory)
    upsert_jobs(factory, [record])
    after_unchanged_refresh = corpus_version(factory)
    assert after_unchanged_refresh == before_refresh

    changed = {**record, "description": "Requirements\n\nPython, retrieval, and evaluation."}
    upsert_jobs(factory, [changed])
    assert corpus_version(factory) != before_refresh


def test_aggregate_route_analyzes_ranked_jobs_with_job_level_statistics():
    factory = sessions()
    embedding = FakeEmbeddingProvider()
    llm = FakeLLMProvider()
    result = answer_question(
        factory,
        question="What are the most common skills in these jobs?",
        embedding_provider=embedding,
        llm_provider=llm,
    )
    assert result["intent"] == "aggregate"
    assert result["statistics"]["total_jobs"] == 2
    assert result["statistics"]["denominator"] == "top_relevant_jobs"
    assert result["statistics"]["skill_counts"][0]["skill"] == "Python"
    assert len({source["job_id"] for source in result["sources"]}) == len(result["sources"])
    assert result["answer_validation"]["status"] == "passed"
    assert result["statistics"]["skill_counts"][0]["job_ids"]
    assert embedding.calls == 1
    assert llm.calls == 1
    assert "最相关岗位上下文" in llm.prompts[0][1]


def test_invalid_model_citation_falls_back_to_traceable_evidence():
    factory = sessions()
    result = answer_question(
        factory,
        question="What are the most common skills in these jobs?",
        embedding_provider=FakeEmbeddingProvider(),
        llm_provider=FakeLLMProvider("unsupported answer [99]"),
    )

    assert result["answer_validation"]["status"] == "failed"
    assert result["answer_validation"]["fallback_used"] is True
    assert result["answer_validation"]["invalid_citations"] == [99]
    assert "已回退为可核查统计" in result["answer"]
    assert "[99]" not in result["answer"]


def test_explicit_global_aggregate_route_keeps_full_corpus_behavior():
    factory = sessions()
    embedding = FakeEmbeddingProvider()
    llm = FakeLLMProvider()
    result = answer_question(
        factory,
        question="整个知识库的所有岗位中最常见的技能是什么？",
        embedding_provider=embedding,
        llm_provider=llm,
    )

    assert result["intent"] == "global_aggregate"
    assert result["statistics"]["total_jobs"] == 2
    assert embedding.calls == 0
    assert "全库统计证据" in llm.prompts[0][1]


def test_obvious_out_of_scope_question_is_refused_without_model_calls():
    factory = sessions()
    embedding = FakeEmbeddingProvider()
    llm = FakeLLMProvider()
    result = answer_question(
        factory,
        question="今天的天气如何？",
        embedding_provider=embedding,
        llm_provider=llm,
    )
    assert result["intent"] == "refusal"
    assert embedding.calls == 0
    assert llm.calls == 0
