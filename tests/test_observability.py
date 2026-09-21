from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import Base, RAGRequestLog
from app.rag.observability import operations_summary, record_rag_request


def sessions():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def test_request_log_is_privacy_safe_and_summary_is_aggregated():
    factory = sessions()
    query = "What skills are required for an AI Agent internship?"
    request_id = record_rag_request(
        factory,
        query=query,
        status="success",
        total_ms=123.4,
        result={
            "intent": "aggregate",
            "cache_hit": False,
            "retrieval_cache_hit": True,
            "sources": [{"job_id": "1"}],
            "analysis_scope": {"candidate_job_count": 10, "analyzed_job_count": 8},
            "corpus_version": "corpus-1",
            "telemetry": {
                "retrieval_ms": 23.4,
                "llm_ms": 100,
                "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
                "llm_model": "deepseek-chat",
                "pipeline_version": "evidence-answer-v2",
            },
        },
    )

    with factory() as session:
        row = session.scalar(select(RAGRequestLog))
        assert row.request_id == request_id
        assert row.query_hash != query
        assert query not in repr(row.__dict__)

    summary = operations_summary(factory, days=7)
    assert summary["requests"]["total"] == 1
    assert summary["latency_ms"]["p50"] == 123.4
    assert summary["llm"]["total_tokens"] == 25
    assert summary["privacy"]["raw_queries_stored"] is False
    assert summary["recent_requests"][0]["request_id"] == request_id


def test_error_trace_does_not_store_error_message_or_query():
    factory = sessions()
    query = "private question"
    record_rag_request(
        factory,
        query=query,
        status="error",
        total_ms=10,
        error_type="LLMProviderError",
    )
    with factory() as session:
        row = session.scalar(select(RAGRequestLog))
        assert row.error_type == "LLMProviderError"
        assert query not in repr(row.__dict__)
