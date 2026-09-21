import json
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import Base, Job
from app.db.repository import upsert_jobs
from app.eval.dataset import EVAL_CATEGORIES, load_questions
from app.eval.metrics import condensed_graded_ndcg_at_k, graded_ndcg_at_k, ndcg_at_k, recall_at_k, reciprocal_rank
from app.eval.quality import run_quality_evaluation
from app.eval.runner import unique_in_order
from app.eval.tuning import load_splits, run_tuning, tuning_configs
from app.rag.indexing import index_pending_chunks, prepare_chunks
from app.rag.providers import EmbeddingProvider


class FakeEmbeddingProvider(EmbeddingProvider):
    def embed(self, texts):
        return [[1.0] + [0.0] * 1023 for _ in texts]


def test_evaluation_question_set_has_five_categories():
    path = Path(__file__).parents[1] / "data" / "eval" / "questions.jsonl"
    questions = load_questions(path)
    assert len(questions) == 25
    assert {question.category for question in questions} == EVAL_CATEGORIES


def test_ranking_metrics_are_deterministic():
    retrieved = ["job-c", "job-a", "job-b"]
    relevant = ["job-a", "job-b"]
    assert recall_at_k(retrieved, relevant, 2) == 0.5
    assert reciprocal_rank(retrieved, relevant, 3) == 0.5
    assert 0 < ndcg_at_k(retrieved, relevant, 3) < 1


def test_job_level_ranking_deduplicates_chunk_results_in_order():
    assert unique_in_order(["job-a", "job-a", "job-b", "job-a", "job-c"]) == [
        "job-a",
        "job-b",
        "job-c",
    ]


def test_graded_ndcg_rewards_high_relevance_first():
    relevance = {"job-a": 2, "job-b": 1}
    assert graded_ndcg_at_k(["job-a", "job-b"], relevance, 2) == 1.0
    assert graded_ndcg_at_k(["job-b", "job-a"], relevance, 2) < 1.0


def test_condensed_ndcg_does_not_treat_unreviewed_jobs_as_negative():
    relevance = {"job-a": 2, "job-b": 1}
    assert condensed_graded_ndcg_at_k(["unknown", "job-a", "job-b"], relevance, 3) == 1.0


def test_tuning_grid_and_split_are_deterministic(tmp_path):
    root = Path(__file__).parents[1]
    splits = load_splits(root / "data" / "eval" / "splits.json")
    assert len(splits["development"]) == 15
    assert len(splits["test"]) == 5
    assert len(splits["refusal"]) == 5
    assert len(tuning_configs()) == 40
    report = run_tuning(
        root / "data" / "eval" / "questions.jsonl",
        tmp_path / "missing-annotations.json",
        root / "data" / "eval" / "splits.json",
    )
    assert report["status"] == "insufficient_annotations"
    assert report["experiments"] == []
    assert report["test_set_evaluated"] is False


def test_quality_gate_covers_retrieval_answers_refusal_and_index_health(tmp_path):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    ingestion = upsert_jobs(
        factory,
        [{
            "id": "job-1",
            "site": "linkedin",
            "title": "RAG Engineer",
            "company": "Example",
            "location": "Singapore",
            "job_url": "https://example/1",
            "description": "Requirements\n\nPython, RAG, and vector databases.",
            "date_posted": "2026-09-01",
        }],
    )
    prepare_chunks(factory, job_ids=ingestion.job_ids_to_sync)
    index_pending_chunks(factory, FakeEmbeddingProvider())
    with factory() as session:
        job_id = session.scalar(select(Job.id))

    questions = tmp_path / "questions.jsonl"
    questions.write_text(
        "\n".join([
            json.dumps({"id":"skills-1","category":"skills","query":"What skills does the RAG Engineer require?","filters":{"location":"Singapore"},"should_refuse":False}),
            json.dumps({"id":"refuse-1","category":"insufficient_evidence","query":"今天天气如何？","filters":{},"should_refuse":True}),
        ]),
        encoding="utf-8",
    )
    annotations = tmp_path / "annotations.json"
    annotations.write_text(
        json.dumps({"version":1,"annotations":{"skills-1":{"job_judgments":[{"job_id":job_id,"relevance":2}],"evidence_chunk_ids":[],"should_refuse":False}}}),
        encoding="utf-8",
    )
    gates = tmp_path / "gates.json"
    gates.write_text(
        json.dumps({
            "version":1,
            "minimum_questions":2,
            "minimum_annotated_answerable_questions":1,
            "minimum_recall_at_8":1.0,
            "minimum_graded_ndcg_at_8":1.0,
            "minimum_filter_accuracy":1.0,
            "minimum_analysis_invariant_rate":1.0,
            "minimum_citation_validity_rate":1.0,
            "minimum_refusal_accuracy":1.0,
            "minimum_index_coverage":1.0,
            "maximum_active_stale_jobs":0,
            "maximum_inactive_chunks":0,
        }),
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"
    report = run_quality_evaluation(
        session_factory=factory,
        embedding_provider=FakeEmbeddingProvider(),
        question_path=questions,
        annotation_path=annotations,
        gates_path=gates,
        split_path=None,
        report_path=report_path,
    )

    assert report["passed"] is True
    assert report["layers"]["retrieval"]["recall_at_8"] == 1.0
    assert report["layers"]["answer"]["refusal_accuracy"] == 1.0
    assert report["layers"]["operations"]["index_coverage"] == 1.0
    assert json.loads(report_path.read_text(encoding="utf-8"))["passed"] is True
