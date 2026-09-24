from __future__ import annotations

import io
import math
import threading
import time
import uuid
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from jobspy import scrape_jobs
from jobspy.model import Country, Site
from app.config import settings
from app.db.repository import knowledge_stats, list_jobs, upsert_jobs
from app.db.session import SessionLocal, initialize_database
from app.eval.annotations import all_annotations, get_annotation, save_annotation
from app.eval.dataset import load_questions
from app.eval.quality import load_latest_quality_report, run_quality_evaluation
from app.rag.backfill import backfill_linkedin_descriptions
from app.rag.contract import load_business_contract
from app.rag.indexing import IndexingResult, index_pending_chunks, index_stats, prepare_chunks
from app.rag.lifecycle import DEFAULT_RETENTION_DAYS, expire_stale_jobs
from app.rag.observability import operations_summary, prune_request_logs, record_rag_request
from app.rag.providers import DeepSeekProvider, EmbeddingProviderError, LLMProviderError, get_embedding_provider
from app.rag.retrieval import RetrievalFilters, hybrid_search
from app.rag.service import answer_question, retrieve_question
from app.scrape_scheduler import (
    DailyScrapeScheduler,
    get_schedule,
    latest_run,
    next_run_time,
    save_schedule,
)


ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
EVAL_QUESTIONS = ROOT.parent / "data" / "eval" / "questions.jsonl"
SUPPORTED_SITES = [site.value for site in Site]


class SearchRequest(BaseModel):
    sites: list[str] = Field(default_factory=lambda: ["indeed", "linkedin"])
    search_term: str = ""
    google_search_term: str = ""
    location: str = ""
    distance: int = Field(default=50, ge=0, le=1000)
    is_remote: bool = False
    job_type: Literal["", "fulltime", "parttime", "contract", "temporary", "internship"] = ""
    easy_apply: bool = False
    results_wanted: int = Field(default=20, ge=1, le=1000)
    country_indeed: str = "singapore"
    proxies: list[str] = Field(default_factory=list)
    ca_cert: str = ""
    description_format: Literal["markdown", "html", "plain"] = "markdown"
    linkedin_fetch_description: bool = True
    linkedin_company_ids: list[int] = Field(default_factory=list)
    offset: int = Field(default=0, ge=0, le=10000)
    hours_old: int | None = Field(default=None, ge=1, le=87600)
    enforce_annual_salary: bool = False
    verbose: Literal[0, 1, 2] = 1
    user_agent: str = ""

    @field_validator("sites")
    @classmethod
    def validate_sites(cls, value: list[str]) -> list[str]:
        cleaned = list(dict.fromkeys(value))
        invalid = set(cleaned) - set(SUPPORTED_SITES)
        if not cleaned:
            raise ValueError("请至少选择一个招聘网站")
        if invalid:
            raise ValueError(f"不支持的网站: {', '.join(sorted(invalid))}")
        return cleaned

    @field_validator("country_indeed")
    @classmethod
    def validate_country(cls, value: str) -> str:
        Country.from_string(value)
        return value

    @field_validator("search_term", "google_search_term", "location", "ca_cert", "user_agent")
    @classmethod
    def trim_strings(cls, value: str) -> str:
        return value.strip()


class RetrievalRequest(BaseModel):
    query: str = Field(min_length=2, max_length=1000)
    source: str = ""
    location: str = ""
    title: str = ""
    job_type: str = ""
    date_after: date | None = None
    top_k: int = Field(default=8, ge=1, le=30)


class AskRequest(RetrievalRequest):
    pass


class AnnotationRequest(BaseModel):
    job_judgments: list[dict[str, Any]] = Field(default_factory=list)
    evidence_chunk_ids: list[int] = Field(default_factory=list)
    answer_key_points: list[str] = Field(default_factory=list)
    should_refuse: bool = False
    notes: str = ""

    @field_validator("job_judgments")
    @classmethod
    def validate_judgments(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cleaned: list[dict[str, Any]] = []
        seen: set[str] = set()
        for judgment in value:
            job_id = str(judgment.get("job_id", "")).strip()
            relevance = judgment.get("relevance")
            if not job_id or job_id in seen or relevance not in (0, 1, 2):
                raise ValueError("岗位标注必须包含唯一 job_id，relevance 只能是 0、1 或 2")
            seen.add(job_id)
            cleaned.append({"job_id": job_id, "relevance": relevance})
        return cleaned

    @field_validator("evidence_chunk_ids")
    @classmethod
    def validate_chunk_ids(cls, value: list[int]) -> list[int]:
        if any(chunk_id < 1 for chunk_id in value):
            raise ValueError("证据片段 ID 必须为正整数")
        return list(dict.fromkeys(value))

    @field_validator("answer_key_points")
    @classmethod
    def clean_key_points(cls, value: list[str]) -> list[str]:
        return [point.strip() for point in value if point.strip()]

    @field_validator("notes")
    @classmethod
    def clean_notes(cls, value: str) -> str:
        return value.strip()


class AutoScrapeScheduleRequest(BaseModel):
    enabled: bool = True
    sites: list[str] = Field(default_factory=lambda: ["linkedin"], min_length=1, max_length=2)
    search_term: str = Field(default="AI Agent", min_length=2, max_length=255)
    location: str = Field(default="Singapore", min_length=2, max_length=255)
    job_type: Literal["", "fulltime", "parttime", "contract", "temporary", "internship"] = "internship"
    results_per_site: int = Field(default=10, ge=1, le=20)
    lookback_hours: int = Field(default=72, ge=1, le=168)
    max_index_chunks: int = Field(default=100, ge=1, le=500)

    @field_validator("sites")
    @classmethod
    def validate_scheduled_sites(cls, value: list[str]) -> list[str]:
        cleaned = list(dict.fromkeys(value))
        invalid = set(cleaned) - set(SUPPORTED_SITES)
        if not cleaned or len(cleaned) > 2:
            raise ValueError("自动抓取最多选择 2 个来源")
        if invalid:
            raise ValueError(f"不支持的网站: {', '.join(sorted(invalid))}")
        return cleaned

    @field_validator("search_term", "location")
    @classmethod
    def trim_schedule_strings(cls, value: str) -> str:
        return value.strip()


class TaskStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._tasks: dict[str, dict[str, Any]] = {}

    def create(self, request: SearchRequest) -> dict[str, Any]:
        task_id = uuid.uuid4().hex
        task = {
            "id": task_id,
            "status": "queued",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "finished_at": None,
            "completed_sites": [],
            "total_sites": len(request.sites),
            "errors": {},
            "results": [],
            "ingestion": None,
            "request": request.model_dump(),
        }
        with self._lock:
            self._tasks[task_id] = task
        return self.public(task_id)

    def update(self, task_id: str, **values: Any) -> None:
        with self._lock:
            self._tasks[task_id].update(values)

    def site_done(self, task_id: str, site: str, records: list[dict[str, Any]], error: str | None) -> None:
        with self._lock:
            task = self._tasks[task_id]
            task["completed_sites"].append(site)
            task["results"].extend(records)
            if error:
                task["errors"][site] = error

    def site_retry_done(self, task_id: str, site: str, records: list[dict[str, Any]], error: str | None) -> None:
        with self._lock:
            task = self._tasks[task_id]
            task["completed_sites"].append(site)
            task["results"].extend(records)
            if error:
                task["errors"][site] = error
            else:
                task["errors"].pop(site, None)

    def get(self, task_id: str) -> dict[str, Any]:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise KeyError(task_id)
            return task

    def public(self, task_id: str, include_results: bool = False) -> dict[str, Any]:
        with self._lock:
            task = self.get(task_id)
            output = {key: value for key, value in task.items() if key != "results"}
            output["result_count"] = len(task["results"])
            if include_results:
                output["results"] = list(task["results"])
            return output


store = TaskStore()
runner = ThreadPoolExecutor(max_workers=4, thread_name_prefix="jobspy-task")
scrape_scheduler: DailyScrapeScheduler | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    initialize_database()
    prune_request_logs(SessionLocal)
    expire_stale_jobs(SessionLocal, retention_days=DEFAULT_RETENTION_DAYS)
    prepare_chunks(SessionLocal)
    if scrape_scheduler:
        scrape_scheduler.start()
    try:
        yield
    finally:
        if scrape_scheduler:
            scrape_scheduler.stop()


app = FastAPI(title="JobRAG Local", version="1.1.0", lifespan=lifespan)


@app.middleware("http")
async def disable_local_ui_cache(request, call_next):
    """Always serve the current local UI instead of a stale browser copy."""
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    if request.url.path == "/":
        response.headers["Clear-Site-Data"] = '"cache"'
    return response


def json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd.isna(value) is True:
        return None
    return value


def dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {key: json_value(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def scrape_site(site: str, request: SearchRequest) -> list[dict[str, Any]]:
    values = request.model_dump()
    values.pop("sites")
    values["site_name"] = site
    values["search_term"] = values["search_term"] or None
    values["google_search_term"] = values["google_search_term"] or None
    values["location"] = values["location"] or None
    values["job_type"] = values["job_type"] or None
    values["proxies"] = values["proxies"] or None
    values["ca_cert"] = values["ca_cert"] or None
    values["linkedin_company_ids"] = values["linkedin_company_ids"] or None
    values["user_agent"] = values["user_agent"] or None
    frame = scrape_jobs(**values)
    return dataframe_records(frame)


def run_search(
    task_id: str,
    request: SearchRequest,
    *,
    index_limit: int = 5000,
    index_backlog: bool = False,
    retry_failed_sites: bool = False,
) -> None:
    store.update(task_id, status="running")
    with ThreadPoolExecutor(max_workers=min(len(request.sites), 8)) as site_pool:
        futures = {site_pool.submit(scrape_site, site, request): site for site in request.sites}
        for future in as_completed(futures):
            site = futures[future]
            try:
                records = future.result()
                store.site_done(task_id, site, records, None)
            except Exception as exc:  # one blocked site must not discard other results
                store.site_done(task_id, site, [], f"{type(exc).__name__}: {exc}")

    if retry_failed_sites:
        failed_sites = list(store.get(task_id)["errors"])
        for site in failed_sites:
            try:
                records = scrape_site(site, request)
                store.site_retry_done(task_id, site, records, None)
            except Exception as exc:
                store.site_retry_done(task_id, site, [], f"retry failed: {type(exc).__name__}: {exc}")

    task = store.get(task_id)
    ingestion_result = None
    if task["results"]:
        try:
            ingestion_result = upsert_jobs(SessionLocal, task["results"])
            lifecycle = expire_stale_jobs(SessionLocal, retention_days=DEFAULT_RETENTION_DAYS)
            chunking = prepare_chunks(SessionLocal, job_ids=ingestion_result.job_ids_to_sync)
            indexing = IndexingResult()
            if not index_backlog:
                indexing = index_pending_chunks(
                    SessionLocal,
                    get_embedding_provider(),
                    limit=index_limit,
                    batch_size=settings.embedding_batch_size,
                    job_ids=ingestion_result.job_ids_to_sync,
                )
            ingestion = ingestion_result.to_dict()
            ingestion["chunking"] = chunking.to_dict()
            ingestion["indexing"] = indexing.to_dict()
            ingestion["lifecycle"] = lifecycle.to_dict()
            store.update(task_id, ingestion=ingestion)
        except Exception as exc:
            task["errors"]["knowledge_base"] = f"{type(exc).__name__}: {exc}"
            store.update(task_id, ingestion=None)
    if index_backlog:
        try:
            provider = get_embedding_provider()
            priority_indexing = IndexingResult()
            if ingestion_result and ingestion_result.job_ids_to_sync:
                priority_indexing = index_pending_chunks(
                    SessionLocal,
                    provider,
                    limit=index_limit,
                    batch_size=settings.embedding_batch_size,
                    job_ids=ingestion_result.job_ids_to_sync,
                )
            remaining_limit = max(index_limit - priority_indexing.chunks_indexed, 0)
            backlog_indexing = IndexingResult()
            if remaining_limit:
                backlog_indexing = index_pending_chunks(
                    SessionLocal,
                    provider,
                    limit=remaining_limit,
                    batch_size=settings.embedding_batch_size,
                )
            combined = IndexingResult(
                chunks_indexed=priority_indexing.chunks_indexed + backlog_indexing.chunks_indexed,
                batches=priority_indexing.batches + backlog_indexing.batches,
            )
            ingestion = dict(store.get(task_id).get("ingestion") or {})
            ingestion["indexing"] = combined.to_dict()
            ingestion["index_status"] = index_stats(SessionLocal)
            store.update(task_id, ingestion=ingestion)
        except Exception as exc:
            task["errors"]["indexing"] = f"{type(exc).__name__}: {exc}"
            store.update(task_id, ingestion=task.get("ingestion"))
    if task["results"] and task["errors"]:
        status = "partial"
    elif task["errors"]:
        status = "failed"
    else:
        status = "completed"
    store.update(task_id, status=status, finished_at=datetime.now().isoformat(timespec="seconds"))


def run_scheduled_scrape(config: dict[str, Any], _scheduled_for: datetime) -> dict[str, Any]:
    request = SearchRequest(
        sites=config["sites"],
        search_term=config["search_term"],
        location=config["location"],
        job_type=config["job_type"],
        results_wanted=config["results_per_site"],
        hours_old=config["lookback_hours"],
        country_indeed="singapore",
        description_format="markdown",
        linkedin_fetch_description=True,
        verbose=0,
    )
    task = store.create(request)
    run_search(
        task["id"],
        request,
        index_limit=config["max_index_chunks"],
        index_backlog=True,
        retry_failed_sites=True,
    )
    task = store.get(task["id"])
    ingestion = task.get("ingestion") or {}
    indexing = ingestion.get("indexing") or {}
    return {
        "status": "completed_with_warnings" if indexing.get("error_summary") else task["status"],
        "results_found": len(task["results"]),
        "inserted": int(ingestion.get("inserted", 0)),
        "updated": int(ingestion.get("updated", 0)),
        "unchanged": int(ingestion.get("unchanged", 0)),
        "chunks_indexed": int(indexing.get("chunks_indexed", 0)),
        "chunks_pending": int((ingestion.get("index_status") or index_stats(SessionLocal)).get("pending_chunks", 0)),
        "error_summary": "; ".join(filter(None, [
            "; ".join(f"{site}: {error}" for site, error in task["errors"].items()),
            indexing.get("error_summary"),
        ])),
    }


scrape_scheduler = DailyScrapeScheduler(SessionLocal, run_scheduled_scrape)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/meta")
def meta() -> dict[str, Any]:
    countries = sorted({country.value[0].split(",")[0] for country in Country if country not in (Country.US_CANADA, Country.WORLDWIDE)})
    return {"sites": SUPPORTED_SITES, "countries": countries}


@app.post("/api/search", status_code=202)
def create_search(request: SearchRequest) -> dict[str, Any]:
    if not request.search_term and not request.google_search_term and not request.linkedin_company_ids:
        raise HTTPException(422, "请输入职位关键词、Google 搜索语句或 LinkedIn 公司 ID")
    if "google" in request.sites and not request.google_search_term:
        request.google_search_term = " ".join(filter(None, [request.search_term, "jobs", request.location]))
    task = store.create(request)
    runner.submit(run_search, task["id"], request)
    return task


@app.get("/api/search/{task_id}")
def search_status(task_id: str, include_results: bool = False) -> dict[str, Any]:
    try:
        return store.public(task_id, include_results=include_results)
    except KeyError:
        raise HTTPException(404, "搜索任务不存在") from None


@app.get("/api/kb/stats")
def kb_stats() -> dict[str, Any]:
    with SessionLocal() as session:
        return knowledge_stats(session)


@app.get("/api/auto-scrape")
def auto_scrape_status() -> dict[str, Any]:
    config = get_schedule(SessionLocal)
    next_run = next_run_time(SessionLocal, config["enabled"])
    return {
        "schedule": config,
        "next_run_at": next_run.isoformat(timespec="minutes") if next_run else None,
        "latest_run": latest_run(SessionLocal),
    }


@app.put("/api/auto-scrape")
def update_auto_scrape(request: AutoScrapeScheduleRequest) -> dict[str, Any]:
    config = save_schedule(SessionLocal, request.model_dump())
    scrape_scheduler.notify()
    next_run = next_run_time(SessionLocal, config["enabled"])
    return {
        "schedule": config,
        "next_run_at": next_run.isoformat(timespec="minutes") if next_run else None,
        "latest_run": latest_run(SessionLocal),
    }


@app.post("/api/kb/lifecycle")
def kb_apply_lifecycle(retention_days: int = DEFAULT_RETENTION_DAYS) -> dict[str, Any]:
    if retention_days < 1 or retention_days > 3650:
        raise HTTPException(422, "retention_days 必须介于 1–3650")
    result = expire_stale_jobs(SessionLocal, retention_days=retention_days)
    with SessionLocal() as session:
        stats = knowledge_stats(session)
    return {**result.to_dict(), "stats": stats, "index": index_stats(SessionLocal)}


@app.get("/api/kb/contract")
def kb_contract() -> dict[str, Any]:
    """Expose the active business contract for the UI and integration checks."""
    return load_business_contract()


@app.get("/api/kb/jobs")
def kb_jobs(query: str = "", source: str = "", limit: int = 50, offset: int = 0) -> dict[str, Any]:
    if limit < 1 or limit > 200 or offset < 0:
        raise HTTPException(422, "limit 必须介于 1–200，offset 不能为负数")
    with SessionLocal() as session:
        jobs = list_jobs(session, query=query.strip(), source=source.strip(), limit=limit, offset=offset)
    return {"items": jobs, "count": len(jobs), "limit": limit, "offset": offset}


@app.post("/api/kb/backfill/linkedin")
def kb_backfill_linkedin(limit: int = 25) -> dict[str, Any]:
    if limit < 1 or limit > 100:
        raise HTTPException(422, "limit 必须介于 1–100")
    result = backfill_linkedin_descriptions(SessionLocal, limit=limit)
    indexing = index_pending_chunks(
        SessionLocal,
        get_embedding_provider(),
        limit=5000,
        batch_size=settings.embedding_batch_size,
        job_ids=result.job_ids_to_sync,
    )
    return {**result.to_dict(), "indexing": indexing.to_dict()}


@app.get("/api/kb/index")
def kb_index_status() -> dict[str, Any]:
    return index_stats(SessionLocal)


@app.post("/api/kb/index")
def kb_create_index(limit: int = 500, batch_size: int = 32, retry_failed: bool = False) -> dict[str, Any]:
    max_limit = 100 if retry_failed else 5000
    if limit < 1 or limit > max_limit or batch_size < 1 or batch_size > 100:
        raise HTTPException(422, f"limit 必须介于 1–{max_limit}，batch_size 必须介于 1–100")
    try:
        result = index_pending_chunks(
            SessionLocal,
            get_embedding_provider(),
            limit=limit,
            batch_size=batch_size,
            retry_failed=retry_failed,
        )
    except EmbeddingProviderError as exc:
        raise HTTPException(503, str(exc)) from None
    return {**result.to_dict(), **index_stats(SessionLocal)}


def request_filters(request: RetrievalRequest) -> RetrievalFilters:
    return RetrievalFilters(
        source=request.source.strip(),
        location=request.location.strip(),
        title=request.title.strip(),
        job_type=request.job_type.strip(),
        date_after=request.date_after,
    )


@app.post("/api/kb/retrieve")
def kb_retrieve(request: RetrievalRequest) -> dict[str, Any]:
    try:
        retrieval = retrieve_question(
            SessionLocal,
            question=request.query,
            embedding_provider=get_embedding_provider(),
            filters=request_filters(request),
            top_k=request.top_k,
        )
    except EmbeddingProviderError as exc:
        raise HTTPException(503, str(exc)) from None
    results = retrieval["items"]
    return {
        "query": request.query,
        "count": len(results),
        "items": results,
        "cache_hit": retrieval["cache_hit"],
        "corpus_version": retrieval["corpus_version"],
    }


@app.post("/api/kb/ask")
def kb_ask(request: AskRequest) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = answer_question(
            SessionLocal,
            question=request.query,
            embedding_provider=get_embedding_provider(),
            llm_provider=DeepSeekProvider(),
            filters=request_filters(request),
            top_k=request.top_k,
        )
    except (EmbeddingProviderError, LLMProviderError) as exc:
        record_rag_request(
            SessionLocal,
            query=request.query,
            status="error",
            total_ms=(time.perf_counter() - started) * 1000,
            error_type=type(exc).__name__,
        )
        raise HTTPException(503, str(exc)) from None
    request_id = record_rag_request(
        SessionLocal,
        query=request.query,
        status="success",
        total_ms=(time.perf_counter() - started) * 1000,
        result=result,
    )
    return {**result, "request_id": request_id}


@app.get("/api/ops/summary")
def ops_summary(days: int = 7) -> dict[str, Any]:
    if days < 1 or days > 90:
        raise HTTPException(422, "days 必须介于 1–90")
    return operations_summary(SessionLocal, days=days)


def _evaluation_question(question_id: str):
    for question in load_questions(EVAL_QUESTIONS):
        if question.id == question_id:
            return question
    raise HTTPException(404, "评估问题不存在")


def _evaluation_question_public(question, annotation: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "id": question.id,
        "category": question.category,
        "query": question.query,
        "filters": question.filters,
        "should_refuse": question.should_refuse,
        "notes": question.notes,
        "annotated": annotation is not None,
        "annotation": annotation,
    }


@app.get("/api/eval/questions")
def evaluation_questions() -> dict[str, Any]:
    annotations = all_annotations()
    items = [
        _evaluation_question_public(question, annotations.get(question.id))
        for question in load_questions(EVAL_QUESTIONS)
    ]
    return {
        "items": items,
        "count": len(items),
        "annotated_count": sum(item["annotated"] for item in items),
    }


@app.get("/api/eval/quality")
def evaluation_quality_report() -> dict[str, Any]:
    report = load_latest_quality_report()
    return {"available": report is not None, "report": report}


@app.post("/api/eval/quality/run")
def run_evaluation_quality() -> dict[str, Any]:
    try:
        return run_quality_evaluation(embedding_provider=get_embedding_provider())
    except (EmbeddingProviderError, ValueError) as exc:
        raise HTTPException(503, str(exc)) from None


@app.get("/api/eval/questions/{question_id}/candidates")
def evaluation_candidates(question_id: str, top_k: int = 20) -> dict[str, Any]:
    if top_k < 5 or top_k > 50:
        raise HTTPException(422, "top_k 必须介于 5–50")
    question = _evaluation_question(question_id)
    filters = RetrievalFilters(
        source=question.filters.get("source", ""),
        location=question.filters.get("location", ""),
        title=question.filters.get("title", ""),
        job_type=question.filters.get("job_type", ""),
    )
    try:
        query_vector = get_embedding_provider().embed([question.query])[0]
        groups: dict[str, dict[str, Any]] = {}
        seen_chunks: set[int] = set()
        for mode in ("hybrid", "vector", "keyword"):
            results = hybrid_search(
                SessionLocal,
                query=question.query,
                query_vector=query_vector,
                filters=filters,
                top_k=top_k,
                mode=mode,
            )
            for result in results:
                job_id = result["job_id"]
                group = groups.setdefault(
                    job_id,
                    {
                        "job_id": job_id,
                        "title": result["title"],
                        "company": result["company"],
                        "location": result["location"],
                        "job_type": result["job_type"],
                        "source": result["source"],
                        "job_url": result["job_url"],
                        "retrieved_by": [],
                        "chunks": [],
                    },
                )
                if mode not in group["retrieved_by"]:
                    group["retrieved_by"].append(mode)
                if result["chunk_id"] in seen_chunks:
                    continue
                seen_chunks.add(result["chunk_id"])
                group["chunks"].append(
                    {
                        "chunk_id": result["chunk_id"],
                        "section_type": result["section_type"],
                        "content": result["content"],
                        "score": result["score"],
                    }
                )
        candidates = list(groups.values())
    except EmbeddingProviderError as exc:
        raise HTTPException(503, str(exc)) from None
    return {
        "question": _evaluation_question_public(question, get_annotation(question.id)),
        "candidates": candidates,
    }


@app.put("/api/eval/questions/{question_id}/annotation")
def save_evaluation_annotation(question_id: str, request: AnnotationRequest) -> dict[str, Any]:
    _evaluation_question(question_id)
    annotation = request.model_dump()
    annotation["updated_at"] = datetime.now().isoformat(timespec="seconds")
    save_annotation(question_id, annotation)
    return {"id": question_id, "annotation": annotation, "saved": True}


@app.get("/api/search/{task_id}/export/{kind}")
def export_results(task_id: str, kind: Literal["csv", "xlsx"]):
    try:
        task = store.get(task_id)
    except KeyError:
        raise HTTPException(404, "搜索任务不存在") from None
    if not task["results"]:
        raise HTTPException(409, "当前没有可导出的结果")
    frame = pd.DataFrame(task["results"])
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if kind == "csv":
        payload = frame.to_csv(index=False).encode("utf-8-sig")
        media_type = "text/csv; charset=utf-8"
    else:
        stream = io.BytesIO()
        frame.to_excel(stream, index=False, engine="openpyxl")
        payload = stream.getvalue()
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    return StreamingResponse(
        io.BytesIO(payload),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="jobspy-{stamp}.{kind}"'},
    )


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/eval", include_in_schema=False)
def evaluation_page() -> FileResponse:
    return FileResponse(STATIC / "eval.html", headers={"Cache-Control": "no-store"})


app.mount("/static", StaticFiles(directory=STATIC), name="static")
