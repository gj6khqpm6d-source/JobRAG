from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date
from typing import Any

from sqlalchemy import func, select, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Job, JobChunk


@dataclass(frozen=True)
class RetrievalFilters:
    source: str = ""
    location: str = ""
    title: str = ""
    job_type: str = ""
    date_after: date | None = None


# The corpus is mostly English job descriptions while users often ask in
# Chinese.  These are deliberately conservative retrieval aliases: they add
# English search terms without replacing the original query, and do not alter
# the answer text sent to the LLM.
QUERY_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("检索增强生成", ("rag", "retrieval augmented generation")),
    ("检索增强", ("rag", "retrieval augmented generation")),
    ("生成式人工智能", ("generative ai", "genai")),
    ("人工智能", ("ai", "artificial intelligence")),
    ("大模型", ("llm", "large language model")),
    ("智能体", ("agent", "agents")),
    ("机器学习", ("machine learning", "ml")),
    ("深度学习", ("deep learning")),
    ("软件工程", ("software engineering", "software engineer")),
    ("向量数据库", ("vector database", "vector databases")),
    ("数据库", ("database", "databases")),
    ("后端", ("backend", "back end")),
    ("前端", ("frontend", "front end")),
    ("实习生", ("intern", "internship", "internships")),
    ("实习", ("intern", "internship", "internships")),
    ("岗位要求", ("requirements", "qualifications")),
    ("任职要求", ("requirements", "qualifications")),
    ("工作要求", ("requirements", "qualifications")),
    ("技能", ("skills", "technologies", "technical stack")),
    ("技术栈", ("skills", "technologies", "technical stack")),
    ("职责", ("responsibilities", "duties")),
    ("经验", ("experience")),
    ("远程", ("remote")),
)

ANALYSIS_PHRASES = (
    re.compile(r"最常见|通常|一般|主要|出现最多|占比|比例|多少|统计|分析"),
    re.compile(r"哪些技能|什么技能|技术要求|岗位要求|任职要求|工作要求|有哪些要求|有什么要求|工作职责|岗位职责"),
    re.compile(r"岗位|技能|要求|是什么|有哪些|有什么|请问"),
    re.compile(r"\bmost common\b|\bhow many\b|\bpercentage\b|\bproportion\b|\bdistribution\b", re.I),
    re.compile(r"\bwhat skills?\b|\brequirements?\b|\bqualifications?\b|\bresponsibilities\b", re.I),
)

GENERIC_RANKING_TERMS = {
    "skill",
    "skills",
    "technology",
    "technologies",
    "technical stack",
    "requirements",
    "qualifications",
    "responsibilities",
    "duties",
    "岗位",
    "技能",
    "要求",
    "哪些",
    "什么",
    "常见",
    "职责",
}

# Development-set tuning showed that title/concept boosts duplicated signals
# already present in BM25 and BGE, allowing keyword-heavy near-duplicates to
# crowd out semantically relevant jobs. Keep the final job score focused on
# retrieval evidence and useful section coverage.
JOB_RANKING_WEIGHTS = {
    "retrieval_relevance": 0.80,
    "context_completeness": 0.20,
}
UNREQUESTED_PHD_PENALTY = 0.08


def query_terms(query: str) -> list[str]:
    """Return safe lexical terms for FTS/BM25 and the SQLite fallback.

    Alias terms are added alongside the original alphanumeric terms.  CJK
    queries are represented as short character n-grams so a Chinese question
    can still match Chinese metadata or descriptions without a third-party
    tokenizer.
    """
    lowered = query.casefold()
    terms: set[str] = set(re.findall(r"[a-z0-9][a-z0-9+#.-]*", lowered))
    for source, aliases in QUERY_ALIASES:
        if source in query:
            terms.update(alias.casefold() for alias in aliases)
    cjk = "".join(re.findall(r"[\u3400-\u9fff]+", query))
    for index in range(max(0, len(cjk) - 1)):
        terms.add(cjk[index : index + 2])
    return sorted(term for term in terms if term)


def scope_query(query: str, filters: RetrievalFilters = RetrievalFilters()) -> str:
    """Remove analysis wording so retrieval focuses on the target role."""
    cleaned = query.strip()
    for pattern in ANALYSIS_PHRASES:
        cleaned = pattern.sub(" ", cleaned)
    cleaned = re.sub(r"(?:^|\s)的(?=\s|$)", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ，。！？,.?：:")
    if filters.title and filters.title.casefold() not in cleaned.casefold():
        cleaned = f"{filters.title} {cleaned}".strip()
    return cleaned or query.strip()


def _ranking_terms(query: str) -> list[str]:
    return [
        term
        for term in query_terms(query)
        if term not in GENERIC_RANKING_TERMS and (len(term) >= 2 or term in {"ai", "ml"})
    ]


def _contains_term(text_value: str, term: str) -> bool:
    lowered = text_value.casefold()
    normalized = term.casefold()
    if re.fullmatch(r"[a-z0-9+#.-]+", normalized):
        return re.search(rf"(?<![a-z0-9]){re.escape(normalized)}(?![a-z0-9])", lowered) is not None
    return normalized in lowered


def _title_family(title: str, company: str | None) -> str:
    """Collapse repost variants without merging genuinely different roles."""
    normalized = title.casefold()
    normalized = re.sub(r"\(\s*ph\.?d\.?\s*\)", " ", normalized)
    normalized = re.sub(r"[-–—]\s*20\d{2}\s*start.*$", " ", normalized)
    normalized = re.sub(r"\b20\d{2}\s*start\b", " ", normalized)
    normalized = re.sub(r"[\W_]+", " ", normalized, flags=re.UNICODE)
    return f"{(company or '').casefold()}|{' '.join(normalized.split())}"


SQLITE_FTS_TABLE = "job_chunks_fts"


def _fts_signature(session: Session) -> tuple[int, str, str]:
    return (
        int(session.scalar(select(func.count()).select_from(JobChunk)) or 0),
        str(session.scalar(select(func.min(JobChunk.content_hash))) or ""),
        str(session.scalar(select(func.max(JobChunk.content_hash))) or ""),
    )


def rebuild_sqlite_keyword_index(session: Session) -> bool:
    """Create/rebuild the small local FTS5 index used by SQLite retrieval.

    SQLite is the development/default deployment, so keeping this index local
    avoids requiring a separate search service.  PostgreSQL continues to use
    its native GIN full-text index.
    """
    if session.bind is None or session.bind.dialect.name != "sqlite":
        return False
    try:
        session.execute(
            text(
                f"""CREATE VIRTUAL TABLE IF NOT EXISTS {SQLITE_FTS_TABLE} USING fts5(
                    chunk_id UNINDEXED,
                    job_id UNINDEXED,
                    title,
                    company,
                    location,
                    section_type,
                    content,
                    tokenize='unicode61'
                )"""
            )
        )
        session.execute(text(f"DELETE FROM {SQLITE_FTS_TABLE}"))
        session.execute(
            text(
                f"""INSERT INTO {SQLITE_FTS_TABLE}
                    (chunk_id, job_id, title, company, location, section_type, content)
                SELECT CAST(c.id AS TEXT), c.job_id, j.title, COALESCE(j.company, ''),
                       COALESCE(j.location, ''), c.section_type, c.content
                FROM job_chunks AS c
                JOIN jobs AS j ON j.id = c.job_id"""
            )
        )
        session.execute(
            text(
                """CREATE TABLE IF NOT EXISTS job_chunks_fts_meta (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    chunk_count INTEGER NOT NULL,
                    min_content_hash TEXT NOT NULL,
                    max_content_hash TEXT NOT NULL
                )"""
            )
        )
        count, minimum, maximum = _fts_signature(session)
        session.execute(text("DELETE FROM job_chunks_fts_meta"))
        session.execute(
            text(
                """INSERT INTO job_chunks_fts_meta
                    (id, chunk_count, min_content_hash, max_content_hash)
                VALUES (1, :chunk_count, :min_content_hash, :max_content_hash)"""
            ),
            {
                "chunk_count": count,
                "min_content_hash": minimum,
                "max_content_hash": maximum,
            },
        )
        return True
    except OperationalError:
        # Some minimal SQLite builds omit FTS5.  The caller will use the
        # deterministic Python lexical fallback in that case.
        session.rollback()
        return False


def _ensure_sqlite_keyword_index(session: Session) -> bool:
    if session.bind is None or session.bind.dialect.name != "sqlite":
        return False
    try:
        session.execute(
            text(
                """CREATE VIRTUAL TABLE IF NOT EXISTS job_chunks_fts USING fts5(
                    chunk_id UNINDEXED, job_id UNINDEXED, title, company,
                    location, section_type, content, tokenize='unicode61'
                )"""
            )
        )
        session.execute(
            text(
                """CREATE TABLE IF NOT EXISTS job_chunks_fts_meta (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    chunk_count INTEGER NOT NULL,
                    min_content_hash TEXT NOT NULL,
                    max_content_hash TEXT NOT NULL
                )"""
            )
        )
        expected = _fts_signature(session)
        meta = session.execute(
            text(
                """SELECT chunk_count, min_content_hash, max_content_hash
                FROM job_chunks_fts_meta WHERE id = 1"""
            )
        ).one_or_none()
        if meta is None or tuple(meta) != expected:
            rebuilt = rebuild_sqlite_keyword_index(session)
            if rebuilt:
                # A retrieval session is otherwise read-only, so commit only
                # the one-time index repair; without this, closing the session
                # would roll the FTS5 rebuild back.
                session.commit()
            return rebuilt
        return True
    except OperationalError:
        session.rollback()
        return False


def _conditions(filters: RetrievalFilters, *, require_embedding: bool = True) -> list[Any]:
    conditions: list[Any] = [Job.is_active.is_(True)]
    if require_embedding:
        conditions.append(JobChunk.embedding.is_not(None))
    if filters.source:
        conditions.append(Job.source == filters.source)
    if filters.location:
        conditions.append(Job.location.ilike(f"%{filters.location}%"))
    if filters.title:
        conditions.append(Job.title.ilike(f"%{filters.title}%"))
    if filters.job_type:
        conditions.append(Job.job_type.ilike(f"%{filters.job_type}%"))
    if filters.date_after:
        conditions.append(Job.date_posted >= filters.date_after)
    return conditions


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _keyword_score(query: str, content: str) -> float:
    terms = set(query_terms(query))
    if not terms:
        return 0.0
    lowered = content.casefold()
    return sum(1 for term in terms if term in lowered) / len(terms)


def _candidate_rows(
    session: Session,
    filters: RetrievalFilters,
    max_candidates: int = 5000,
    *,
    require_embedding: bool = True,
):
    return session.execute(
        select(JobChunk, Job)
        .join(Job, Job.id == JobChunk.job_id)
        .where(*_conditions(filters, require_embedding=require_embedding))
        .limit(max_candidates)
    ).all()


def _sqlite_keyword_rows(
    session: Session,
    *,
    query: str,
    filters: RetrievalFilters,
    candidate_k: int,
) -> list[tuple[JobChunk, Job, float]]:
    """Retrieve lexical candidates with FTS5, then add deterministic fallback hits.

    The fallback is important for Chinese text because SQLite's built-in
    unicode61 tokenizer is not a Chinese word segmenter.  English aliases go
    through BM25; native CJK substring matches are appended from the same
    filtered candidate set.
    """
    fts_rows: list[tuple[JobChunk, Job, float]] = []
    terms = query_terms(query)
    fts_terms = [
        f'"{term}"'
        for term in terms
        if re.fullmatch(r"[a-z0-9+#.-]+(?: [a-z0-9+#.-]+)*", term)
    ]
    if fts_terms and _ensure_sqlite_keyword_index(session):
        where = ["j.is_active = 1"]
        params: dict[str, Any] = {
            "match_query": " OR ".join(fts_terms),
            "limit": candidate_k,
        }
        if filters.source:
            where.append("j.source = :source")
            params["source"] = filters.source
        if filters.location:
            where.append("LOWER(j.location) LIKE LOWER(:location)")
            params["location"] = f"%{filters.location}%"
        if filters.title:
            where.append("LOWER(j.title) LIKE LOWER(:title)")
            params["title"] = f"%{filters.title}%"
        if filters.job_type:
            where.append("LOWER(j.job_type) LIKE LOWER(:job_type)")
            params["job_type"] = f"%{filters.job_type}%"
        if filters.date_after:
            where.append("j.date_posted >= :date_after")
            params["date_after"] = filters.date_after.isoformat()
        try:
            raw_rows = session.execute(
                text(
                    """SELECT CAST(f.chunk_id AS INTEGER) AS chunk_id,
                              bm25(job_chunks_fts, 5.0, 3.0, 2.0, 4.0, 1.0) AS rank_score
                       FROM job_chunks_fts AS f
                       JOIN job_chunks AS c ON c.id = CAST(f.chunk_id AS INTEGER)
                       JOIN jobs AS j ON j.id = c.job_id
                      WHERE job_chunks_fts MATCH :match_query
                        AND """
                    + " AND ".join(where)
                    + " ORDER BY rank_score ASC LIMIT :limit"
                ),
                params,
            ).all()
            ids = [int(row.chunk_id) for row in raw_rows]
            if ids:
                objects = session.execute(
                    select(JobChunk, Job)
                    .join(Job, Job.id == JobChunk.job_id)
                    .where(JobChunk.id.in_(ids), Job.is_active.is_(True))
                ).all()
                by_id = {chunk.id: (chunk, job) for chunk, job in objects}
                for row in raw_rows:
                    pair = by_id.get(int(row.chunk_id))
                    if pair:
                        # SQLite BM25 returns lower (usually negative) scores
                        # for better matches; expose a higher-is-better value.
                        fts_rows.append((pair[0], pair[1], float(-row.rank_score)))
        except OperationalError:
            session.rollback()

    # Supplement BM25 with native substring matches.  This also ensures a
    # query containing an unmapped Chinese skill does not silently disappear.
    fallback_rows = _candidate_rows(session, filters, require_embedding=False)
    fallback_scored = [
        (chunk, job, _keyword_score(query, chunk.content))
        for chunk, job in fallback_rows
    ]
    fallback_scored = [row for row in fallback_scored if row[2] > 0]
    fallback_scored.sort(key=lambda row: row[2], reverse=True)
    seen = {chunk.id for chunk, _, _ in fts_rows}
    combined = list(fts_rows)
    combined.extend(row for row in fallback_scored if row[0].id not in seen)
    return combined[:candidate_k]


def hybrid_search(
    session_factory: sessionmaker,
    *,
    query: str,
    query_vector: list[float],
    filters: RetrievalFilters = RetrievalFilters(),
    top_k: int = 8,
    candidate_k: int = 50,
    max_chunks_per_job: int = 2,
    mode: str = "hybrid",
    rrf_k: int = 60,
    vector_weight: float = 1.0,
    keyword_weight: float = 1.0,
) -> list[dict[str, Any]]:
    if mode not in {"hybrid", "vector", "keyword"}:
        raise ValueError(f"unsupported retrieval mode: {mode}")
    if candidate_k < 1 or max_chunks_per_job < 1 or rrf_k < 1:
        raise ValueError("retrieval limits and rrf_k must be positive")
    if vector_weight <= 0 or keyword_weight <= 0:
        raise ValueError("retrieval weights must be positive")
    with session_factory() as session:
        if session.bind.dialect.name == "postgresql":
            similarity = (1 - JobChunk.embedding.cosine_distance(query_vector)).label("similarity")
            vector_rows = session.execute(
                select(JobChunk, Job, similarity)
                .join(Job, Job.id == JobChunk.job_id)
                .where(*_conditions(filters))
                .order_by(JobChunk.embedding.cosine_distance(query_vector))
                .limit(candidate_k)
            ).all()
            keyword_query = " ".join(query_terms(query))
            ts_query = func.websearch_to_tsquery("english", keyword_query or query)
            rank = func.ts_rank_cd(func.to_tsvector("english", JobChunk.content), ts_query).label("rank")
            keyword_rows = session.execute(
                select(JobChunk, Job, rank)
                .join(Job, Job.id == JobChunk.job_id)
                .where(
                    *_conditions(filters, require_embedding=False),
                    func.to_tsvector("english", JobChunk.content).op("@@")(ts_query),
                )
                .order_by(rank.desc())
                .limit(candidate_k)
            ).all()
        else:
            rows = _candidate_rows(session, filters)
            scored = [
                (chunk, job, _cosine(list(chunk.embedding), query_vector), _keyword_score(query, chunk.content))
                for chunk, job in rows
            ]
            vector_rows = [(chunk, job, similarity) for chunk, job, similarity, _ in sorted(scored, key=lambda item: item[2], reverse=True)[:candidate_k]]
            keyword_rows = _sqlite_keyword_rows(
                session,
                query=query,
                filters=filters,
                candidate_k=candidate_k,
            )

        fused: dict[int, dict[str, Any]] = {}
        row_groups = {
            "vector": vector_rows,
            "keyword": keyword_rows,
        }
        selected_types = ("vector", "keyword") if mode == "hybrid" else (mode,)
        weights = {"vector": vector_weight, "keyword": keyword_weight}
        for result_type in selected_types:
            rows = row_groups[result_type]
            for rank_index, (chunk, job, raw_score) in enumerate(rows, start=1):
                item = fused.setdefault(
                    chunk.id,
                    {
                        "chunk": chunk,
                        "job": job,
                        "rrf_score": 0.0,
                        "vector_score": None,
                        "keyword_score": None,
                    },
                )
                item["rrf_score"] += weights[result_type] / (rrf_k + rank_index)
                item[f"{result_type}_score"] = float(raw_score)

        ranked_candidates = sorted(fused.values(), key=lambda item: item["rrf_score"], reverse=True)
        ranked = []
        job_counts: dict[str, int] = {}
        for item in ranked_candidates:
            job_id = item["job"].id
            if job_counts.get(job_id, 0) >= max_chunks_per_job:
                continue
            ranked.append(item)
            job_counts[job_id] = job_counts.get(job_id, 0) + 1
            if len(ranked) >= top_k:
                break
        return [
            {
                "chunk_id": item["chunk"].id,
                "job_id": item["job"].id,
                "title": item["job"].title,
                "company": item["job"].company,
                "location": item["job"].location,
                "job_type": item["job"].job_type,
                "source": item["job"].source,
                "date_posted": item["job"].date_posted.isoformat() if item["job"].date_posted else None,
                "job_url": item["job"].job_url_direct or item["job"].job_url,
                "section_type": item["chunk"].section_type,
                "content": item["chunk"].content,
                "score": round(item["rrf_score"], 8),
                "vector_score": round(item["vector_score"], 6) if item["vector_score"] is not None else None,
                "keyword_score": round(item["keyword_score"], 6) if item["keyword_score"] is not None else None,
            }
            for item in ranked
        ]


def ranked_job_search(
    session_factory: sessionmaker,
    *,
    query: str,
    query_vector: list[float],
    filters: RetrievalFilters = RetrievalFilters(),
    top_jobs: int = 8,
    candidate_jobs: int = 50,
    max_context_chunks: int = 3,
    mode: str = "hybrid",
) -> dict[str, Any]:
    """Retrieve broadly, aggregate chunks by job, and rank jobs transparently.

    The candidate pool is an internal recall mechanism. The returned jobs are
    the same jobs used for analysis, context, and user-facing citations.
    """
    if top_jobs < 1 or candidate_jobs < 1 or max_context_chunks < 1:
        raise ValueError("job retrieval limits must be positive")
    candidate_jobs = max(candidate_jobs, top_jobs)
    retrieval_query = scope_query(query, filters)
    raw_chunks = hybrid_search(
        session_factory,
        query=retrieval_query,
        query_vector=query_vector,
        filters=filters,
        top_k=max(candidate_jobs * 4, top_jobs * max_context_chunks),
        candidate_k=max(candidate_jobs * 4, 100),
        max_chunks_per_job=4,
        mode=mode,
    )
    if not raw_chunks:
        return {
            "scope_query": retrieval_query,
            "candidate_job_count": 0,
            "analyzed_job_count": 0,
            "jobs": [],
        }

    grouped: dict[str, dict[str, Any]] = {}
    for chunk in raw_chunks:
        job = grouped.setdefault(
            chunk["job_id"],
            {
                "job_id": chunk["job_id"],
                "title": chunk["title"],
                "company": chunk["company"],
                "location": chunk["location"],
                "job_type": chunk["job_type"],
                "source": chunk["source"],
                "date_posted": chunk["date_posted"],
                "job_url": chunk["job_url"],
                "chunks": [],
            },
        )
        job["chunks"].append(chunk)

    terms = _ranking_terms(retrieval_query)
    relevant_sections = {
        "responsibilities",
        "required_qualifications",
        "preferred_qualifications",
        "skills",
        "experience",
        "education",
        "projects_research",
    }
    scored_jobs: list[dict[str, Any]] = []
    semantic_values: list[float] = []
    for job in grouped.values():
        job["chunks"].sort(key=lambda item: item["score"], reverse=True)
        scores = [float(chunk["score"]) for chunk in job["chunks"]]
        semantic_raw = scores[0] + (0.35 * scores[1] if len(scores) > 1 else 0.0)
        job["_semantic_raw"] = semantic_raw
        semantic_values.append(semantic_raw)

    max_semantic = max(semantic_values) if semantic_values else 1.0
    for job in grouped.values():
        title = str(job["title"] or "")
        combined = "\n".join([title, *(str(chunk["content"]) for chunk in job["chunks"])])
        title_hits = sum(_contains_term(title, term) for term in terms)
        concept_hits = sum(_contains_term(combined, term) for term in terms)
        title_match = min(title_hits / 2, 1.0) if terms else 0.0
        concept_coverage = min(concept_hits / 3, 1.0) if terms else 0.0
        sections = {str(chunk["section_type"]) for chunk in job["chunks"]}
        completeness = min(len(sections & relevant_sections) / 4, 1.0)
        semantic = job.pop("_semantic_raw") / max_semantic if max_semantic else 0.0
        specificity_penalty = (
            UNREQUESTED_PHD_PENALTY
            if "phd" not in retrieval_query.casefold() and re.search(r"ph\.?d", title, re.I)
            else 0.0
        )
        rank_score = (
            JOB_RANKING_WEIGHTS["retrieval_relevance"] * semantic
            + JOB_RANKING_WEIGHTS["context_completeness"] * completeness
            - specificity_penalty
        )
        job["rank_score"] = round(rank_score, 6)
        job["ranking_signals"] = {
            "retrieval_relevance": round(semantic, 4),
            "title_match": round(title_match, 4),
            "concept_coverage": round(concept_coverage, 4),
            "context_completeness": round(completeness, 4),
            "specificity_penalty": round(specificity_penalty, 4),
        }
        job["chunks"] = job["chunks"][:max_context_chunks]
        scored_jobs.append(job)

    scored_jobs.sort(
        key=lambda item: (item["rank_score"], item["date_posted"] or "", item["job_id"]),
        reverse=True,
    )
    candidates = scored_jobs[:candidate_jobs]
    best_score = candidates[0]["rank_score"] if candidates else 0.0
    relevant = [job for job in candidates if not best_score or job["rank_score"] >= best_score * 0.35]
    selected: list[dict[str, Any]] = []
    seen_families: set[str] = set()
    for job in relevant:
        family = _title_family(str(job["title"] or ""), job.get("company"))
        if family in seen_families:
            continue
        seen_families.add(family)
        selected.append(job)
        if len(selected) >= top_jobs:
            break
    return {
        "scope_query": retrieval_query,
        "candidate_job_count": len(candidates),
        "analyzed_job_count": len(selected),
        "jobs": selected,
    }
