from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.orm import sessionmaker

from app.db.models import Job
from app.db.repository import upsert_jobs
from app.rag.indexing import prepare_chunks


@dataclass
class BackfillResult:
    candidates: int = 0
    descriptions_added: int = 0
    failed: int = 0
    chunks_created: int = 0
    job_ids_to_sync: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, int]:
        values = asdict(self)
        values.pop("job_ids_to_sync")
        return values


def _linkedin_job_id(job: Job) -> str | None:
    source_id = job.source_job_id or ""
    match = re.search(r"(?:li-)?(\d+)$", source_id)
    if not match:
        match = re.search(r"/jobs/view/(?:[^/?#]*-)?(\d+)", job.job_url or "")
    return match.group(1) if match else None


def _default_fetcher() -> Callable[[str], dict[str, Any]]:
    from jobspy.linkedin import LinkedIn
    from jobspy.model import DescriptionFormat, ScraperInput, Site

    scraper = LinkedIn()
    scraper.scraper_input = ScraperInput(
        site_type=[Site.LINKEDIN],
        description_format=DescriptionFormat.MARKDOWN,
        linkedin_fetch_description=True,
    )
    return scraper._get_job_details


def backfill_linkedin_descriptions(
    session_factory: sessionmaker,
    *,
    limit: int = 25,
    fetch_details: Callable[[str], dict[str, Any]] | None = None,
    retries: int = 2,
) -> BackfillResult:
    result = BackfillResult()
    loader = fetch_details or _default_fetcher()
    with session_factory() as session:
        jobs = session.scalars(
            select(Job)
            .where(
                Job.source == "linkedin",
                or_(Job.description.is_(None), func.length(func.trim(Job.description)) == 0),
            )
            .order_by(Job.last_seen_at.desc())
            .limit(limit)
        ).all()
        records = []
        for job in jobs:
            job_id = _linkedin_job_id(job)
            if not job_id:
                result.failed += 1
                continue
            result.candidates += 1
            details: dict[str, Any] = {}
            for attempt in range(retries):
                details = loader(job_id) or {}
                if details.get("description"):
                    break
                if attempt + 1 < retries:
                    time.sleep(0.5)
            if not details.get("description"):
                result.failed += 1
                continue
            record = dict(job.raw_data or {})
            record.update(
                {
                    "site": job.source,
                    "id": job.source_job_id,
                    "title": job.title,
                    "company": job.company,
                    "location": job.location,
                    "job_url": job.job_url,
                    "job_url_direct": details.get("job_url_direct") or job.job_url_direct,
                    "description": details["description"],
                    "date_posted": job.date_posted.isoformat() if job.date_posted else None,
                }
            )
            records.append(record)

    if records:
        ingestion = upsert_jobs(session_factory, records)
        result.descriptions_added = ingestion.updated
        result.job_ids_to_sync = ingestion.job_ids_to_sync
        result.chunks_created = prepare_chunks(
            session_factory,
            job_ids=ingestion.job_ids_to_sync,
        ).chunks_created
    return result
