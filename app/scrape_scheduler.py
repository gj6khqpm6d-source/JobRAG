from __future__ import annotations

import threading
import uuid
from datetime import datetime, time, timedelta, timezone
from typing import Any, Callable
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from app.db.models import AutoScrapeRun, AutoScrapeSchedule


SCHEDULE_TIMEZONE = ZoneInfo("Asia/Singapore")
DEFAULT_SCHEDULE = {
    "enabled": True,
    "sites": ["linkedin"],
    "search_term": "AI Agent",
    "location": "Singapore",
    "job_type": "internship",
    "results_per_site": 10,
    "lookback_hours": 72,
    "max_index_chunks": 100,
    "run_hour": 9,
    "run_minute": 0,
    "timezone": "Asia/Singapore",
}


def _schedule_payload(row: AutoScrapeSchedule) -> dict[str, Any]:
    return {
        **DEFAULT_SCHEDULE,
        "enabled": row.enabled,
        "sites": list(row.sites or []),
        "search_term": row.search_term,
        "location": row.location,
        "job_type": row.job_type,
        "results_per_site": row.results_per_site,
        "lookback_hours": row.lookback_hours,
        "max_index_chunks": row.max_index_chunks,
        "updated_at": row.updated_at.isoformat(),
    }


def get_schedule(session_factory: sessionmaker, *, create: bool = True) -> dict[str, Any]:
    with session_factory() as session:
        row = session.get(AutoScrapeSchedule, 1)
        if row is not None:
            return _schedule_payload(row)
        if not create:
            return dict(DEFAULT_SCHEDULE)
        row = AutoScrapeSchedule(id=1, **{key: DEFAULT_SCHEDULE[key] for key in (
            "enabled", "sites", "search_term", "location", "job_type",
            "results_per_site", "lookback_hours", "max_index_chunks",
        )})
        session.add(row)
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            row = session.get(AutoScrapeSchedule, 1)
            if row is None:
                raise
        return _schedule_payload(row)


def save_schedule(session_factory: sessionmaker, values: dict[str, Any]) -> dict[str, Any]:
    get_schedule(session_factory)
    with session_factory() as session:
        row = session.get(AutoScrapeSchedule, 1)
        if row is None:
            raise RuntimeError("自动抓取配置初始化失败")
        for key in (
            "enabled", "sites", "search_term", "location", "job_type",
            "results_per_site", "lookback_hours", "max_index_chunks",
        ):
            if key in values:
                setattr(row, key, values[key])
        row.updated_at = datetime.now(timezone.utc)
        session.commit()
        return _schedule_payload(row)


def latest_run(session_factory: sessionmaker) -> dict[str, Any] | None:
    with session_factory() as session:
        row = session.scalar(select(AutoScrapeRun).order_by(AutoScrapeRun.started_at.desc()).limit(1))
        if row is None:
            return None
        return {
            "id": row.id,
            "scheduled_for": row.scheduled_for,
            "status": row.status,
            "started_at": row.started_at.isoformat(),
            "finished_at": row.finished_at.isoformat() if row.finished_at else None,
            "settings": row.settings_snapshot,
            "results_found": row.results_found,
            "inserted": row.inserted,
            "updated": row.updated,
            "unchanged": row.unchanged,
            "chunks_indexed": row.chunks_indexed,
            "chunks_pending": row.chunks_pending,
            "error_summary": row.error_summary,
        }


def next_run_time(session_factory: sessionmaker, enabled: bool) -> datetime | None:
    if not enabled:
        return None
    now = datetime.now(SCHEDULE_TIMEZONE)
    next_run = datetime.combine(now.date(), time(9, 0), tzinfo=SCHEDULE_TIMEZONE)
    if now >= next_run:
        scheduled_for = next_run.isoformat(timespec="minutes")
        with session_factory() as session:
            already_ran = session.scalar(
                select(AutoScrapeRun.id).where(AutoScrapeRun.scheduled_for == scheduled_for)
            )
        if already_ran:
            next_run += timedelta(days=1)
    return next_run


def _claim_run(session_factory: sessionmaker, scheduled_for: datetime, config: dict[str, Any]) -> str | None:
    run_id = uuid.uuid4().hex
    with session_factory() as session:
        session.add(
            AutoScrapeRun(
                id=run_id,
                scheduled_for=scheduled_for.isoformat(timespec="minutes"),
                status="running",
                settings_snapshot={key: config[key] for key in DEFAULT_SCHEDULE},
            )
        )
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            return None
    return run_id


def _finish_run(session_factory: sessionmaker, run_id: str, result: dict[str, Any] | None, error: str | None) -> None:
    with session_factory() as session:
        row = session.get(AutoScrapeRun, run_id)
        if row is None:
            return
        row.finished_at = datetime.now(timezone.utc)
        if error:
            row.status = "failed"
            row.error_summary = error[:2000]
            session.commit()
            return
        outcome = result or {}
        row.status = outcome.get("status", "completed")
        row.results_found = int(outcome.get("results_found", 0))
        row.inserted = int(outcome.get("inserted", 0))
        row.updated = int(outcome.get("updated", 0))
        row.unchanged = int(outcome.get("unchanged", 0))
        row.chunks_indexed = int(outcome.get("chunks_indexed", 0))
        row.chunks_pending = int(outcome.get("chunks_pending", 0))
        row.error_summary = str(outcome.get("error_summary") or "")[:2000] or None
        session.commit()


def _recover_stale_runs(session_factory: sessionmaker) -> None:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=2)
    with session_factory() as session:
        stale = session.scalars(
            select(AutoScrapeRun).where(AutoScrapeRun.status == "running", AutoScrapeRun.started_at < cutoff)
        ).all()
        for row in stale:
            row.status = "failed"
            row.finished_at = datetime.now(timezone.utc)
            row.error_summary = "服务在调度任务完成前退出；数据入库前需检查任务记录。"
        if stale:
            session.commit()


class DailyScrapeScheduler:
    def __init__(
        self,
        session_factory: sessionmaker,
        run_callback: Callable[[dict[str, Any], datetime], dict[str, Any]],
    ) -> None:
        self.session_factory = session_factory
        self.run_callback = run_callback
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        get_schedule(self.session_factory)
        _recover_stale_runs(self.session_factory)
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="jobrag-daily-scrape", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=5)

    def notify(self) -> None:
        self._wake.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                config = get_schedule(self.session_factory)
                if not config["enabled"]:
                    self._wait(60)
                    continue
                now = datetime.now(SCHEDULE_TIMEZONE)
                scheduled_for = datetime.combine(now.date(), time(9, 0), tzinfo=SCHEDULE_TIMEZONE)
                if now < scheduled_for:
                    self._wait(min(max((scheduled_for - now).total_seconds(), 1), 60))
                    continue
                run_id = _claim_run(self.session_factory, scheduled_for, config)
                if run_id is None:
                    tomorrow = scheduled_for + timedelta(days=1)
                    self._wait(max((tomorrow - datetime.now(SCHEDULE_TIMEZONE)).total_seconds(), 60))
                    continue
                try:
                    result = self.run_callback(config, scheduled_for)
                    _finish_run(self.session_factory, run_id, result, None)
                except Exception as exc:
                    _finish_run(self.session_factory, run_id, None, f"{type(exc).__name__}: {exc}")
            except Exception:
                self._wait(60)

    def _wait(self, timeout: float) -> bool:
        notified = self._wake.wait(timeout)
        self._wake.clear()
        return notified
