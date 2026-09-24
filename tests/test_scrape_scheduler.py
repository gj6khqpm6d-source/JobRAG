from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import Base
from app.scrape_scheduler import (
    DEFAULT_SCHEDULE,
    _claim_run,
    _finish_run,
    get_schedule,
    latest_run,
    next_run_time,
    save_schedule,
)


def schedule_sessions():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def test_schedule_settings_persist_and_can_be_paused():
    factory = schedule_sessions()

    config = get_schedule(factory)
    assert config["enabled"] is True
    assert config["run_hour"] == 9
    assert config["timezone"] == "Asia/Singapore"
    assert config["results_per_site"] == 10
    assert config["max_index_chunks"] == 100

    paused = save_schedule(factory, {**config, "enabled": False})
    assert paused["enabled"] is False
    assert get_schedule(factory)["enabled"] is False
    assert next_run_time(factory, False) is None


def test_scheduled_run_is_unique_and_persists_summary():
    factory = schedule_sessions()
    config = get_schedule(factory)
    scheduled_for = datetime.fromisoformat("2026-09-25T09:00:00+08:00")

    run_id = _claim_run(factory, scheduled_for, config)
    assert run_id is not None
    assert _claim_run(factory, scheduled_for, config) is None
    assert latest_run(factory)["status"] == "running"

    _finish_run(
        factory,
        run_id,
        {
            "status": "completed",
            "results_found": 8,
            "inserted": 2,
            "updated": 1,
            "unchanged": 5,
            "chunks_indexed": 9,
            "chunks_pending": 0,
        },
        None,
    )
    latest = latest_run(factory)
    assert latest["status"] == "completed"
    assert latest["inserted"] == 2
    assert latest["chunks_indexed"] == 9
    assert latest["settings"]["run_hour"] == DEFAULT_SCHEDULE["run_hour"]
