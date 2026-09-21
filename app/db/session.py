from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from app.config import PROJECT_ROOT, settings


def build_engine(database_url: str) -> Engine:
    options = {"pool_pre_ping": True}
    if database_url.startswith("sqlite"):
        if database_url.startswith("sqlite:///./"):
            relative = database_url.removeprefix("sqlite:///./")
            path = PROJECT_ROOT / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            database_url = f"sqlite:///{path}"
        options["connect_args"] = {"check_same_thread": False}
    db_engine = create_engine(database_url, **options)
    if database_url.startswith("sqlite"):
        @event.listens_for(db_engine, "connect")
        def set_sqlite_pragma(dbapi_connection, _connection_record) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
    return db_engine


engine = build_engine(settings.database_url)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def initialize_database() -> None:
    from app.db.models import Base

    if engine.dialect.name == "postgresql":
        with engine.begin() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(bind=engine)
