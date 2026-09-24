"""Persist the daily scrape configuration and run history.

Revision ID: 0004_daily_scrape
Revises: 0003_rag_query_cache
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect


revision = "0004_daily_scrape"
down_revision = "0003_rag_query_cache"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if not inspector.has_table("auto_scrape_schedule"):
        op.create_table(
            "auto_scrape_schedule",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("enabled", sa.Boolean(), nullable=False),
            sa.Column("sites", sa.JSON(), nullable=False),
            sa.Column("search_term", sa.String(length=255), nullable=False),
            sa.Column("location", sa.String(length=255), nullable=False),
            sa.Column("job_type", sa.String(length=32), nullable=False),
            sa.Column("results_per_site", sa.Integer(), nullable=False),
            sa.Column("lookback_hours", sa.Integer(), nullable=False),
            sa.Column("max_index_chunks", sa.Integer(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
    if not inspect(bind).has_table("auto_scrape_runs"):
        op.create_table(
            "auto_scrape_runs",
            sa.Column("id", sa.String(length=32), nullable=False),
            sa.Column("scheduled_for", sa.String(length=64), nullable=False),
            sa.Column("status", sa.String(length=24), nullable=False),
            sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("settings_snapshot", sa.JSON(), nullable=False),
            sa.Column("results_found", sa.Integer(), nullable=False),
            sa.Column("inserted", sa.Integer(), nullable=False),
            sa.Column("updated", sa.Integer(), nullable=False),
            sa.Column("unchanged", sa.Integer(), nullable=False),
            sa.Column("chunks_indexed", sa.Integer(), nullable=False),
            sa.Column("chunks_pending", sa.Integer(), nullable=False),
            sa.Column("error_summary", sa.Text(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("scheduled_for"),
        )
        op.create_index("ix_auto_scrape_runs_scheduled_for", "auto_scrape_runs", ["scheduled_for"])
        op.create_index("ix_auto_scrape_runs_status", "auto_scrape_runs", ["status"])


def downgrade() -> None:
    bind = op.get_bind()
    if inspect(bind).has_table("auto_scrape_runs"):
        op.drop_index("ix_auto_scrape_runs_status", table_name="auto_scrape_runs")
        op.drop_index("ix_auto_scrape_runs_scheduled_for", table_name="auto_scrape_runs")
        op.drop_table("auto_scrape_runs")
    if inspect(bind).has_table("auto_scrape_schedule"):
        op.drop_table("auto_scrape_schedule")
