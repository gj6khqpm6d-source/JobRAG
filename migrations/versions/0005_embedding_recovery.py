"""Track embedding failures for bounded automatic recovery.

Revision ID: 0005_embedding_recovery
Revises: 0004_daily_scrape
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect


revision = "0005_embedding_recovery"
down_revision = "0004_daily_scrape"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = inspect(op.get_bind())
    if not inspector.has_table("job_chunks"):
        return
    columns = {column["name"] for column in inspector.get_columns("job_chunks")}
    if "embedding_attempts" not in columns:
        op.add_column("job_chunks", sa.Column("embedding_attempts", sa.Integer(), nullable=False, server_default="0"))
    if "embedding_failed_at" not in columns:
        op.add_column("job_chunks", sa.Column("embedding_failed_at", sa.DateTime(timezone=True), nullable=True))
    if "embedding_error" not in columns:
        op.add_column("job_chunks", sa.Column("embedding_error", sa.Text(), nullable=True))


def downgrade() -> None:
    inspector = inspect(op.get_bind())
    if not inspector.has_table("job_chunks"):
        return
    columns = {column["name"] for column in inspector.get_columns("job_chunks")}
    for name in ("embedding_error", "embedding_failed_at", "embedding_attempts"):
        if name in columns:
            op.drop_column("job_chunks", name)
