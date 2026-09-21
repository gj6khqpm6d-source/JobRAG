"""add versioned RAG query cache

Revision ID: 0003_rag_query_cache
Revises: 0002_retrieval_indexes
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect


revision = "0003_rag_query_cache"
down_revision = "0002_retrieval_indexes"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if inspect(bind).has_table("rag_query_cache"):
        return
    op.create_table(
        "rag_query_cache",
        sa.Column("cache_key", sa.String(length=64), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("corpus_version", sa.String(length=255), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("cache_key"),
    )
    op.create_index("ix_rag_query_cache_kind", "rag_query_cache", ["kind"])
    op.create_index("ix_rag_query_cache_corpus_version", "rag_query_cache", ["corpus_version"])
    op.create_index("ix_rag_query_cache_expires_at", "rag_query_cache", ["expires_at"])


def downgrade() -> None:
    bind = op.get_bind()
    if not inspect(bind).has_table("rag_query_cache"):
        return
    op.drop_index("ix_rag_query_cache_expires_at", table_name="rag_query_cache")
    op.drop_index("ix_rag_query_cache_corpus_version", table_name="rag_query_cache")
    op.drop_index("ix_rag_query_cache_kind", table_name="rag_query_cache")
    op.drop_table("rag_query_cache")
