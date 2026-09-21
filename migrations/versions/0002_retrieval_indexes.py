"""add PostgreSQL hybrid retrieval indexes

Revision ID: 0002_retrieval_indexes
Revises: 0001_initial
"""

from alembic import op


revision = "0002_retrieval_indexes"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_job_chunks_embedding_hnsw "
        "ON job_chunks USING hnsw (embedding vector_cosine_ops) "
        "WHERE embedding IS NOT NULL"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_job_chunks_content_fts "
        "ON job_chunks USING gin (to_tsvector('english', content))"
    )


def downgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute("DROP INDEX IF EXISTS ix_job_chunks_content_fts")
    op.execute("DROP INDEX IF EXISTS ix_job_chunks_embedding_hnsw")
