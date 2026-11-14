"""create model_versions table

Revision ID: 0001
Revises:
Create Date: 2026-11-02
"""
import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "model_versions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("classifier_type", sa.String(64), nullable=False),
        sa.Column("detector_type", sa.String(64), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("path", sa.String(512), nullable=True),
        sa.Column("metrics_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    # I1 invariant: at most one ModelVersion can have status = 'active'
    op.execute(
        "CREATE UNIQUE INDEX uq_model_versions_one_active "
        "ON model_versions (status) WHERE status = 'active'"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_model_versions_one_active")
    op.drop_table("model_versions")
