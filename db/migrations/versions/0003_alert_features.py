"""add features_json to alerts

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-12
"""
import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("alerts", sa.Column("features_json", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("alerts", "features_json")
