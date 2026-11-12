"""add features_json to alerts

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-12
"""
from alembic import op
import sqlalchemy as sa

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("alerts", sa.Column("features_json", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("alerts", "features_json")
