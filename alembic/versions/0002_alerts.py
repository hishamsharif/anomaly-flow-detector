"""create alerts table

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-12
"""
from alembic import op
import sqlalchemy as sa

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "alerts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("flow_id", sa.String(64), nullable=False),
        # I2: consensus and ae_* fields are append-only; only analyst_verdict / feedback_at updatable
        sa.Column("consensus", sa.String(32), nullable=False),
        sa.Column("severity", sa.String(32), nullable=False),
        sa.Column("predicted_label", sa.Integer(), nullable=True),
        sa.Column("classifier_confidence", sa.Float(), nullable=True),
        sa.Column("ae_score", sa.Float(), nullable=False),
        sa.Column("ae_threshold", sa.Float(), nullable=False),
        sa.Column("top_features_json", sa.Text(), nullable=True),
        sa.Column("analyst_verdict", sa.String(32), nullable=True),
        sa.Column("feedback_at", sa.DateTime(), nullable=True),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_alerts_consensus", "alerts", ["consensus"])
    op.create_index("ix_alerts_analyst_verdict", "alerts", ["analyst_verdict"])
    op.create_index("ix_alerts_timestamp", "alerts", ["timestamp"])


def downgrade() -> None:
    op.drop_index("ix_alerts_timestamp", "alerts")
    op.drop_index("ix_alerts_analyst_verdict", "alerts")
    op.drop_index("ix_alerts_consensus", "alerts")
    op.drop_table("alerts")
