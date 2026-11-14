"""Data Access Object — Dashboard summary query."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Alert


async def get_dashboard_summary(db: AsyncSession, threshold: int) -> dict:
    since = datetime.now(UTC) - timedelta(hours=24)

    # 24-hour windowed consensus counts
    consensus_24h = (await db.execute(
        select(Alert.consensus, func.count())
        .where(Alert.timestamp >= since)
        .group_by(Alert.consensus)
    )).all()
    consensus_map = {r[0]: r[1] for r in consensus_24h}
    total_24h: int = sum(consensus_map.values())

    pending_24h = (await db.execute(
        select(func.count()).select_from(Alert).where(
            Alert.timestamp >= since,
            Alert.analyst_verdict.is_(None),
        )
    )).scalar_one()

    # Severity breakdown — all-time cumulative view
    severity_rows = (await db.execute(
        select(Alert.severity, func.count()).group_by(Alert.severity)
    )).all()
    severity_map = {r[0]: r[1] for r in severity_rows}

    # Retrain eligibility — all-time running total
    eligible = (await db.execute(
        select(func.count()).select_from(Alert).where(
            Alert.analyst_verdict.in_(["confirmed", "override"])
        )
    )).scalar_one()

    return {
        "total_24h": total_24h,
        "safe_24h": consensus_map.get("SAFE", 0),
        "known_attack_24h": consensus_map.get("KNOWN_ATTACK", 0),
        "novel_attack_24h": consensus_map.get("NOVEL_ATTACK", 0),
        "pending_verdict_24h": pending_24h,
        "by_severity": {
            "low": severity_map.get("low", 0),
            "medium": severity_map.get("medium", 0),
            "high": severity_map.get("high", 0),
            "critical": severity_map.get("critical", 0),
        },
        "verdict_eligible": eligible,
        "verdict_threshold": threshold,
        "retrain_unlocked": eligible >= threshold,
    }
