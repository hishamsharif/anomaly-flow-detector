from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import and_, create_engine, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from api.db.models import Alert, ModelVersion


# ── Sync helpers for training threads ─────────────────────────────────────────

def _sync_engine(db_url: str):
    return create_engine(db_url.replace("sqlite+aiosqlite", "sqlite"))


def sync_update_model_version(db_url: str, version_id: int, **kwargs) -> None:
    engine = _sync_engine(db_url)
    with Session(engine) as session:
        session.execute(update(ModelVersion).where(ModelVersion.id == version_id).values(**kwargs))
        session.commit()
    engine.dispose()


def sync_get_confirmed_benign_features(db_url: str) -> list[dict] | None:
    """Return feature dicts for benign-labelled flows in the HITL replay buffer.

    Included:
    - `confirmed` + `consensus=SAFE`  → analyst agreed the system was correct (true negative)
    - `override`                      → analyst corrected a false-positive attack verdict to benign
    """
    engine = _sync_engine(db_url)
    with Session(engine) as session:
        rows = session.execute(
            select(Alert.features_json).where(
                or_(
                    and_(Alert.analyst_verdict == "confirmed", Alert.consensus == "SAFE"),
                    Alert.analyst_verdict == "override",
                ),
                Alert.features_json.is_not(None),
            )
        ).scalars().all()
    engine.dispose()
    result = []
    for fj in rows:
        try:
            result.append(json.loads(fj))
        except (json.JSONDecodeError, TypeError):
            pass
    return result if result else None


async def async_get_confirmed_benign_features(db: AsyncSession) -> list[dict] | None:
    """Async counterpart — called from the training router before dispatching the fine-tune thread."""
    rows = (await db.execute(
        select(Alert.features_json).where(
            or_(
                and_(Alert.analyst_verdict == "confirmed", Alert.consensus == "SAFE"),
                Alert.analyst_verdict == "override",
            ),
            Alert.features_json.is_not(None),
        )
    )).scalars().all()
    result = []
    for fj in rows:
        try:
            result.append(json.loads(fj))
        except (json.JSONDecodeError, TypeError):
            pass
    return result if result else None


def sync_set_active(db_url: str, version_id: int) -> None:
    """Archive existing active, promote version_id to active."""
    engine = _sync_engine(db_url)
    with Session(engine) as session:
        session.execute(update(ModelVersion).where(ModelVersion.status == "active").values(status="archived"))
        session.execute(update(ModelVersion).where(ModelVersion.id == version_id).values(status="active"))
        session.commit()
    engine.dispose()


async def get_model_versions(db: AsyncSession) -> list[ModelVersion]:
    result = await db.execute(
        select(ModelVersion).order_by(ModelVersion.created_at.desc())
    )
    return list(result.scalars().all())


async def get_active_model(db: AsyncSession) -> ModelVersion | None:
    result = await db.execute(
        select(ModelVersion).where(ModelVersion.status == "active")
    )
    return result.scalar_one_or_none()


async def mark_stale_training_as_failed(db: AsyncSession) -> None:
    """On startup: any version stuck in 'training' means the process crashed mid-run (ADR-07)."""
    await db.execute(
        update(ModelVersion)
        .where(ModelVersion.status == "training")
        .values(status="failed")
    )
    await db.commit()


# ── Alert CRUD ─────────────────────────────────────────────────────────────────

async def bulk_insert_alerts(db: AsyncSession, alert_dicts: list[dict]) -> list[Alert]:
    alerts = [Alert(**d) for d in alert_dicts]
    db.add_all(alerts)
    await db.commit()
    for a in alerts:
        await db.refresh(a)
    return alerts


def sync_insert_alerts(db_url: str, alert_dicts: list[dict]) -> list[int]:
    """Sync helper for background predict threads — returns inserted IDs."""
    engine = _sync_engine(db_url)
    with Session(engine) as session:
        alerts = [Alert(**d) for d in alert_dicts]
        session.add_all(alerts)
        session.flush()   # auto-increment IDs assigned here, before commit
        ids = [a.id for a in alerts]
        session.commit()
    engine.dispose()
    return ids


async def get_alerts_paginated(
    db: AsyncSession,
    page: int,
    size: int,
    consensus: str | None = None,
    verdict: str | None = None,
) -> tuple[list[Alert], int]:
    base = select(Alert)
    count_base = select(func.count()).select_from(Alert)

    if consensus is not None:
        base = base.where(Alert.consensus == consensus)
        count_base = count_base.where(Alert.consensus == consensus)
    if verdict is not None:
        if verdict == "null":
            base = base.where(Alert.analyst_verdict.is_(None))
            count_base = count_base.where(Alert.analyst_verdict.is_(None))
        else:
            base = base.where(Alert.analyst_verdict == verdict)
            count_base = count_base.where(Alert.analyst_verdict == verdict)

    total_result = await db.execute(count_base)
    total = total_result.scalar_one()

    paged = base.order_by(Alert.timestamp.desc()).offset((page - 1) * size).limit(size)
    result = await db.execute(paged)
    return list(result.scalars().all()), total


async def get_alert_by_id(db: AsyncSession, alert_id: int) -> Alert | None:
    result = await db.execute(select(Alert).where(Alert.id == alert_id))
    return result.scalar_one_or_none()


async def get_eligible_verdict_count(db: AsyncSession) -> int:
    result = await db.execute(
        select(func.count()).select_from(Alert).where(
            Alert.analyst_verdict.in_(["confirmed", "override"])
        )
    )
    return result.scalar_one()


async def get_model_version_by_id(db: AsyncSession, version_id: int) -> ModelVersion | None:
    result = await db.execute(select(ModelVersion).where(ModelVersion.id == version_id))
    return result.scalar_one_or_none()


async def async_set_active(db: AsyncSession, version_id: int) -> None:
    """Archive existing active version, then promote version_id to active."""
    await db.execute(update(ModelVersion).where(ModelVersion.status == "active").values(status="archived"))
    await db.execute(update(ModelVersion).where(ModelVersion.id == version_id).values(status="active"))
    await db.commit()


async def get_dashboard_summary(db: AsyncSession, threshold: int) -> dict:
    from datetime import timedelta, timezone
    since = datetime.now(timezone.utc) - timedelta(hours=24)

    # 24-hour windowed consensus counts (alerts inserted in the last 24 h)
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

    # Severity breakdown — all-time; distribution is meaningful as a cumulative view
    severity_rows = (await db.execute(
        select(Alert.severity, func.count()).group_by(Alert.severity)
    )).all()
    severity_map = {r[0]: r[1] for r in severity_rows}

    # Retrain eligibility — all-time cumulative (threshold is a running total)
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
