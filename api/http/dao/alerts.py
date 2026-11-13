"""Data Access Object — Alert queries.

Async helpers use an AsyncSession injected by FastAPI's Depends(get_db).
Sync helpers (prefixed sync_) run in background threads and must not hold
an async session.
"""
from __future__ import annotations

import json

from sqlalchemy import and_, create_engine, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from db.models import Alert


def _sync_engine(db_url: str):
    return create_engine(db_url.replace("sqlite+aiosqlite", "sqlite"))


# ── Sync helpers (background threads) ─────────────────────────────────────────

def sync_insert_alerts(db_url: str, alert_dicts: list[dict]) -> list[int]:
    """Sync helper for background predict threads — returns inserted IDs."""
    engine = _sync_engine(db_url)
    with Session(engine) as session:
        alerts = [Alert(**d) for d in alert_dicts]
        session.add_all(alerts)
        session.flush()
        ids = [a.id for a in alerts]
        session.commit()
    engine.dispose()
    return ids


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


# ── Async helpers (FastAPI routers) ───────────────────────────────────────────

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


async def bulk_insert_alerts(db: AsyncSession, alert_dicts: list[dict]) -> list[Alert]:
    alerts = [Alert(**d) for d in alert_dicts]
    db.add_all(alerts)
    await db.commit()
    for a in alerts:
        await db.refresh(a)
    return alerts


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

    total = (await db.execute(count_base)).scalar_one()
    paged = base.order_by(Alert.timestamp.desc()).offset((page - 1) * size).limit(size)
    return list((await db.execute(paged)).scalars().all()), total


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
