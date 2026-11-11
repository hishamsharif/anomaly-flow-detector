from __future__ import annotations

from sqlalchemy import create_engine, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from api.db.models import ModelVersion


# ── Sync helpers for training threads ─────────────────────────────────────────

def _sync_engine(db_url: str):
    return create_engine(db_url.replace("sqlite+aiosqlite", "sqlite"))


def sync_update_model_version(db_url: str, version_id: int, **kwargs) -> None:
    engine = _sync_engine(db_url)
    with Session(engine) as s:
        s.execute(update(ModelVersion).where(ModelVersion.id == version_id).values(**kwargs))
        s.commit()
    engine.dispose()


def sync_set_active(db_url: str, version_id: int) -> None:
    """Archive existing active, promote version_id to active."""
    engine = _sync_engine(db_url)
    with Session(engine) as s:
        s.execute(update(ModelVersion).where(ModelVersion.status == "active").values(status="archived"))
        s.execute(update(ModelVersion).where(ModelVersion.id == version_id).values(status="active"))
        s.commit()
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
