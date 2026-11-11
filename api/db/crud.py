from __future__ import annotations

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.models import ModelVersion


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
