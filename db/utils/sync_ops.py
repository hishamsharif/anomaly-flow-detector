"""Synchronous database helpers for training pipelines.

These functions run in background threads (not inside the FastAPI async loop)
and must NOT import from api/ — keeping training/ fully decoupled from the
web layer satisfies SPEC §9.

SQLAlchemy Core table reflection is used so this module has zero ORM / api
dependencies: it reads the live schema from the DB file at call time.
"""

from __future__ import annotations

from sqlalchemy import MetaData, create_engine, update


def sync_engine_for(db_url: str):
    """Return a synchronous SQLAlchemy engine from an async-dialect URL."""
    return create_engine(db_url.replace("sqlite+aiosqlite", "sqlite"))


def _model_versions_table(engine):
    meta = MetaData()
    meta.reflect(bind=engine, only=["model_versions"])
    return meta.tables["model_versions"]


def sync_update_model_version(db_url: str, version_id: int, **kwargs) -> None:
    """Update arbitrary columns on a model_versions row."""
    engine = sync_engine_for(db_url)
    tbl = _model_versions_table(engine)
    with engine.connect() as conn:
        conn.execute(update(tbl).where(tbl.c.id == version_id).values(**kwargs))
        conn.commit()
    engine.dispose()


def sync_set_active(db_url: str, version_id: int) -> None:
    """Archive the existing active model version, then promote version_id to active."""
    engine = sync_engine_for(db_url)
    tbl = _model_versions_table(engine)
    with engine.connect() as conn:
        conn.execute(update(tbl).where(tbl.c.status == "active").values(status="archived"))
        conn.execute(update(tbl).where(tbl.c.id == version_id).values(status="active"))
        conn.commit()
    engine.dispose()
