from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from api.http.dao.db_conn import get_db
from pipeline.store import ModelStore
from runtime.manifest import get_host_environment
from runtime.preflight import build_preflight_report

router = APIRouter()


@router.get("/health")
async def health_check(db: AsyncSession = Depends(get_db)):
    db_ok = False
    try:
        await db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        pass
    model_loaded = ModelStore.get().get_active() is not None
    return {"status": "ok", "model_loaded": model_loaded, "db_ok": db_ok}


@router.get("/api/preflight")
async def preflight():
    env = get_host_environment()
    return build_preflight_report(env)
