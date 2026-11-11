import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import get_settings
from api.database import AsyncSessionLocal
from api.db import crud
from api.routers import health
from api.routers import models as models_router
from runtime.manifest import resolve_host_environment

settings = get_settings()
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Startup — resolving host environment…")
    resolve_host_environment()
    logger.info("Host environment ready.")
    try:
        async with AsyncSessionLocal() as db:
            await crud.mark_stale_training_as_failed(db)
    except Exception as exc:
        logger.warning("Stale-training cleanup skipped: %s", exc)
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Anomaly Flow Detector",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(models_router.router)
