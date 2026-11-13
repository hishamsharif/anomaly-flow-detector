import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import get_settings
from api.http.dao.db_conn import AsyncSessionLocal
from api.http.dao import model_versions as model_versions_dao
from api.http.routers import health
from api.http.routers import models as models_router
from api.http.routers import training as training_router
from api.http.routers import predict as predict_router
from api.http.routers import alerts as alerts_router
from api.http.routers import about as about_router
from api.http.routers import dashboard as dashboard_router
from api.http.routers import dataset as dataset_router
from pipeline.store import ModelStore
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
            await model_versions_dao.mark_stale_training_as_failed(db)
            active = await model_versions_dao.get_active_model(db)
        if active and active.path:
            try:
                ModelStore.get().load_into_memory(active.path)
                logger.info("Loaded active model: %s", active.name)
            except Exception as exc:
                logger.warning("Active model could not be loaded (%s) — inference will be unavailable until retrained.", exc)
    except Exception as exc:
        logger.warning("Startup DB operations skipped: %s", exc)
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
app.include_router(training_router.router)
app.include_router(predict_router.router)
app.include_router(alerts_router.router)
app.include_router(dashboard_router.router)
app.include_router(about_router.router)
app.include_router(dataset_router.router)
