"""Offline initial training script — equivalent of POST /api/training/start (initial mode).

Run via:  make train
Or directly:  PYTHONPATH=. python scripts/train_offline.py [--clf xgboost] [--det mlp_ae]

Creates a ModelVersion row, runs the classifier + AE pipeline, and marks the
resulting version as active.  Device is resolved from settings (DEVICE env var)
the same way the API router does (ADR-23).
"""
from __future__ import annotations

import argparse
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Project root must be on PYTHONPATH (enforced by Makefile: PYTHONPATH=.)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _create_version_sync(db_url: str, run_id: str, clf_type: str, det_type: str) -> tuple[int, str]:
    """Insert a model_versions row and return (id, name)."""
    from sqlalchemy import MetaData, Table, create_engine, insert

    from core.registry import CLASSIFIER_LABELS, DETECTOR_LABELS

    sync_url = db_url.replace("sqlite+aiosqlite", "sqlite")
    engine = create_engine(sync_url)
    meta = MetaData()
    meta.reflect(bind=engine, only=["model_versions"])
    tbl: Table = meta.tables["model_versions"]

    with engine.connect() as conn:
        result = conn.execute(
            insert(tbl).values(
                name="(pending)",
                classifier_type=clf_type,
                detector_type=det_type,
                status="training",
                created_at=datetime.now(timezone.utc),
            )
        )
        conn.commit()
        version_id: int = result.inserted_primary_key[0]

        clf_label = CLASSIFIER_LABELS.get(clf_type, clf_type)
        det_label = DETECTOR_LABELS.get(det_type, det_type)
        version_name = f"{clf_label} + {det_label} — v{version_id}"
        from sqlalchemy import update
        conn.execute(update(tbl).where(tbl.c.id == version_id).values(name=version_name))
        conn.commit()

    engine.dispose()
    return version_id, version_name


def main() -> None:
    parser = argparse.ArgumentParser(description="AFD V2 offline initial training")
    parser.add_argument("--clf", default="xgboost", help="Classifier type (default: xgboost)")
    parser.add_argument("--det", default="mlp_ae", help="Detector type (default: mlp_ae)")
    args = parser.parse_args()

    from config.settings import get_settings
    from training.pipelines.initial import run_initial_training

    settings = get_settings()

    # Resolve device (ADR-23): probe CUDA here at the entry point, not inside training/
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
    except Exception:
        cuda_ok = False

    if settings.device == "auto":
        device = "cuda" if cuda_ok else "cpu"
    else:
        device = settings.device

    log.info("Dataset   : %s", settings.bundled_dataset_dir)
    log.info("Device    : %s%s", device, " (CUDA detected)" if device == "cuda" else "")
    log.info("Classifier: %s  |  Detector: %s", args.clf, args.det)
    log.info("")

    # Ensure DB file exists (Alembic migrations should already be applied via make train)
    db_path = Path("data/afd.db")
    if not db_path.exists():
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db_path.touch()
        log.warning("data/afd.db not found — created empty file; run 'make migrate' first if schema is missing")

    run_id = str(uuid.uuid4())
    version_id, version_name = _create_version_sync(settings.database_url, run_id, args.clf, args.det)

    log.info("Run ID   : %s", run_id)
    log.info("Version  : %s (id=%d)", version_name, version_id)
    log.info("")

    def emit(event: str, data: str) -> None:
        log.info("[%s] %s", event, data)

    run_initial_training(
        run_id, version_id, version_name,
        args.clf, args.det,
        settings, emit, settings.database_url, device,
    )

    log.info("")
    log.info("Training complete — model '%s' is now active.", version_name)
    log.info("Start the stack with:  make start")


if __name__ == "__main__":
    main()
