"""Pipeline coordinator — orchestrates all ML pipeline stages (ADR-24).

Both api/http/ and api/offline/ call this coordinator; neither contains ML logic.
The on_progress callback is the polymorphic seam between HTTP (SSE) and offline (print).
"""
from __future__ import annotations

# TODO: implement PipelineCoordinator by extracting orchestration logic from
#       api/http/routers/training.py (HTTP path) and api/offline/train.py (CLI path).
#
# Interface:
#   coordinator = PipelineCoordinator()
#   coordinator.run_eda(dataset_dir)
#   coordinator.run_training(dataset_dir, device, on_progress=None)
#   coordinator.run_fine_tune(model_version_id, device, on_progress=None)
#   coordinator.run_inference(input_path, device="cpu")
