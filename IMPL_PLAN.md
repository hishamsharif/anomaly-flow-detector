# AFD V2 — Implementation Plan
## Incremental Vertical Working Slices

Each slice delivers a working, end-to-end feature — not a horizontal layer.
Every slice ships its own tests; nothing is marked done without a passing test.

---

### Slice 0 — Project Scaffold (~2h)

**Deliverable:** `make run` starts the full stack; browser shows "Connected".

| File / component | Work |
|-----------------|------|
| Folder structure | Create all dirs per §9; add `.gitkeep` where empty |
| `requirements.txt` / `pyproject.toml` | Pin all deps; include `cu121` torch wheel |
| `Makefile` | `up` / `down` (stack), `dev` (hot-reload), `train`, `test`, `lint`, `export-spec`, `migrate`, `clean-models` targets |
| `api/config.py` | `Settings` singleton; all config vars from §7 |
| `api/database.py` | async SQLAlchemy engine; session factory |
| `api/main.py` | FastAPI app, lifespan: marks stale `training` rows → `failed` on startup (ADR-07 crash recovery), then loads active model into memory; mounts all routers |
| `runtime/probe.py` | `probe_cuda()`, `probe_docker()`, `probe_images()` → `HostEnvironment` dataclass |
| `runtime/manifest.py` | `HostEnvironment` singleton — resolved once at startup via `probe.py` |
| `runtime/preflight.py` | `build_preflight_report()` → `{cuda, docker, images}` with `ok`/`warn`/`blocker` levels |
| `GET /health` | Returns `{"status": "ok", "model_loaded": false, "db_ok": true}` |
| `GET /api/preflight` | Calls `build_preflight_report()`; returns structured check results |
| `docker-compose.yml` | App + Vite dev server (SQLite — no Postgres container needed; see ADR-09) |
| `.env.demo` | Committed reference file with working defaults for all §7 config vars; developers copy to `.env` to get started |
| Alembic init | `alembic.ini` + `alembic/env.py`; empty baseline migration |
| SPA shell | React 18 + TypeScript + Vite 5 + Tailwind (ADR-17); navbar + 4 tab stubs (Dashboard, Training, Inference, About); polls `GET /health` and `GET /api/preflight` on load |
| UI — preflight screen | Renders if any check is `blocker`; lists fix instructions per failing check; flow halted until resolved |
| `tests/conftest.py` | `app` fixture (TestClient); `client` async fixture (AsyncClient via httpx for async endpoint tests); `tmp_model_dir` fixture |
| Tests | `test_health.py` — 200 OK; `test_preflight.py` — returns all three check keys |

---

### Slice 1 — Model Registry (~4h)

**Deliverable:** Training tab dropdowns show "XGBoost Classifier" and "MLP Autoencoder"; active model returns null.

| File / component | Work |
|-----------------|------|
| `api/schemas.py` | Establishes the centralised Pydantic schema module (SDD contract layer, ADR-18); adds `ModelVersionResponse` |
| `core/base.py` | `BaseClassifier` + `BaseAnomalyDetector` Protocol classes (structural subtyping, ADR-16) |
| `core/registry.py` | `CLASSIFIER_REGISTRY` + `ANOMALY_DETECTOR_REGISTRY` dicts; one line to register a new model type |
| `api/db/models.py` | `ModelVersion` ORM (id, name, classifier_type, detector_type, status, path, metrics_json, created_at) |
| Alembic migration | `model_versions` table |
| `GET /api/models/available-types` | Returns registered classifier + detector names |
| `GET /api/models/active` | Returns `null` until a model is trained |
| `GET /api/models` | Returns `{"items": [], "total": 0}` |
| UI — Training tab | Classifier + detector dropdowns populated from `/available-types` |
| Tests | Available-types returns both; active returns `null`; versions returns `[]` |

---

### Slice 1.5 — EDA (~3h)

**Deliverable:** `training/eda/eda_cicids2017.ipynb` committed — time-series analysis of CIC-IDS2017 that justifies the day-based split, SMOTE cap, and AE threshold percentile. Required by §5.0.

> **Prerequisite:** `make dataset` must have run (CIC-IDS2017 CSVs present).

| File / component | Work |
|-----------------|------|
| `training/eda/eda_cicids2017.ipynb` | Full EDA notebook (see §5.0 for required analyses) |
| Flow volume by hour/day | Confirm diurnal cycle; show Mon–Thu and Fri-AM share the same pattern |
| Attack-type distribution by day | Evidence that Fri-PM = DDoS + PortScan only → genuine unseen test set |
| Label imbalance per split | BENIGN:attack ratio for train/val/test; primary justification for SMOTE (ADR-14) |
| Feature distribution Mon→Fri | Rolling mean/std of key features; detects concept drift across capture days |
| Benign seasonality Fri-AM vs Mon–Thu | Validates val-benign is representative; informs AE threshold calibration (ADR-05) |
| Correlation heatmap | Pre- and post-pruning; confirms correlation filter effect |
| Feature variance ranking | Confirms VarianceThreshold drops no informative feature |

**Findings feed directly into Slice 2:** SMOTE cap (`SMOTE_MAX_PER_CLASS`), threshold percentile (`AE_ANOMALY_THRESHOLD_PERCENTILE`), and correlation threshold (`CORRELATION_THRESHOLD`) should be set after reviewing the notebook, not before.

---

### Slice 2 — Initial Training Pipeline (~8h)

**Deliverable:** User clicks Start Training; CIC-IDS2017 is loaded from `BUNDLED_DATASET_DIR`, the full pipeline runs with SSE progress streaming, and the trained model version appears in history.

| File / component | Work |
|-----------------|------|
| `training/data_loader.py` | DuckDB CSV scan; returns `pd.DataFrame` |
| `training/feature_eng/variance_filter.py` | Drop features ≤ `VARIANCE_THRESHOLD` |
| `training/feature_eng/correlation_filter.py` | Drop correlated features ≥ `CORR_THRESHOLD` |
| `training/feature_eng/selector.py` | SelectKBest (mutual information) or RF-RFE feature selector; outputs `selector_mask` |
| `training/feature_eng/pipeline.py` | Chains all steps (§5.1 + §2b): variance filter → correlation filter → **SMOTE** (training split only, ADR-14) → feature selector → dual scalers (classifier path + detector benign-only path) |
| `core/classifiers/xgboost_clf.py` | XGBoost; respects `DEVICE` config var |
| `core/detectors/mlp_ae.py` | MLP-AE; PyTorch; respects `DEVICE` config var |
| `training/pipelines/initial.py` | Orchestrates full training: data load → feature engineering → fit XGBoost classifier → fit + calibrate MLP-AE on benign-only data → evaluate on held-out test split → save artifacts atomically (ADR-12) → auto-promote to active |
| `core/store.py` | `ModelStore`: `save(version)` / `load(id)` / `set_active(id)`; saves all three masks (`variance_mask`, `corr_mask`, `selector_mask`) and both scalers |
| `api/jobs.py` | In-process thread-job bridge: `ThreadPoolExecutor(max_workers=1)` + append-only event log with cursor replay; SSE keep-alive heartbeat every 30 s (ADR-07, ADR-08) |
| `POST /api/training/start` | 409 when a job is already running; launches job via `launch_training_job()`; returns `{"run_id": str}` |
| `GET /api/training/{run_id}/events` | SSE stream; replays buffered event history on reconnect |
| `GET /api/training/{run_id}/status` | Poll fallback when SSE is unavailable; returns `{"run_id", "pct", "mode", "started_at"}` |
| `GET /api/models` | Now returns the saved version; response shape: `{"items": ModelVersion[], "total": int}` |
| UI — Training tab | Start button; SSE progress bar; version history table |
| Tests | Training completes on a 1 000-row fixture CSV; artifacts saved; version appears |

**GPU note:** `DEVICE: "auto"` resolves to `cuda` on the host (RTX A2000). Docker requires `--build-arg TORCH_BUILD_TYPE=cu121` rebuild + `nvidia-container-toolkit`.

---

### Slice 3 — Inference / Alert Generation (~6h)

**Deliverable:** User uploads a CSV on the Inference tab; a paginated alerts table appears with verdict, ae_score, severity.

| File / component | Work |
|-----------------|------|
| `core/rule_engine.py` | Stage 1 deterministic rules: exits immediately as `KNOWN_ATTACK` on CVE signatures / known-malicious IPs; exits as `SAFE` on allowlisted assets; ambiguous flows pass to Stage 2 |
| `inference/pipeline.py` | 3-stage inference pipeline (§5.2): Stage 1 rules (always takes precedence) → Stage 2 XGBoost (exits `KNOWN_ATTACK` when confidence ≥ `CLF_CONFIDENCE_THRESHOLD`) → Stage 3 MLP-AE (always runs; finalises `NOVEL_ATTACK` or `SAFE` when Stage 2 passed through) |
| `inference/adapters/csv_adapter.py` | CSV → validated `List[FeatureRow]`; validates column names against `feature_names.json` (invariant I3); unrecognised columns returned in `BatchResult.rejected_rows` |
| `api/db/models.py` | `Alert` ORM (Alembic migration) |
| `api/schemas.py` | `AlertResponse`, `BatchPredictResponse` schemas |
| I3 shape check | `csv_adapter.py`: validates input column names match `feature_names.json`; missing/extra columns → row added to `rejected_rows` with reason (§3.2 I3) |
| `POST /api/predict/batch` | Sync (≤ SYNC_MAX_ROWS): returns `BatchResult` directly; async (> SYNC_MAX_ROWS): returns `{"job_id": str, "flow_count": int}` (§4.4) |
| `GET /api/predict/{job_id}/status` | Async poll: `{"job_id", "pct", "done"}` — required for files > SYNC_MAX_ROWS |
| `GET /api/predict/{job_id}/results` | Async result fetch once `done=true`; 409 if still running |
| Explainability — `top_features` | Stage 2: top-N features ranked by `feature_importances_` × scaled input value (local attribution proxy, §5.2); Stage 3: top-N features by per-feature reconstruction error `(x_i − x̂_i)²` (ADR-13); both fields in every `Alert` per §4.4 schema |
| `GET /api/alerts` | Paginated alert list; supports `verdict`, `severity` filters |
| UI — Inference tab | CSV upload; results table (flow id, verdict, confidence, ae_score, severity) |
| Tests | Batch predict with known-attack fixture produces `KNOWN_ATTACK`; Stage 2 low-confidence flow passes to Stage 3 (confidence gate); SAFE passes through; `top_features` populated; pagination correct; 409 when no active model; 413 when file exceeds MAX_UPLOAD_MB |

---

### Slice 4 — Analyst Verdict + Dashboard (~4h)

**Deliverable:** Analyst confirms/corrects alerts; Dashboard shows live summary and "Retrain AE" button activates at threshold.

| File / component | Work |
|-----------------|------|
| `POST /api/alerts/{id}/verdict` | Accepts `{verdict, notes}`; sets `analyst_verdict`, updates `retrain_eligible`; 409 when verdict already submitted (invariant I2) |
| `GET /api/alerts/verdict-count` | Returns `{"eligible": int, "threshold": int, "unlocked": bool}` |
| `GET /api/dashboard/summary` | Returns alert counts by verdict + severity |
| UI — Dashboard | Alert summary cards; verdict progress bar; "Retrain AE" button (disabled until unlocked) |
| UI — Inference tab | Confirm / Correct buttons on each row |
| Tests | Verdict submission persists; retrain-status `unlocked` flips at threshold; 409 on second verdict submission |

---

### Slice 5 — AE Fine-tune (~4h)

**Deliverable:** User clicks "Retrain AE" on the Dashboard and watches the AE retrain on analyst-confirmed flows; new version appears.

| File / component | Work |
|-----------------|------|
| `training/pipelines/fine_tune.py` | Full §5.3 pipeline — (1) fetch eligible benign flows: `confirmed` on SAFE alerts + `override` on any alert; (2) combine with `AE_REPLAY_BUFFER_FRACTION × historical benign data` (ADR-06 replay buffer); (3) load previous AE checkpoint, **freeze encoder** for `AE_FREEZE_EPOCHS` epochs; (4) unfreeze + fine-tune remaining epochs; (5) recompute AE threshold on val-benign (p`AE_ANOMALY_THRESHOLD_PERCENTILE`); (6) **regression gate** — compare new FAR + recall vs previous version; emit WARN if `far_increase > FAR_GATE_MAX` or `recall_drop > RECALL_GATE_MAX` (advisory only, never blocks); (7) copy `classifier.joblib` unchanged from previous version (invariant I5) |
| `POST /api/training/start` (mode=`ae_retrain`) | **409** when no active model; **409** when `eligible_verdicts < FINE_TUNE_K_THRESHOLD` (invariant I4); launches fine-tune job; SSE progress (reuses `api/jobs.py`) |
| `POST /api/models/{id}/promote` | Promotes a version to active |
| UI — Dashboard | Retrain AE button triggers fine-tune; version history updates; regression gate WARN displayed when advisory check fired |
| Tests | 409 when eligible_verdicts < threshold; fine-tune runs on eligible alerts; replay buffer included in training data; encoder frozen for first N epochs; AE threshold recalibrated on val-benign; regression gate fires WARN on FAR increase; `classifier.joblib` byte-identical to previous version; new version saved; promotion updates active |

---

### Slice 6 — PCAP Adapter + About + Polish (~3h)

**Deliverable:** Full feature-complete system.

| File / component | Work |
|-----------------|------|
| `inference/adapters/pcap_adapter.py` | PCAP → `List[FeatureRow]` (dpkt or scapy) |
| `predict.py` (root) | Standalone examiner; loads active model once, batch-scores CSV through the full 3-stage pipeline without HTTP overhead; target ≥ 10k flows/s on GPU (28k achieved in V1) |
| `docs/architecture.md` | Human-readable architecture doc: 3-stage pipeline diagram, data-flow, component responsibilities |
| `GET /api/about` | Returns system info, active model version, config vars |
| UI — Inference tab | PCAP upload alongside CSV upload |
| UI — About page | Renders `/api/about` data; copy (labels, descriptions) should track spec language |
| Tests | PCAP adapter produces correct feature rows on a small fixture |

---

## Summary

| Slice | What it delivers | Est. effort |
|-------|-----------------|-------------|
| 0 — Scaffold | `make dev` starts; `/health` + `/api/preflight` work | 3h |
| 1 — Model Registry | Training tab dropdowns populated | 4h |
| 1.5 — EDA | `eda_cicids2017.ipynb`; time-series analysis; informs Slice 2 config | 3h |
| 2 — Initial Training | Upload CSV → progress → version history | 8h |
| 3 — Inference | Upload CSV → alerts table (sync + async); `top_features` in every alert | 6h |
| 4 — Verdict + Dashboard | Verdict submission → Dashboard live | 4h |
| 5 — AE Fine-tune | Retrain AE → new version (replay buffer + encoder freeze + regression gate) | 4h |
| 6 — PCAP + About + Polish | PCAP upload; About page | 3h |
| **Total** | | **~34h** |

## Slice dependencies

```
0 ──► 1 ──► 1.5 ──► 2 ──► 3 ──► 4 ──► 5 ──► 6
```

Each slice is blocked on the previous; there are no parallel tracks because every slice
adds to the same running stack. If time is short, Slice 6 can be dropped — Slices 0–5
already demonstrate all A2 assessment criteria.
