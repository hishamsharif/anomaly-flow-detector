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
| `Makefile` | `dev`, `test`, `export-spec`, `migrate`, `clean-models` targets |
| `api/config.py` | `Settings` singleton; all config vars from §7 |
| `api/database.py` | async SQLAlchemy engine; session factory |
| `api/main.py` | FastAPI app, lifespan (loads active model on startup), router mounts |
| `runtime/probe.py` | `probe_cuda()`, `probe_docker()`, `probe_images()` → `HostEnvironment` dataclass |
| `runtime/manifest.py` | `HostEnvironment` singleton — resolved once at startup via `probe.py` |
| `runtime/preflight.py` | `build_preflight_report()` → `{cuda, docker, images}` with `ok`/`warn`/`blocker` levels |
| `GET /health` | Returns `{"status": "ok", "model_loaded": false, "db_ok": true}` |
| `GET /api/preflight` | Calls `build_preflight_report()`; returns structured check results |
| `docker-compose.yml` | App + Vite dev server (SQLite — no Postgres container needed; see ADR-09) |
| Alembic init | `alembic.ini` + `alembic/env.py`; empty initial migration |
| UI shell | React + Vite + Tailwind; navbar + tab placeholders; hits `/health` + `/api/preflight` |
| UI — prerequisites screen | Renders on load if any `blocker` present; shows fix instructions |
| `tests/conftest.py` | `app` fixture (TestClient); `tmp_model_dir` fixture |
| Tests | `test_health.py` — 200 OK; `test_preflight.py` — returns all three check keys |

---

### Slice 1 — Model Registry (~4h)

**Deliverable:** Training tab dropdowns show "XGBoost Classifier" and "MLP Autoencoder"; active model returns null.

| File / component | Work |
|-----------------|------|
| `api/schemas.py` | `ModelVersionResponse` Pydantic schema |
| `core/base.py` | `Classifier` + `Detector` protocols |
| `core/registry.py` | `CLASSIFIERS` + `DETECTORS` dicts |
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

**Deliverable:** User uploads a CSV on the Training tab, clicks Start, watches the SSE progress bar fill, then sees the new version appear in history.

| File / component | Work |
|-----------------|------|
| `training/data_loader.py` | DuckDB CSV scan; returns `pd.DataFrame` |
| `training/feature_eng/variance_filter.py` | Drop features ≤ `VARIANCE_THRESHOLD` |
| `training/feature_eng/correlation_filter.py` | Drop correlated features ≥ `CORR_THRESHOLD` |
| `training/feature_eng/selector.py` | MI / RFE selector (saves `variance_mask`, `corr_mask`) |
| `training/feature_eng/pipeline.py` | Chains the four steps: variance → correlation → selector → **SMOTE** (Step 4, §5.1) |
| `core/classifiers/xgboost_clf.py` | XGBoost; respects `DEVICE` config var |
| `core/detectors/mlp_ae.py` | MLP-AE; PyTorch; respects `DEVICE` config var |
| `training/pipelines/initial.py` | Full pipeline: load → feature eng → fit CLF → fit AE → save |
| `core/store.py` | `ModelStore`: `save(version)` / `load(id)` / `set_active(id)` |
| `api/jobs.py` | SSE bridge; yields training log lines |
| `POST /api/training/start` | Launches training job; returns `{"job_id": "..."}` |
| `GET /api/training/{run_id}/events` | SSE endpoint; streams job progress |
| `GET /api/models` | Now returns the saved version |
| UI — Training tab | Start button; SSE progress bar; version history table |
| Tests | Training completes on a 1 000-row fixture CSV; artifacts saved; version appears |

**GPU note:** `DEVICE: "auto"` resolves to `cuda` on the host (RTX A2000). Docker requires `--build-arg TORCH_BUILD_TYPE=cu121` rebuild + `nvidia-container-toolkit`.

---

### Slice 3 — Inference / Alert Generation (~6h)

**Deliverable:** User uploads a CSV on the Inference tab; a paginated alerts table appears with verdict, ae_score, severity.

| File / component | Work |
|-----------------|------|
| `core/rule_engine.py` | Stage 1 deterministic rules; CVE sigs, allowlisted IPs |
| `inference/pipeline.py` | 3-stage runner: rules → classifier → detector → consensus |
| `inference/adapters/csv_adapter.py` | CSV → `List[FeatureRow]` |
| `api/db/models.py` | `Alert` ORM (Alembic migration) |
| `api/schemas.py` | `AlertResponse`, `BatchPredictResponse` schemas |
| I3 shape check | `inference/pipeline.py`: validate feature vector length against saved mask; raise HTTP 400 on mismatch (§3.2 I3) |
| `POST /api/predict/batch` | Sync (≤ SYNC_MAX_ROWS): returns `BatchResult` directly; async (> SYNC_MAX_ROWS): returns `{"job_id": str, "flow_count": int}` (§4.4) |
| `GET /api/predict/{job_id}/status` | Async poll: `{"job_id", "pct", "done"}` — required for files > SYNC_MAX_ROWS |
| `GET /api/predict/{job_id}/results` | Async result fetch once `done=true`; 409 if still running |
| Explainability — `top_features` | Every `Alert` includes top-N features by classifier `feature_importances_` (Stage 2) **and** per-feature AE reconstruction error `(x_i − x̂_i)²` (Stage 3); both required by §4.4 Alert schema |
| `GET /api/alerts` | Paginated alert list; supports `verdict`, `severity` filters |
| UI — Inference tab | CSV upload; results table (flow id, verdict, confidence, ae_score, severity) |
| Tests | Batch predict with known-attack fixture produces `KNOWN_ATTACK`; Stage 2 low-confidence flow passes to Stage 3 (confidence gate); SAFE passes through; `top_features` populated; pagination correct |

---

### Slice 4 — Analyst Verdict + Dashboard (~4h)

**Deliverable:** Analyst confirms/corrects alerts; Dashboard shows live summary and "Retrain AE" button activates at threshold.

| File / component | Work |
|-----------------|------|
| `POST /api/alerts/{id}/verdict` | Accepts `{verdict, notes}`; sets `analyst_verdict`, updates `retrain_eligible` |
| `GET /api/alerts/verdict-count` | Returns `{"eligible": int, "threshold": int, "unlocked": bool}` |
| `GET /api/dashboard/summary` | Returns alert counts by verdict + severity |
| UI — Dashboard | Alert summary cards; verdict progress bar; "Retrain AE" button (disabled until unlocked) |
| UI — Inference tab | Confirm / Correct buttons on each row |
| Tests | Verdict submission persists; retrain-status `unlocked` flips at threshold |

---

### Slice 5 — AE Fine-tune (~3h)

**Deliverable:** User clicks "Retrain AE" on the Dashboard and watches the AE retrain on analyst-confirmed flows; new version appears.

| File / component | Work |
|-----------------|------|
| `training/pipelines/fine_tune.py` | Loads confirmed + override flows; freezes CLF weights; retrains AE only |
| `POST /api/training/start` (mode=`ae_retrain`) | Launches AE fine-tune job; SSE progress (reuses `api/jobs.py`) |
| `POST /api/models/{id}/promote` | Promotes a version to active |
| UI — Dashboard | Retrain AE button triggers fine-tune; version history updates |
| Tests | Fine-tune runs on eligible alerts; new version saved; promotion updates active |

---

### Slice 6 — PCAP Adapter + About + Polish (~3h)

**Deliverable:** Full feature-complete system.

| File / component | Work |
|-----------------|------|
| `inference/adapters/pcap_adapter.py` | PCAP → `List[FeatureRow]` (dpkt or scapy) |
| `cli/predict.py` | Standalone prediction script; loads active model, runs CSV through pipeline, prints results |
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
| 5 — AE Fine-tune | Retrain AE → new version | 3h |
| 6 — PCAP + About + Polish | PCAP upload; About page | 3h |
| **Total** | | **~34h** |

## Slice dependencies

```
0 ──► 1 ──► 1.5 ──► 2 ──► 3 ──► 4 ──► 5 ──► 6
```

Each slice is blocked on the previous; there are no parallel tracks because every slice
adds to the same running stack. If time is short, Slice 6 can be dropped — Slices 0–5
already demonstrate all A2 assessment criteria.
