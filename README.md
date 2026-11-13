# Anomaly Flow Detector (AFD)

3-stage Network Intrusion Detection System: Rule Engine → XGBoost Classifier → MLP Autoencoder.
Trained on CIC-IDS2017; produces **Known Attack**, **Novel/Zero-day Attack**, or **Safe** verdicts per flow with XAI feature attributions. Includes a Human-in-the-Loop (HITL) feedback loop that fine-tunes the autoencoder as analysts submit verdicts.

---

## Quick Start (Docker)

> Requires Docker Desktop (running). `make setup` handles everything else.

```bash
git clone https://github.com/hishamsharif/anomaly-flow-detector
cd anomaly-flow-detector
make setup    # create afd_v2 conda env + deps (GPU if CUDA found) + download dataset
make start    # Docker if available, local fallback otherwise
```

- API → `http://localhost:8000`
- UI  → `http://localhost:5173`

---

## Prerequisites

### Docker path (examiner / CI)
| Requirement | Notes |
|---|---|
| Docker Desktop | daemon must be running (`docker version` to verify) |
| NVIDIA Container Toolkit | optional — `make start` applies GPU passthrough automatically when detected |

### Local dev path
| Requirement | Notes |
|---|---|
| Miniconda / conda | manages the `afd_v2` Python env — `make setup` creates it |
| Python 3.12 | inside the conda env, managed by `make setup` |
| Node 20+ | only needed when running the UI outside Docker (`make start ui`) |

---

## Local Dev (without Docker)

```bash
make setup          # one-off: create env + install deps (auto GPU) + download dataset
make start api      # apply migrations + start API on :8000 (hot-reload, separate terminal)
make start ui       # start Vite dev server on :5173 (separate terminal)
```

All `make` targets use the `afd_v2` env directly — no `conda activate` needed.
Logs from both services are written to `logs/api.log` and `logs/ui.log`.

---

## Clean Slate

```bash
make resetup        # stop stack, wipe env + DB + models + logs, then run setup from scratch
```

Use `make resetup` before a demo or when you want to verify the full setup path from zero.
After `resetup`, run `make start` → navigate to Training → start initial training.

---

## Configuration

Copy `.env.demo` to `.env`. All variables are optional — defaults shown.

```ini
DATABASE_URL=sqlite+aiosqlite:///./data/afd.db
MODEL_ARTIFACT_DIR=models
DEVICE=auto                         # "auto" resolves via torch.cuda.is_available()

CLF_CONFIDENCE_THRESHOLD=0.5
MAX_UPLOAD_MB=100
SYNC_MAX_ROWS=5000

VARIANCE_THRESHOLD=0.0
CORRELATION_THRESHOLD=0.95
FEATURE_SELECTION_K=0               # 0 = disabled
FEATURE_SELECTION_METHOD=mi         # mi | rfe

AE_ANOMALY_THRESHOLD_PERCENTILE=95.0
AE_REPLAY_BUFFER_FRACTION=0.10
AE_FREEZE_EPOCHS=5
FINE_TUNE_K_THRESHOLD=50            # analyst verdicts needed to unlock AE fine-tune

LOG_LEVEL=INFO
CORS_ORIGINS=["http://localhost:5173","http://localhost"]
```

---

## Make Targets

```
Setup
  make setup              Create afd_v2 env + deps (auto GPU) + download CIC-IDS2017 dataset
  make resetup            Wipe env + DB + models + logs, then run setup from scratch

Offline (no server required)
  make train              Initial training: classifier + AE on CIC-IDS2017  (GPU if available)
  make infer CSV=<file>   Score a CSV through the full pipeline

Runtime
  make start              Start full stack  (Docker if available, local guide otherwise)
  make start api          Start just the API locally on :8000  (hot-reload)
  make start ui           Start just the UI dev server on :5173
  make stop               Stop the Docker stack

Logs  (auto-detects Docker or local log files)
  make logs               Follow all logs
  make logs api           Follow API logs only
  make logs ui            Follow UI logs only

Application Status
  make status             Check both API and UI are reachable
  make status api         Check API /health endpoint
  make status ui          Check UI is serving on :5173

Development
  make test               Run pytest suite
  make lint               Ruff check + format check
  make migrate            Apply Alembic migrations
  make export-spec        Write docs/openapi.json from the live app
  make clean-models       Delete versioned model artifacts under models/
```

---

## Demo Workflow

1. `make start` — starts API + UI
2. Navigate to **Training** → select XGBoost + MLP-AE → **Start Training** (SSE progress bar)
3. Navigate to **Inference** → upload a CSV or PCAP file → **Run Inference**
4. Expand alert rows → review XAI feature attributions → submit analyst verdicts
5. Navigate to **Dashboard** → watch the HITL verdict progress bar fill toward K=50
6. Once K=50 is reached → **Retrain AE** button unlocks → trigger AE fine-tune
7. Navigate to **Dataset** → EDA charts for CIC-IDS2017 class distribution and feature correlations

Offline CLI (no server required):
```bash
# Initial training
make train

# Score a CSV
make infer CSV=data/sample.csv OUTPUT=results.csv

# Or invoke directly
PYTHONPATH=. python inference/inference_offline.py --csv data/sample.csv
PYTHONPATH=. python inference/inference_offline.py --pcap data/capture.pcap --output results.csv
```

---

## API Endpoints

### `GET /health`
```json
{ "status": "ok", "model_loaded": false, "db_ok": true }
```

### `GET /api/preflight`
Host-environment readiness. Each check is `ok`, `warn`, or `blocker`.
```json
{
  "cuda":   { "level": "ok",   "message": "CUDA available", "detail": "CUDA 12.1" },
  "docker": { "level": "ok",   "message": "Docker available", "detail": "Docker 29.2.1" },
  "images": { "level": "warn", "message": "Images not built yet", "detail": "Run make start once." }
}
```

### `GET /api/models/available-types`
Supported classifier and detector IDs with display labels.

### `GET /api/models` / `GET /api/models/active`
Paginated model version list; active model or `null`.

### `POST /api/training/start`
Start initial training or AE fine-tune. Returns `run_id` for SSE subscription.

### `GET /api/training/{run_id}/events`
SSE stream: `progress` (log line), `step` (0–100 int), `done`, or `error` events.

### `POST /api/predict/batch`
Upload a `.csv` or `.pcap` file. Returns alerts with XAI attributions.
- Sync (≤ 5 000 rows): `200 BatchPredictResponse`
- Async (> 5 000 rows): `202 { "job_id": str, "flow_count": int }`

### `GET /api/alerts` / `POST /api/alerts/{id}/verdict`
Paginated alert listing with optional `consensus` / `verdict` filters; verdict submission.

### `GET /api/dashboard/summary`
24-hour windowed alert counts, severity breakdown, HITL retrain eligibility.

---

## Tests

```bash
make test
```

Tests use `httpx.AsyncClient` with in-process ASGI transport — no live server needed.

---

## Persistent Directories

These directories are bind-mounted out of the container — they survive `make stop` / `make start`.

| Directory | Contents | Notes |
|---|---|---|
| `data/afd.db` | SQLite database (alerts, model registry, verdicts) | Reset with `make resetup` |
| `data/CIC-IDS2017/` | Dataset CSV files (~225 MB) | Populated by `make setup` |
| `models/` | Versioned model artifacts (one sub-folder per trained version) | Cleared by `make resetup` |
| `logs/` | `api.log`, `ui.log` | `make logs [api|ui]` to follow |

---

## Project Layout

```
config/       Pydantic Settings — single source of configuration
db/           Shared ORM models, DeclarativeBase, sync DB helpers
api/          FastAPI app — routers, DAO layer, schemas, jobs
alembic/      DB schema migrations
core/         Rule engine, classifier + detector protocols, model store
training/     Feature engineering, initial training, AE fine-tune pipelines
inference/    3-stage inference pipeline, CSV + PCAP adapters
tests/        Pytest suite
ui/           React 18 + Vite + Tailwind SPA (5 tabs)
data/         SQLite DB + CIC-IDS2017 dataset (not committed)
models/       Versioned model artifacts (not committed)
docs/         OpenAPI spec, architecture notes
```
