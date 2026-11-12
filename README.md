# Anomaly Flow Detector

3-stage NIDS pipeline: Rule Engine → XGBoost Classifier → MLP Autoencoder.

---

## Quick Start

> Requires Docker Desktop (running). Nothing else needed on your machine.

```bash
git clone https://github.com/hishamsharif/anomaly-flow-detector
cd anomaly-flow-detector
docker compose up --build
```

- API → `http://localhost:8000`
- UI  → `http://localhost:5173`

**Logs**

```bash
make logs        # all services
make logs-api    # API only
make logs-ui     # UI only
```

Re-running is safe — if the stack is already up, `make run` prints the URLs and exits.

---

## Prerequisites

### Docker path (examiner / CI)
| Requirement | Notes |
|---|---|
| Docker Desktop | daemon must be running (`docker version` to verify) |
| NVIDIA Container Toolkit | optional — `make run` applies GPU passthrough automatically when detected; plain `docker compose up --build` runs CPU-only without it |

### Local dev path
| Requirement | Notes |
|---|---|
| Miniconda / conda | manages the `afd_v2` Python env |
| Python 3.12 | inside the conda env — `make env` handles this |
| Node 20+ | only if running the UI outside Docker (`make run-ui`) |

---

## Local Dev (without Docker)

```bash
make env        # create afd_v2 env + install deps; auto-installs GPU torch if CUDA found
make dataset    # download CIC-IDS2017 (~225 MB); skips if already present
make run        # detects no Docker, prints instructions for the next two commands
make run-api    # apply migrations + start API on :8000 (separate terminal)
make run-ui     # start Vite dev server on :5173 (separate terminal)
```

All `make` targets use the `afd_v2` env directly — no `conda activate` needed.
Logs from both services are written to `logs/api.log` and `logs/ui.log` — follow with `make logs-api` / `make logs-ui`.

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

LOG_LEVEL=INFO
CORS_ORIGINS=["http://localhost:5173","http://localhost"]
```

---

## Make Targets

```
Setup
  make env          Create afd_v2 env + deps; auto-installs GPU torch if CUDA found
  make env-gpu      Manually upgrade torch to CUDA 12.1 (RTX A2000)
  make dataset      Download CIC-IDS2017 (~225 MB); skips if already present

Stack
  make run          Start stack — Docker if available, local guidance otherwise
  make stop         Stop the Docker stack
  make run-api      Run API locally on :8000, hot-reload (logs → logs/api.log)
  make run-ui       Start Vite dev server on :5173 (logs → logs/ui.log)

Logs (auto-detects Docker or local)
  make logs         Follow all logs
  make logs-api     Follow API logs
  make logs-ui      Follow UI logs

Verify
  make verify       Check API and UI are reachable
  make verify-api   Check API /health endpoint
  make verify-ui    Check UI is serving on :5173

Dev
  make test         Run pytest suite
  make lint         Ruff check + format check
  make migrate      Apply Alembic migrations
  make export-spec  Write docs/openapi.json from the live app
  make clean-models Delete versioned model artifacts under models/
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
  "images": { "level": "warn", "message": "Images not built yet", "detail": "Run make run once." }
}
```

### `GET /api/models/available-types`
Returns supported classifier and detector IDs with display labels.

### `GET /api/models`
Paginated list of trained model versions.

### `GET /api/models/active`
Currently promoted model, or `null`.

---

## Tests

```bash
make test
```

```
tests/test_health.py::test_health_returns_200                   PASSED
tests/test_health.py::test_health_shape                         PASSED
tests/test_health.py::test_health_db_ok_true                    PASSED
tests/test_models.py::test_available_types_has_expected_entries PASSED
tests/test_models.py::test_active_model_is_null_initially       PASSED
tests/test_models.py::test_model_list_is_empty_initially        PASSED
tests/test_preflight.py::test_preflight_returns_all_checks      PASSED
tests/test_preflight.py::test_preflight_check_shape             PASSED
```

Tests use `httpx.AsyncClient` with in-process ASGI transport — no live server needed.

---

## Persistent Directories

These directories are bind-mounted out of the container to the project root — they survive `make stop` / `make run` and can be inspected directly on the host.

| Directory | Contents | Access |
|---|---|---|
| `data/` | SQLite database (`afd.db`) + CIC-IDS2017 CSV files | read/write any time |
| `models/` | Versioned model artifacts (one sub-folder per trained version) | read/write any time |
| `logs/api.log` | Uvicorn request + application logs | `tail -f logs/api.log` or `make logs-api` |

> Training a new model writes artifacts to `models/<name>/` and updates `data/afd.db`.
> Deleting `data/afd.db` resets the model registry; deleting a `models/<name>/` folder removes that version's artifacts.

---

## Project Layout

```
api/          FastAPI app — routers, config, database, schemas
alembic/      Async migrations
core/         Classifier + Detector protocols and registries
training/     Feature engineering pipelines
inference/    PCAP / CSV adapters
runtime/      Host-environment probes (CUDA, Docker) and preflight checks
tests/        Pytest suite
ui/           React + Vite + Tailwind frontend
data/         SQLite DB and CIC-IDS2017 dataset (not committed)
models/       Versioned model artifacts (not committed)
```
