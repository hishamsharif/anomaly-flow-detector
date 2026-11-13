# Anomaly Flow Detector — V2 Specification

> **This file is the authoritative source of truth.** All code, tests, and documentation derive
> from it. Before changing behaviour, change the spec first. Drift between the spec and the
> running system is a bug.

**Version:** 2.0.0  
**Date:** 2026-11-02  
**Supersedes:** `dev/anomaly-net-flow/` (V1, CS7003NU A2 implementation)  
**Assessment:** CS7003NU A2 — Individual Coursework (criterion a + b)

---

## Table of Contents

1. [Purpose & Scope](#1-purpose--scope)
2. [Non-Goals](#2-non-goals)
3. [Domain Model](#3-domain-model)
4. [API Contract](#4-api-contract)
5. [Data Pipeline Contracts](#5-data-pipeline-contracts)
6. [UI Contract](#6-ui-contract)
7. [Configuration Contract](#7-configuration-contract)
8. [Tech Stack](#8-tech-stack)
9. [Folder Layout](#9-folder-layout)
10. [V1 Learnings](#10-v1-learnings)
11. [SDD Workflow for this Project](#11-sdd-workflow-for-this-project)

---

## 1. Purpose & Scope

**Anomaly Flow Detector (AFD)** is a network intrusion detection service for SOC analysts.
It scores network flows through a 3-stage pipeline — Rule Engine → XGBoost Classifier → MLP
Autoencoder — producing a verdict of `SAFE`, `KNOWN_ATTACK`, or `NOVEL_ATTACK` per flow.
Analyst verdicts on flagged traffic feed back into the anomaly detector so the system improves
over time without a full manual retrain.

**Target user:** SOC analyst (or demo operator running the system locally).

**Core value proposition:**

| Problem | AFD's answer |
|---------|-------------|
| Known-critical CVE signatures and trusted internal sources require deterministic certainty — ML can misclassify confirmed threats or silently override business allowlists. | **Stage 1 — Rule Engine:** Exits immediately as `KNOWN_ATTACK` (CVE signatures, known-malicious IPs) or `SAFE` (allowlisted assets, trusted internal ranges); only genuinely ambiguous flows reach ML. Rules always take precedence — add or update rules at any time to enforce correct behaviour without retraining. |
| SOC analysts are overwhelmed by repetitive known-attack alerts — patterns the classifier can handle automatically. | **Stage 2 — XGBoost Classifier:** Classifies known attack patterns automatically and exits immediately as `KNOWN_ATTACK`, freeing analysts for genuinely ambiguous traffic. |
| Classifiers trained on known-attack labels silently mark novel/zero-day attacks as `BENIGN` — a coverage gap no labelled training set can close. | **Stage 3 — MLP Autoencoder:** Trained on benign-only traffic; detects deviation from normality regardless of attack type, surfacing `NOVEL_ATTACK` flows the classifier cannot see. |
| Novel/zero-day attack judgments require human expertise, but today that expertise is discarded — the system never learns from it and degrades as the threat landscape drifts. | **Stage 3 — HITL + Periodic AE retrain:** Analyst verdicts (`confirmed` / `override`) accumulate as fine-tune training signal; once the threshold is reached, the MLP-AE is retrained on analyst-labelled benign flows, keeping the benign baseline calibrated as threats evolve. |

**Scope:**
- CIC-IDS2017 dataset (2.8M flows, 15 attack classes) as the training source
- Batch CSV/PCAP inference — upload a file, get scored alerts back
- Local deployment (laptop or single server) — no cloud dependency
- Single-user, single-tenant

---

## 2. Non-Goals

Explicitly out of scope for V2. Listing these prevents scope creep during implementation.

| Out of scope | Reason / future hook |
|-------------|---------------------|
| Real-time packet capture (streaming NIDS) | Batch CSV/PCAP is sufficient; architecture allows adding a stream adapter in a future version |
| Multi-user / RBAC / TLS | Single-user local deployment only; adding auth/authz is a deployment concern, not a model concern |
| Celery, Redis, or external task queues | In-process threading is sufficient and adds zero external dependencies (ADR-07) |
| Automated ML experiment tracking (MLflow, W&B) | Out of scope; metrics are stored in the DB and surfaced in the UI |

---

## 3. Domain Model

### 3.1 Entities

**Flow**
A single network connection record. Inputs arrive as rows in a CSV or extracted from a PCAP.
- Has up to 78 raw CIC-IDS2017 feature columns
- A flow that fails feature validation is rejected with a per-row error, not a bulk failure
- Must pass feature shape validation against the active model's `feature_names.json` before entering the pipeline (see I3)

**Alert**
The output of scoring one Flow through the 3-stage pipeline.
- Consensus verdict: `SAFE | KNOWN_ATTACK | NOVEL_ATTACK`
- Severity — derived from verdict + confidence at inference time:

  | Verdict | Condition | Severity |
  |---------|-----------|----------|
  | `SAFE` | — | `low` |
  | `KNOWN_ATTACK` | `classifier_confidence ≥ 0.8` | `high` |
  | `KNOWN_ATTACK` | `classifier_confidence < 0.8` | `medium` |
  | `NOVEL_ATTACK` | `ae_score` in top quartile of val-benign errors | `critical` |
  | `NOVEL_ATTACK` | otherwise | `high` |

- Append-only once created — only `analyst_verdict` and `feedback_at` may be updated (see I2)
- Analyst verdict lifecycle:

  | Verdict | Retraining-eligible | Counts toward K | Downstream effect |
  |---------|--------------------|-----------------|--------------------|
  | `confirmed` | Yes | Yes | Counts toward AE retrain threshold |
  | `override` | Yes | Yes | Counts toward AE retrain threshold |
  | `uncertain` | No | No | Stored only — audit signal for borderline flows |
  | `escalate` | No | No | Stored + UI badge; notification = future work |
  | `null` | No | No | No verdict submitted yet |

**ModelVersion**
A trained (Classifier, AnomalyDetector) pair saved to an artifact directory.
- Status lifecycle: `training → active | failed`; `active → archived` when superseded
- At most one ModelVersion has `status = active` at any time (see I1)
- If artifact files are missing on startup, status transitions to `failed`
- XGBoost classifier weights are **never updated** after initial training (frozen)

**Verdict**
An analyst's judgment on one Alert.
- Records: which alert, what verdict, what label (if override), when
- Only `confirmed` and `override` are retraining-eligible and count toward `FINE_TUNE_K_THRESHOLD`

### 3.2 Invariants

| ID | Invariant | How enforced |
|----|-----------|-------------|
| I1 | At most one ModelVersion is active at any time | DB UNIQUE partial index + promote endpoint |
| I2 | Alerts are append-only (except verdict fields) | No UPDATE on non-verdict columns in CRUD layer |
| I3 | Inference validates feature shape against active model's `feature_names.json` | `csv_adapter.py` shape check before pipeline entry |
| I4 | AE fine-tune requires `eligible_verdicts ≥ FINE_TUNE_K_THRESHOLD` | Checked in `POST /api/training/start` |
| I5 | XGBoost artifacts copied unchanged during AE fine-tune | `fine_tune.py` copies `classifier.joblib` without loading it |
| I6 | Artifact directory write is atomic | Write to `models/tmp_<run_id>/`, rename to `models/<name>/` on success |

### 3.3 3-Stage Pipeline State Machine

```
Flow in
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 1: Rule Engine                                   │
│  (deterministic — always evaluated first, always        │
│   takes precedence; add/update rules without retraining)│
└─────────────────────────────────────────────────────────┘
  │──► KNOWN_ATTACK (exit immediately — CVE signatures, known-malicious IPs, high-risk overrides)
  │──► SAFE         (exit immediately — trusted internal sources, allowlisted assets)
  │
  │ (ambiguous — neither definitively safe nor definitively known-dangerous)
  ▼
┌─────────────────────┐
│  Stage 2: XGBoost   │── verdict: KNOWN_ATTACK  (confidence ≥ CLF_CONFIDENCE_THRESHOLD)
│  Classifier         │── verdict: BENIGN        (passes through; confidence < threshold)
└─────────────────────┘
  │
  │ (all ambiguous flows proceed — Stage 3 always runs to record ae_score)
  ▼
┌─────────────────────┐
│  Stage 3: MLP-AE    │── ae_score computed on every alert
│  Anomaly Detector   │── finalises verdict only when Stage 2 = BENIGN:
└─────────────────────┘       NOVEL_ATTACK  (ae_score > threshold)
                               SAFE          (ae_score ≤ threshold)
```

**Precedence rule:** Stage 1 verdict always wins. Within the ML stages, Stage 2 takes precedence over Stage 3 — if Stage 2 fires `KNOWN_ATTACK`, Stage 3 still runs (for the anomaly score), but cannot change the verdict.

---

## 4. API Contract

**Base URL:** `http://localhost:8000`  
**Content-type:** `application/json` (except multipart uploads)  
**Errors:** RFC 7807 Problem Details — `{"type", "title", "detail", "status"}`

### 4.1 Health

```
GET /health
→ 200  {"status": "ok", "model_loaded": bool, "db_ok": bool}
```

### 4.2 Models

```
GET /api/models
→ 200  {"items": ModelVersion[], "total": int}

GET /api/models/active
→ 200  ModelVersion | null

POST /api/models/{id}/promote
→ 200  ModelVersion
→ 404  model id not found
→ 409  already active, or status is "training"

GET /api/models/available-types
→ 200  {
    "classifiers": [{"id": str, "label": str}],
    "detectors":   [{"id": str, "label": str}]
  }
```

**ModelVersion schema:**
```json
{
  "id": 1,
  "name": "XGBoost + MLP-AE — v1",
  "classifier_type": "xgboost",
  "detector_type": "mlp_ae",
  "status": "active",
  "metrics": {
    "f1": 0.818,
    "far": 0.00016,
    "amr": 0.735,
    "pr_auc": 0.818,
    "known_attack_f1": 0.052,
    "n_features_final": 47,
    "ae_threshold": 0.003667,
    "training_duration_s": 888,
    "train_rows": 2589848,
    "val_rows": 191033,
    "test_rows": 512212
  },
  "created_at": "2026-11-01T10:00:00Z"
}
```

### 4.3 Training

```
POST /api/training/start
  body: {"classifier_type": str, "detector_type": str, "mode": "initial" | "ae_retrain"}
→ 202  {"run_id": str}
→ 409  mode="ae_retrain" and no active model
→ 409  mode="ae_retrain" and eligible_verdicts < FINE_TUNE_K_THRESHOLD
→ 409  a training job is already running

GET /api/training/{run_id}/events           SSE stream
  event: progress   data: "Loading CIC-IDS2017 …"
  event: step       data: "45"              (integer percent 0–100)
  event: done       data: "Training complete — XGBoost + MLP-AE v1 active"
  event: error      data: "…error message…"
  event: heartbeat  data: "ping"            (every 30s keep-alive)
→ 200  text/event-stream
→ 404  run_id not found

GET /api/training/{run_id}/status           poll fallback (SSE not available)
→ 200  {"run_id": str, "pct": int, "mode": str, "started_at": str}
→ 404  run_id not found
```

### 4.4 Inference

```
POST /api/predict/batch
  multipart: file (CSV or PCAP, ≤ MAX_UPLOAD_MB)
→ 200  BatchResult                          (sync, ≤ SYNC_MAX_ROWS rows)
→ 202  {"job_id": str, "flow_count": int}   (async, > SYNC_MAX_ROWS)
→ 400  unrecognised file format
→ 409  no active model
→ 413  file exceeds MAX_UPLOAD_MB

GET /api/predict/{job_id}/status         (async jobs only)
→ 200  {"job_id": str, "pct": int, "done": bool}
→ 404  job_id not found

GET /api/predict/{job_id}/results        (async jobs only)
→ 200  BatchResult                       (available when done=true)
→ 404  job_id not found
→ 409  job not complete yet
```

**BatchResult schema:**
```json
{
  "alerts": [Alert],
  "rejected_rows": [{"row": 3, "reason": "missing feature: Flow Duration"}],
  "summary": {
    "total": 1000, "safe": 700, "known_attack": 250,
    "novel_attack": 40, "rejected": 10
  }
}
```

**Alert schema:**
```json
{
  "id": 42,
  "flow_id": "uuid-or-row-index",
  "consensus": "NOVEL_ATTACK",
  "severity": "critical",
  "predicted_label": null,
  "classifier_confidence": 0.43,
  "ae_score": 0.0089,
  "ae_threshold": 0.003667,
  "top_features": [
    {"name": "Flow Duration", "contribution": 0.41},
    {"name": "Bwd Packet Length Max", "contribution": 0.28}
  ],
  "analyst_verdict": null,
  "feedback_at": null,
  "timestamp": "2026-11-02T14:00:00Z"
}
```

### 4.5 Alerts

```
GET /api/alerts
  query: page=1 & size=50 & verdict=null & consensus=NOVEL_ATTACK
→ 200  {"items": Alert[], "total": int, "page": int, "size": int}

POST /api/alerts/{id}/verdict
  body: {
    "verdict": "confirmed" | "override" | "uncertain" | "escalate",
    "label": str | null      (required when verdict = "override")
  }
→ 200  Alert
→ 404  alert not found
→ 409  verdict already set (immutable after first submission)

GET /api/alerts/verdict-count
→ 200  {"eligible": 23, "threshold": 50, "unlocked": false}
```

### 4.6 Dashboard

```
GET /api/dashboard/summary
→ 200  {
    "total_24h":           int,   // alerts inserted in the last 24 h
    "safe_24h":            int,
    "known_attack_24h":    int,
    "novel_attack_24h":    int,
    "pending_verdict_24h": int,   // alerts with analyst_verdict = null in last 24 h
    "by_severity": {
      "low":      int,
      "medium":   int,
      "high":     int,
      "critical": int             // severity breakdown — all-time cumulative
    },
    "verdict_eligible":    int,   // all-time cumulative analyst verdicts (confirmed + override)
    "verdict_threshold":   int,   // K threshold to unlock AE fine-tune
    "retrain_unlocked":    bool   // verdict_eligible >= verdict_threshold
  }
```

### 4.7 About

```
GET /api/about
→ 200  {
    "version": str,
    "active_model": ModelVersion | null,
    "config": {key: str}   (non-sensitive config vars only)
  }
```

---

## 5. Data Pipeline Contracts

### 5.0 Data Understanding — EDA

Before any model is trained, an exploratory analysis of CIC-IDS2017 must be produced and
saved to `training/eda/eda_cicids2017.ipynb`. The notebook is a living artefact: it is
produced once, committed, and updated when the dataset understanding changes. It directly
informs — and is cited in — the A2 report.

**Required analyses:**

| Analysis | Purpose |
|---|---|
| Flow volume by hour and day | Confirm diurnal business-hour cycle; validate that Mon–Thu and Fri-AM share the same temporal pattern (ADR-02 justification) |
| Attack-type distribution by day | Show that Fri-PM contains DDoS + PortScan exclusively — the evidence for why Fri-PM is a genuine novel-attack test set |
| Label imbalance per split | BENIGN:attack ratio for train / val / test — quantifies why SMOTE is required (see ADR-14) |
| Feature distribution over time (Mon→Fri) | Rolling mean / std of key features — detects concept drift between capture days |
| Benign-traffic seasonality (Fri-AM vs Mon–Thu) | Validates that val-benign (used for AE threshold calibration) is representative of training-benign; misrepresentation would bias the p95 threshold |
| Correlation heatmap (pre- and post-pruning) | Visualises the effect of the correlation filter (§5.1 Step 2); confirms high-correlation feature pairs are removed |
| Feature variance ranking | Shows which features VarianceThreshold (§5.1 Step 1) drops; confirms no informative feature is lost |

**Outputs consumed by the pipeline:**
- Findings feed into the SMOTE cap choice (`SMOTE_MAX_PER_CLASS`) and threshold percentile (`AE_ANOMALY_THRESHOLD_PERCENTILE`)
- Label imbalance ratios are cited in §12 ADR-14 (SMOTE justification)
- Benign seasonality analysis is cited in §12 ADR-05 (threshold calibration)

### 5.1 Initial Training Pipeline

**Inputs:**
- CIC-IDS2017 CSV files in `BUNDLED_DATASET_DIR`
- `classifier_type` and `detector_type` strings

**Outputs — artifact directory (atomic write):**
```
models/<name>/
  classifier.joblib       fitted classifier (XGBoost or RF)
  detector.pt             MLP-AE state dict  (or detector.joblib for IF)
  clf_scaler.joblib       scaler for classifier data path
  det_scaler.joblib       scaler for detector data path
  variance_mask.npy       boolean mask from VarianceThreshold step
  corr_mask.npy           boolean mask from correlation pruning step
  selector_mask.npy       boolean mask from feature selection step (all-True when FEATURE_SELECTION_K=0)
  feature_names.json      ordered list of final selected feature names
  metadata.json           metrics + full config snapshot at training time
```

**Behavioral guarantees:**
- Split is deterministic and date-based: Mon–Thu → train, Fri-AM → val, Fri-PM → test
- AE trains on benign-only rows from the training split
- AE threshold = p[`AE_ANOMALY_THRESHOLD_PERCENTILE`] of val-benign reconstruction errors
- SMOTE caps at `SMOTE_MAX_PER_CLASS` samples per minority class
- On pipeline success, the new version is **automatically promoted** to active
- On pipeline failure, the partial artifact dir is removed and version status = `failed`
- Feature engineering config is **snapshotted into `metadata.json`** — re-running with different
  env vars produces a new, independent version (no silent config drift)

**Feature engineering steps (Steps 1–3 produce a shared reduced feature set; Steps 4a/4b are separate scaling paths for each model):**
```
Step 1  — VarianceThreshold(threshold=VARIANCE_THRESHOLD)
Step 2  — Drop one feature from each pair with |Pearson r| > CORRELATION_THRESHOLD
Step 2b — SMOTE (training split only): oversample minority attack classes up to SMOTE_MAX_PER_CLASS
           val and test splits are never resampled (ADR-14)
Step 3  — Optional: SelectKBest(k=FEATURE_SELECTION_K, score_func=mutual_info)
           or RF-RFE (FEATURE_SELECTION_METHOD=rfe)
           or skip (FEATURE_SELECTION_METHOD=none / k=0)
Step 4a — clf_scaler (determined by classifier PreprocessingSpec.preferred_scalers)
Step 4b — det_scaler (determined by detector PreprocessingSpec.preferred_scalers)
```

### 5.2 Inference Pipeline (per flow)

```
Input: dict of {feature_name: value} for one network flow

Stage 1 — Rule Engine
  Apply deterministic rules (evaluated first; always takes precedence over ML)
  → emit KNOWN_ATTACK and exit if rule matches a CVE signature, known-malicious IP, or high-risk override
  → emit SAFE and exit if source is an allowlisted asset or trusted internal range
  → pass to Stage 2 if ambiguous (no rule matched either end of the certainty spectrum)

Stage 2 — Known-Attack Classifier (XGBoost)
  Apply variance_mask → corr_mask → selector_mask → clf_scaler
  Predict: label, confidence
  → record KNOWN_ATTACK verdict if label ≠ BENIGN and confidence ≥ CLF_CONFIDENCE_THRESHOLD
  → record BENIGN verdict otherwise (label = BENIGN, or confidence < CLF_CONFIDENCE_THRESHOLD)

Stage 3 — MLP Anomaly Detector (always runs; ae_score populated on every alert)
  Apply variance_mask → corr_mask → selector_mask → det_scaler
  Compute reconstruction error (ae_score)
  → record NOVEL_ATTACK if ae_score > ae_threshold
  → record SAFE otherwise

Consensus:
  → final verdict = KNOWN_ATTACK if Stage 2 recorded KNOWN_ATTACK
  → final verdict = Stage 3 result (NOVEL_ATTACK or SAFE) otherwise
```

**Explainability output (always computed, included in Alert):**
- Classifier: top N features by `feature_importances_` × input value
- AE: top N features by per-feature reconstruction error `(x_i − x̂_i)²`

### 5.3 AE Fine-Tune Pipeline

**Pre-condition:** `eligible_verdict_count ≥ FINE_TUNE_K_THRESHOLD`

**Process:**
1. Fetch all confirmed benign flows — filter: `(analyst_verdict = 'confirmed' AND consensus = 'SAFE') OR analyst_verdict = 'override'`
   (`confirmed` on a SAFE alert = analyst agreed the flow is benign; `override` on any attack alert = analyst corrected the verdict to benign)
   (Savepoint tracking — fetching only flows since the last fine-tune — is a V3 enhancement)
2. Combine with `AE_REPLAY_BUFFER_FRACTION` × historical benign training data
3. Load previous AE checkpoint; freeze encoder weights for `AE_FREEZE_EPOCHS` epochs
4. Fine-tune for remaining epochs with encoder unfrozen
5. Recompute AE threshold on val-benign flows
6. Regression gate: compare new FAR and recall vs. previous version
   - If `far_increase > FAR_GATE_MAX` or `recall_drop > RECALL_GATE_MAX` → emit WARN
   - Gate is **advisory** — it warns, it does not block promotion (V2; hard block = future work)
7. Copy `classifier.joblib` unchanged from previous version (frozen)
8. Write new artifact directory atomically; promote to active (proceeds regardless of regression gate result)

---

## 6. UI Contract

**SPA with 5 tabs — React 18 + TypeScript + Vite 5 + Tailwind:**

Tab order: **Dashboard → Dataset → Training → Inference → About**

### Dashboard
- Active model name + status badge
- Alert summary for the last 24 hours: total / SAFE / KNOWN_ATTACK / NOVEL_ATTACK
- Pending-verdict count for the last 24 hours
- Severity breakdown (all-time): low / medium / high / critical
- Verdict progress bar: `eligible / threshold` with "Retrain AE" button (enabled when `unlocked=true`)

### Training
- Classifier dropdown + Detector dropdown (populated from `GET /api/models/available-types`)
- "Start Training" button — disabled while any job is running
- SSE progress bar (0–100%) with live log lines below it
- Model version history table: name / status / F1 / FAR / Miss Rate (AMR) ↓ / created_at / Promote button

### Inference
- Drag-and-drop or click-to-upload for CSV or PCAP
- After upload: summary card (total / safe / known / novel / rejected)
- Results table (paginated, 50 rows): flow_id / consensus / severity / label / confidence / ae_score
- Click a row → expand inline: top features, verdict buttons (Confirm / Override / Uncertain / Escalate)
- Verdict buttons disabled once a verdict is set

### Dataset
- CIC-IDS2017 overview: 2.83M flows, 15 attack classes, 5-day capture
- Class distribution chart (log scale — illustrates BENIGN dominance)
- Attack volume by capture day (stacked bar — shows Fri-PM DDoS/PortScan concentration)
- Feature correlation heatmap (20 × 20 — motivates the correlation-filter step in §5.1)

### About
- System overview: what AFD does and why
- 3-stage pipeline diagram (text/ASCII is fine)
- Tech stack list

**Behavioral guarantees:**
- Training tab shows SSE-driven progress (not polling)
- Verdict submission disables the buttons immediately (optimistic UI update)
- Alert table reflects verdict state without a page reload
- All API errors display an inline error message; no silent failures

---

## 7. Configuration Contract

All settings have working defaults. Override via `.env` file or environment variables.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DATABASE_URL` | str | `sqlite+aiosqlite:///./data/afd.db` | SQLite DB path |
| `BUNDLED_DATASET_DIR` | str | `data/CIC-IDS2017` | CIC-IDS2017 CSV folder |
| `MODEL_ARTIFACT_DIR` | str | `models` | Trained model artifacts root |
| `FINE_TUNE_K_THRESHOLD` | int | `50` | Eligible verdicts to unlock AE retrain |
| `AE_ANOMALY_THRESHOLD_PERCENTILE` | float | `95.0` | p-tile of val-benign errors for threshold |
| `AE_REPLAY_BUFFER_FRACTION` | float | `0.10` | Historical benign fraction replayed during retrain |
| `AE_FREEZE_EPOCHS` | int | `5` | Frozen-encoder warm-start epochs |
| `DEVICE` | str | `auto` | `"auto"` = detect at startup via `torch.cuda.is_available()`; override with `"cuda"` or `"cpu"` |
| `FAR_GATE_MAX` | float | `0.05` | Max FAR increase allowed before WARN |
| `RECALL_GATE_MAX` | float | `0.05` | Max recall drop allowed before WARN |
| `SMOTE_MAX_PER_CLASS` | int | `50000` | SMOTE cap per minority class |
| `VARIANCE_THRESHOLD` | float | `0.0` | Features with variance ≤ this value are dropped; 0.0 = constant features only |
| `CORRELATION_THRESHOLD` | float | `0.95` | Pearson \|r\| pruning threshold |
| `FEATURE_SELECTION_K` | int | `0` | Top-K features (0 = keep all) |
| `FEATURE_SELECTION_METHOD` | str | `mi` | `mi` \| `rfe` \| `none` |
| `CLF_CONFIDENCE_THRESHOLD` | float | `0.5` | Min confidence to emit KNOWN_ATTACK |
| `MAX_UPLOAD_MB` | int | `100` | Max inference file size |
| `SYNC_MAX_ROWS` | int | `5000` | Max rows for synchronous inference (else async) |
| `CORS_ORIGINS` | list[str] | `["http://localhost:5173", "http://localhost"]` | Allowed CORS origins |
| `LOG_LEVEL` | str | `INFO` | Logging level |

**Rule:** Config is defined once, in `config/settings.py` (Pydantic Settings). It is never duplicated
in `docker-compose.yml` environment blocks — Docker passes through env vars from the host or
`env_file`, never re-declares them.

---

## 8. Tech Stack

Locked choices with rationale. These do not change without a spec update.

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Language | Python 3.12 | type-safe, ecosystem, GPU support |
| API framework | FastAPI | async, auto-OpenAPI, Pydantic v2 native |
| DB ORM | SQLAlchemy 2.x async + aiosqlite | async-native, swappable to PG |
| DB migrations | **Alembic** | versioned migrations; no startup ALTER TABLE hacks |
| ML — classifier | XGBoost (GPU) | matches A1 comparison; `device='cuda'` |
| ML — detector | PyTorch MLP-AE (GPU) | flexible architecture; GPU support |
| Feature eng | scikit-learn + imbalanced-learn | VT, correlation, SMOTE |
| Data loading | DuckDB | fast CSV scan without full pandas load |
| UI | React 18 + TypeScript + Vite 5 + Tailwind | fast builds, type-safe, minimal config |
| UI routing | React Router v6 | standard SPA routing |
| Containers | Docker Compose 2 services: `api` + `ui` (nginx) | one-command demo setup |
| Dev commands | **Makefile** | replaces `ops.sh` + `setup.sh`; universal, no bash dialect quirks |
| Tests | pytest + httpx (async) | from day 1; not an afterthought |
| Code style | ruff (lint + format) | single tool, fast |

**Swappable via registry (one line to add):**
- Classifier: `xgboost` (default) | `random_forest`
- Detector: `mlp_ae` (default) | `isolation_forest`

**GPU:** RTX A2000 8GB (host). Device resolved at startup via the `DEVICE` config var
(`"auto"` → `torch.cuda.is_available()` → `"cuda"` or `"cpu"`). Docker GPU still requires
the CUDA torch wheel (`--build-arg TORCH_BUILD_TYPE=cu121` rebuild) and
`nvidia-container-toolkit` on the host; on a plain Docker build or CPU-only host the
`"auto"` path selects CPU automatically.

---

## 9. Folder Layout

```
anomaly-flow-detector/
├── SPEC.md                         ← this file (authoritative contract)
├── Makefile                        ← make setup | resetup | start | stop | migrate | test | lint
├── config/
│   └── settings.py                 ← Settings (pydantic-settings); ONLY place config lives
├── db/
│   ├── base.py                     ← shared DeclarativeBase (no FastAPI dependency)
│   ├── models.py                   ← SQLAlchemy ORM (derived from schemas.py)
│   └── utils/
│       └── sync_ops.py             ← sync DB helpers for training pipelines (zero api/ imports — SPEC §9)
├── api/
│   ├── main.py                     ← FastAPI app, lifespan, CORS, router includes
│   ├── schemas.py                  ← ALL Pydantic request/response schemas ← SDD contract layer
│   ├── jobs.py                     ← ThreadPoolExecutor + SSE bridge (~147 lines: training jobs + predict jobs)
│   ├── dao/
│   │   ├── db_conn.py              ← async engine, AsyncSessionLocal, get_db FastAPI dependency
│   │   ├── alerts.py               ← alert CRUD (sync for thread workers, async for routers)
│   │   ├── model_versions.py       ← model-version CRUD
│   │   └── dashboard.py            ← dashboard summary query
│   └── routers/
│       ├── health.py
│       ├── models.py
│       ├── training.py
│       ├── inference.py
│       └── alerts.py
├── core/
│   ├── base.py                     ← BaseClassifier + BaseAnomalyDetector protocols
│   ├── rule_engine.py              ← deterministic rule evaluation; always takes precedence over ML
│   ├── registry.py                 ← CLASSIFIERS / ANOMALY_DETECTORS dicts (one line to add)
│   ├── store.py                    ← ModelStore singleton (in-memory active model)
│   ├── classifiers/
│   │   ├── xgboost_clf.py
│   │   └── random_forest_clf.py
│   └── detectors/
│       ├── mlp_ae.py
│       └── isolation_forest.py
├── training/
│   ├── train_offline.py            ← offline initial training entry point (`make train`)
│   ├── pipelines/
│   │   ├── initial.py              ← CIC-IDS2017 → artifacts; no api.config import
│   │   └── fine_tune.py            ← AE-only retrain; no api.config import
│   ├── feature_eng/
│   │   ├── variance_filter.py      ← drop features ≤ VARIANCE_THRESHOLD
│   │   ├── correlation_filter.py   ← drop correlated features ≥ CORR_THRESHOLD
│   │   ├── selector.py             ← MI / RFE selector (saves variance_mask, corr_mask)
│   │   └── pipeline.py             ← chains the three steps above
│   └── data_loader.py              ← DuckDB CSV scan (CIC-IDS2017 loading only)
├── inference/
│   ├── pipeline.py                 ← 3-stage runner: rules → classifier → detector → consensus
│   ├── inference_offline.py        ← standalone CLI examiner (no server required; `python inference/inference_offline.py --csv ...`)
│   └── adapters/
│       ├── csv_adapter.py          ← CSV → feature DataFrame (handles CIC-IDS2017 column names)
│       └── pcap_adapter.py         ← PCAP → NFStream flow extraction → feature DataFrame
├── runtime/
│   ├── probe.py                    ← CUDA, Docker, image inventory probes; returns HostEnvironment
│   ├── manifest.py                 ← HostEnvironment singleton (resolved once at startup)
│   └── preflight.py                ← structured check results for GET /api/preflight
├── alembic/                        ← DB migrations (replaces all startup ALTER TABLE code)
│   ├── env.py
│   └── versions/
├── tests/
│   ├── conftest.py                 ← pytest fixtures: async client, in-memory DB, mock model
│   ├── test_health.py
│   ├── test_models.py
│   ├── test_training.py
│   ├── test_inference_offline.py
│   └── test_alerts.py
├── ui/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── DashboardPage.tsx
│   │   │   ├── TrainingPage.tsx
│   │   │   ├── InferencePage.tsx
│   │   │   └── AboutPage.tsx
│   │   ├── components/             ← shared UI components
│   │   ├── api/
│   │   │   └── client.ts           ← typed API client (derived from schemas.py)
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json
│   └── vite.config.ts
├── docs/
│   ├── openapi.json                ← exported from FastAPI (`make export-spec`)
│   └── architecture.md            ← human-readable architecture doc
├── data/
│   └── CIC-IDS2017/               ← CSV files (.gitignored)
├── models/                         ← artifact directories (.gitignored except pinned version)
├── alembic.ini
├── Dockerfile.api
├── Dockerfile.ui
├── docker-compose.yml
├── requirements.txt                ← single file; no requirements.prod.txt split
├── .env.demo
└── .gitignore
```

**Key structural rules (enforced by convention):**
- `training/` never imports from `api/` — it receives config as plain arguments
- `core/` has zero HTTP, DB, or file-system dependencies
- `runtime/` owns all host-environment discovery; `core/`, `training/`, and `inference/` receive resolved config as parameters — never probe the environment directly
- `api/schemas.py` is the only place Pydantic request/response models are defined
- All ORM models in `db/models.py` mirror schema fields (schemas are the source of truth)

---

## 10. V1 Learnings

This section documents what was learned from V1 (`dev/anomaly-net-flow/`) and drives V2 decisions.

### What worked well — carry forward

| Component | What was good |
|-----------|---------------|
| 3-stage pipeline (Rule → XGBoost → MLP-AE) | Scientifically correct; maps directly to A1 conclusions |
| Protocol-based model interfaces + registry | `BaseClassifier`, `BaseAnomalyDetector` protocols; one line to add a new model |
| Pydantic Settings with `.env` defaults | Zero-config local run worked perfectly |
| SSE progress streaming for training | Good UX; real-time feedback during 15-min training runs |
| Docker Compose 2-service (api + ui/nginx) | Single-command setup; worked reliably |
| Day-based temporal split (Mon–Thu / Fri-AM / Fri-PM) | Scientifically correct for temporal data; novel-attack test set by design |
| HITL feedback loop (analyst verdicts → threshold → retrain) | Core differentiator; K=50 threshold + encoder freeze + replay buffer worked |
| `inference/inference_offline.py` standalone examiner entry point | 28k flows/s on GPU; no server required for demos |
| `core/` with zero HTTP dependency | Kept ML code portable and testable; V2 strengthens this to zero HTTP, DB, and file-system dependencies |
| `DuckDB` for CSV loading | 10× faster than pandas for 2.8M row scan |

### What went wrong — fix in V2

| V1 Problem | Root cause | V2 Fix |
|------------|-----------|--------|
| `ALTER TABLE` in `main.py` startup | Schema changed after initial deploy; no migration tool | **Alembic** from day 1 |
| `training/` imported from `api/config` | Convenient shortcut that created coupling | Training functions receive config as plain args |
| `LabelledFlow.features_json` (entire 78-col JSON per flow) | Over-engineered; only label + flow_id needed for retraining | Analyst verdict stored as columns on `Alert` (`analyst_verdict`, `override_label`, `feedback_at`); `features_json` retained on `Alert` for HITL replay — no separate verdicts table (§3.1 is the authoritative schema) |
| `FeatureSchemaVersion` table (mostly NULL FKs) | Over-abstracted for a single-dataset app | Dropped; feature schema lives in `metadata.json` artifact |
| `celery_task_id` field in `TrainingRun` | Celery was tried then abandoned; field never cleaned up | Field removed |
| `requirements.prod.txt` alongside `requirements.txt` | Two files drifted; unclear which to use | Single `requirements.txt` |
| CORS defined in both `config.py` and `docker-compose.yml` env block | CORS origins differ between local and Docker | Config.py only; docker-compose uses `env_file` |
| `api/tasks.py` — 414 lines (async/thread bridge) | Grew organically with each feature | `api/jobs.py` — clean rewrite (~147 lines: handles both training SSE jobs and async predict jobs) |
| `inference/explainers/` — empty stub dir | Placeholder for future work that never came | Create dirs only when files go in them |
| `api/routers/pipeline.py` — unclear purpose | Added during a refactor, never clarified | Dropped |
| Scattered Pydantic schemas across routers | No central contract; drift between routers | `api/schemas.py` — single source |
| No tests | Tests deferred until "after it works" | `tests/` with pytest fixtures from day 1 |
| V9 models v1–v8 artifact dirs accumulate | No artifact lifecycle management | Gitignore all except pinned examiner version |
| About page text stale ("Incremental Fine-tune") | UI copy not tied to spec | UI copy derived from spec language |

---

## 11. SDD Workflow for this Project

Based on the [InfoQ article "Spec Driven Development: When Architecture Becomes Executable"](https://www.infoq.com/articles/spec-driven-development/) (Griffin & Carroll, Jan 2026), SDD elevates the specification above code as the system's primary source of truth. The spec governs what gets generated, what gets tested, and what the runtime is allowed to do. Generated artifacts are disposable — if the spec changes, they are regenerated; if they are deleted, nothing is lost.

**The five SDD layers, mapped to this project:**

| Layer | AFD artifact | Role |
|-------|-------------|------|
| **Specification** | `SPEC.md` (this file) | Authoritative intent — humans write and govern; version-controlled and peer-reviewed |
| **Generation** | `api/schemas.py` → FastAPI | Pydantic models = machine-executable contract; FastAPI generates `docs/openapi.json` from them |
| **Artifacts** | `docs/openapi.json` | Regenerable output — recreated by `make export-spec`; committed to `docs/` |
| **Validation** | `tests/` | Each test asserts one behavioral guarantee from §4 or §5; a failing test is a conformance failure |
| **Runtime** | FastAPI app + `inference/pipeline.py` | Shape is entirely constrained by the layers above |

**The workflow for any change:**

1. **Update SPEC.md first** — describe what must be true (not how)
2. **Update `api/schemas.py`** to reflect the new contract
3. **Write a failing test** that asserts the new behavior
4. **Implement** until the test passes
5. **Run `make export-spec`** — regenerates `docs/openapi.json` from `api/schemas.py` via FastAPI's `app.openapi()`
   (`ui/src/api/client.ts` is hand-maintained in sync with `api/schemas.py` — not auto-generated)

**Drift = bug.** Drift is any divergence between declared intent (SPEC.md) and observed system behavior. Examples:
- API returns a field not in `api/schemas.py`
- `inference/pipeline.py` applies a consensus rule not described in §5.2
- A test asserts behavior not described in SPEC.md
- A schema field type changes in code without a SPEC.md commit

Fix the spec or fix the code — never both simultaneously without an explicit spec-update commit. `docs/openapi.json` is a generated artifact; if it diverges from `api/schemas.py`, run `make export-spec` — do not patch it by hand. `ui/src/api/client.ts` is hand-maintained in sync with `api/schemas.py` — treat any divergence between them as a conformance bug. Without continuous conformance checking, SDD collapses back into documentation-driven development.

**Breaking-change gate:** Any change that removes a field, changes a type, or alters a behavioral guarantee (verdict values, error codes, threshold semantics) requires a SPEC.md commit before any code change. That commit is the approval gate.

**What this buys for A2:**
- §4 (API Contract) directly satisfies Assessment criterion (a): functional specification
- §3 + §9 (Domain Model + Folder Layout) satisfies criterion (b): software architecture
- The spec is the living document; the `.docx` report is an export of it

---

## 12. Architectural & Design Decisions

Each entry is an Architecture Decision Record (ADR) — the decision, why it was made,
what was considered, and what the trade-off is. These are binding: changing them requires
updating this section first.

---

### ADR-01 — 3-stage pipeline (Rule Engine → Classifier → AE)

**Decision:** Use a sequential 3-stage pipeline rather than a single end-to-end model.

**Rationale:**
- Each stage addresses a different threat class: deterministic rules → known attacks →
  novel/zero-day. Defence in depth.
- Classifier (XGBoost) is trained on labelled data — high precision on *known* attack types.
- AE is trained on benign-only — high recall on *unknown* attack types. These are complementary,
  not competitive.
- A single supervised model cannot detect novel attacks by definition.

**Trade-off:** Two models to train and maintain vs. one. Complexity is justified by coverage.

**A1 connection:** §4.6 of the A1 group report concluded that layered defence outperforms
either model alone. V2 implements that conclusion.

---

### ADR-02 — Day-based temporal split (not random)

**Decision:** Split CIC-IDS2017 by capture date: Mon–Thu → train, Fri-AM → val, Fri-PM → test.

**Rationale:**
- Network traffic is temporally correlated. A random split leaks Fri-PM flows into training,
  inflating metrics — the model effectively memorises future attacks.
- Fri-PM contains DDoS + PortScan exclusively. With a day-based split, these are genuinely
  unseen at train time → the test set measures real novel-attack detection.
- This is the scientifically correct evaluation methodology for temporal data.

**Trade-off:** Smaller effective training set (Mon–Thu only). SMOTE compensates for the
resulting class imbalance.

**Known consequence:** `known_attack_f1 = 0.052` on the Fri-PM test set is *expected and correct*.
The classifier was never trained on Fri-PM attack types. The AE metric (Recall = 73.5%, DDoS 99.7%)
is the relevant measure for that test set.

---

### ADR-03 — XGBoost classifier frozen after initial training

**Decision:** XGBoost weights are never updated after the initial training run.

**Rationale:**
- The classifier is trained on labelled data from a fixed historical dataset (CIC-IDS2017).
  Adding new classes would require re-balancing all 15 classes, re-running SMOTE, and
  re-evaluating the full pipeline — not something that can be triggered safely from 50 analyst
  verdicts.
- Analyst verdicts are predominantly benign flows labelled by a single analyst on production
  traffic — not a balanced, verified multi-class training set.
- Freezing the classifier prevents a well-intentioned fine-tune from degrading classifier
  performance on known attacks.

**Trade-off:** The classifier's knowledge of attack types does not grow over time. This is
acceptable because novel-attack detection is delegated to the AE (the adaptive component).

**Fine-tune path:** Only the AE is periodically retrained. The `fine_tune.py` pipeline copies
`classifier.joblib` unchanged from the previous version.

---

### ADR-04 — MLP Autoencoder trained on benign-only data

**Decision:** The AE learns the distribution of normal (benign) traffic only.

**Rationale:**
- Anomaly detection = "does this deviate from normal?" Benign-only training directly captures
  that: anything the AE cannot reconstruct well is anomalous by definition.
- Training on attack samples would require knowing attack labels — defeating the purpose of
  detecting *novel* attacks.
- The reconstruction error threshold (p95 of val-benign errors) sets the false-alarm rate
  at ~5% of benign flows, not as a magic number but as a principled percentile choice.

**Trade-off:** The AE cannot distinguish attack *types* — it only says "anomalous". Type
attribution comes from the classifier (Stage 2). If Stage 2 misses an attack, Stage 3 flags
it as `NOVEL_ATTACK` with no type label.

---

### ADR-05 — AE threshold at p95 of validation-benign errors

**Decision:** Set the anomaly threshold to the 95th percentile of reconstruction errors on
the Fri-AM benign validation set.

**Rationale:**
- p95 means the AE flags ~5% of benign flows as anomalous (FAR ≈ 5% before Stage 1/2 filter).
- Computed on the *val* set (not train) so the threshold generalises beyond the training
  distribution.
- Configurable via `AE_ANOMALY_THRESHOLD_PERCENTILE` — analysts can tune: lower p-tile =
  more sensitive (higher FAR), higher p-tile = fewer false positives.

**Observed result (V9):** FAR(val) = 0.016% (very low on known-distribution benign traffic);
detection_rate(test) = 18% (18% of Fri-PM flows flagged as anomalous — expected, since the test set contains attacks).

---

### ADR-06 — Encoder freeze + replay buffer during AE fine-tune

**Decision:** Freeze the AE encoder for the first `AE_FREEZE_EPOCHS` epochs of fine-tuning,
and include a fraction of historical benign data in every fine-tune batch.

**Rationale — encoder freeze:**
- Transfer learning principle: the encoder has learnt a useful representation of benign traffic.
  Abruptly fine-tuning the encoder on a small incremental batch (50 flows) would overwrite
  that representation (catastrophic forgetting).
- Freezing the encoder forces the decoder to adapt first; subsequent unfrozen epochs then
  refine the encoder gently.

**Rationale — replay buffer:**
- Small incremental batches (50 verdicts) can overfit the fine-tuned AE to a narrow slice of
  recent traffic, degrading performance on older benign patterns.
- Replaying `AE_REPLAY_BUFFER_FRACTION` (10%) of historical benign data ensures the AE
  stays calibrated on the full benign distribution.

**Trade-off:** Fine-tuning takes longer than a pure incremental update. Worth it for stability.

---

### ADR-07 — In-process threading over Celery for training jobs

**Decision:** Run training in a `ThreadPoolExecutor` within the FastAPI process, bridged to
SSE via `asyncio.Queue`.

**Rationale:**
- Zero external infrastructure: no Redis, no Celery worker, no message broker.
- Single-user demo: only one training job runs at a time. Concurrency is not a requirement.
- The complexity budget for a coursework project should be spent on ML, not on distributed
  task infrastructure.

**Trade-off:** If the API process dies mid-training, the job is lost. Mitigated by:
(a) marking stale `training` status rows as `failed` on startup, and
(b) the artifact-write atomicity guarantee (ADR-12).

**V1 issue:** `api/tasks.py` grew to 414 lines. V2 `api/jobs.py` is ~147 lines: the original
< 100 target was aspirational; predict-job management (async polling, result storage) added
a second job type with its own state, bringing the final count to 147. The pattern is clean
and the two job types are clearly separated.

---

### ADR-08 — SSE over WebSocket for training progress

**Decision:** Use Server-Sent Events (SSE) to stream training progress to the UI.

**Rationale:**
- Training progress is strictly unidirectional: server → client. SSE is the right primitive
  for unidirectional server push over HTTP.
- WebSocket adds bidirectional complexity (connection upgrade, reconnect logic, heartbeat)
  for no benefit here.
- SSE works over standard HTTP/1.1; no proxy reconfiguration needed.
- Browser fallback: `GET /api/training/{run_id}/status` for polling when SSE is blocked.

**Trade-off:** SSE connections are per-browser-tab; reconnects need a run_id. Handled by
storing run_id in component state.

---

### ADR-09 — SQLite for the database

**Decision:** SQLite with SQLAlchemy async (aiosqlite) as the default database.

**Rationale:**
- Zero-infra: no Postgres server to spin up. Fits a local-demo deployment perfectly.
- SQLAlchemy + Alembic means migrating to PostgreSQL later is a `DATABASE_URL` change plus
  a migration run — the application code is unchanged.
- Single-user write load is within SQLite's performance envelope.

**Trade-off:** SQLite has no row-level locking; concurrent writes from the training thread and
the async API can conflict. Mitigated by using a sync session exclusively in training threads
and keeping write transactions short.

---

### ADR-10 — DuckDB for CSV loading in training

**Decision:** Use DuckDB to query CIC-IDS2017 CSV files during training, not pandas `read_csv`.

**Rationale:**
- CIC-IDS2017 is 2.8M rows across 8 CSV files (~700 MB). Pandas loads everything into memory.
- DuckDB scans columns on demand; a query selecting only the needed feature columns uses
  ~3× less peak memory and is ~10× faster on a cold read.
- The query result is returned as a pandas DataFrame, so the rest of the pipeline is unchanged.

**Trade-off:** An additional dependency. Justified by the 10× speedup on the data loading step.

---

### ADR-11 — Dual scalers (clf_scaler + det_scaler)

**Decision:** Fit separate scalers for the classifier path and the detector path.

**Rationale:**
- XGBoost (tree-based) is invariant to monotone feature transformations — scaling has no effect
  on tree splits. Setting `PreprocessingSpec(needs_scaling=False)` skips the scaler for XGBoost.
- MLP-AE (gradient-based) requires features on a comparable scale to avoid vanishing/exploding
  gradients. `preferred_scalers=("quantile",)` for the detector path.
- Using the same scaler for both would force the classifier through an unnecessary step.

**Trade-off:** Two scaler artifacts to save and load. Trivial cost vs. correct behaviour.

---

### ADR-12 — Atomic artifact directory writes

**Decision:** Write model artifacts to `models/tmp_<run_id>/`, then rename atomically to
`models/<name>/` on success. On failure, remove the tmp directory.

**Rationale:**
- A crash mid-write leaves a partial artifact directory. Loading a partial directory would
  silently fail or produce wrong predictions.
- `os.rename()` is atomic on POSIX filesystems (same device). The rename is the commit.
- If the process crashes before the rename, the tmp directory is cleaned up on next startup.

**Trade-off:** Double the disk I/O during training. Negligible compared to training time.

---

### ADR-13 — Per-feature reconstruction error as XAI for the AE

**Decision:** Explain AE anomaly scores by returning the top-N features ranked by per-feature
reconstruction error `(x_i − x̂_i)²`.

**Rationale:**
- SHAP for a neural network requires a background dataset and multiple forward passes —
  expensive at inference time.
- Per-feature reconstruction error is already computed during the AE forward pass (zero
  additional cost). It directly answers: "which features does the AE struggle to reconstruct?"
  — which is exactly what an analyst needs to understand why a flow was flagged.
- Interpretable without ML background: "Flow Duration contributed 41% of the anomaly score"
  is actionable for a SOC analyst.

**Trade-off:** Not a formal feature attribution method (not SHAP/LIME). Sufficient for the
demo and the assessment. SHAP is a future enhancement (ADR noted in §2 Non-Goals).

---

### ADR-14 — SMOTE capped at 50k samples per class

**Decision:** Apply SMOTE to minority classes but cap at `SMOTE_MAX_PER_CLASS = 50000` synthetic
samples per class.

**Rationale:**
- CIC-IDS2017 has extreme class imbalance: BENIGN has 2M+ rows; some attack classes have < 500.
  An uncapped SMOTE would generate millions of synthetic samples, ballooning training time.
- 50k per class provides sufficient signal for the classifier while keeping training time under
  15 minutes on a GPU.
- The cap is configurable — analysts can reduce it for faster iteration during development.

**Trade-off:** The smallest attack classes are still under-represented relative to BENIGN
even after SMOTE. Acceptable because Stage 3 (AE) handles unknown attacks regardless.

---

### ADR-15 — K=50 as the AE retrain unlock threshold

**Decision:** Require 50 `retraining_eligible` analyst verdicts before fine-tuning unlocks.

**Rationale:**
- Too low (< 20): fine-tune signal is noisy; the fine-tuned AE may overfit to a narrow batch.
- Too high (> 200): the retrain loop never completes in a demo session (only ~100 flows
  uploaded in a typical demo).
- 50 is a practical midpoint: achievable in a 10-minute demo, sufficient for the encoder-freeze
  warm-start strategy to be effective.

**Trade-off:** Somewhat arbitrary. The right number depends on traffic volume and analyst throughput.
Configurable via `FINE_TUNE_K_THRESHOLD` so it can be tuned without a code change.

---

### ADR-16 — Protocol-based model interfaces (not class inheritance)

**Decision:** `BaseClassifier` and `BaseAnomalyDetector` are Python `Protocol` classes
(structural subtyping), not abstract base classes.

**Rationale:**
- A Protocol allows duck typing: any class that implements the required methods is a valid
  classifier/detector — no explicit `class MyModel(BaseClassifier)` inheritance needed.
- This matters for wrapping third-party models (XGBoost, scikit-learn IF) that cannot subclass
  our base. The wrapper implements the protocol; the wrapped model stays unchanged.
- `@runtime_checkable` allows `isinstance(model, BaseClassifier)` checks at load time.

**Trade-off:** IDEs may not auto-complete Protocol implementations (depends on LSP support).
Pylance/Pyright handle this correctly.

---

### ADR-17 — React SPA over server-rendered HTML

**Decision:** Build the UI as a React 18 + TypeScript Single-Page Application.

**Rationale:**
- SSE-driven training progress requires real-time DOM updates without page reloads. A React
  component that subscribes to an SSE stream and updates a progress bar is idiomatic React;
  server-rendered HTML with JavaScript sprinkled on top would be fragile.
- The alert table needs optimistic verdict updates (submit a verdict → button disables
  immediately, no reload). React state makes this trivial.
- Vite 5 gives < 500ms hot-reload in development; the build step is `vite build` → static
  files served by nginx.

**Trade-off:** A heavier frontend setup than plain HTML. Justified by the reactive UX requirements.

---

### ADR-18 — `api/schemas.py` as the SDD contract layer

**Decision:** All Pydantic request and response schemas live in one file: `api/schemas.py`.
No router file defines its own inline Pydantic models.

**Rationale:**
- SDD principle: the spec (this file) declares what must be true; `schemas.py` is the
  machine-executable realisation of §4 (API Contract).
- If schemas are scattered across routers, drift is invisible: two routers can define
  `Alert` differently, and the only way to detect it is reading all the files.
- Centralising them makes the contract surfaceable: `GET /openapi.json` is a full
  machine-readable spec that can be validated against this document.

**Trade-off:** One large file. Manageable at this project scale.

---

### ADR-19 — Makefile replaces ops.sh + setup.sh

**Decision:** All developer and operator commands are in a `Makefile`.

**Rationale:**
- V1 had `ops.sh` (runtime), `setup.sh` (setup), `setup_gpu.sh` (GPU setup) — three files
  with overlapping responsibility.
- `make` is universal (Linux/macOS/WSL), has built-in documentation (`make help`), and
  tab-complete works without sourcing a shell script.
- A Makefile is the standard contract for "how do I do X in this project" in open-source
  Python/Node projects. Examiners and collaborators recognise it immediately.

**Trade-off:** `make` must be installed (trivial on Linux/WSL; `winget install ezwinports.make`
on Windows).

---

### ADR-20 — Alembic for DB schema migrations

**Decision:** Use Alembic for all database schema changes. No `ALTER TABLE` in application code.

**Rationale:**
- V1 accumulated three startup functions: `_migrate_db_location_on_startup()`,
  `_migrate_schema_on_startup()`, `_mark_stale_training_on_startup()`. Each was added as a
  band-aid when the schema changed. They ran on every startup, could not be rolled back,
  and had no version tracking.
- Alembic provides: version history, rollback, autogenerate from ORM diff, CI validation.
- The discipline: to change the schema, write a migration. No exceptions.

**Trade-off:** Slightly more ceremony for the first migration (alembic init, env.py setup).
One-time cost that pays for itself the first time a schema change is needed.

---

### ADR-21 — XGBoost as default Stage 2 classifier

**Decision:** Use XGBoost as the default Stage 2 classifier; keep Random Forest as a selectable fallback.

**Alternatives considered:**

| Alternative | Why rejected |
|---|---|
| **Logistic Regression** | Assumes linear decision boundaries. CIC-IDS2017 features have complex non-linear interactions (packet length × flow duration × flag combinations) — a linear model structurally underfits 47 features across 15 attack classes. |
| **Naive Bayes** | Assumes feature independence. The correlation pruning step (removing \|r\| > 0.95 pairs) exists precisely because CIC-IDS2017 features are highly correlated — NB's core assumption is directly violated by this dataset. |
| **SVM** | O(n²)–O(n³) training complexity on 2.8M rows is impractical. No native GPU support. No built-in feature importances — fails the Stage 2 XAI requirement. |
| **Random Forest** | Closest competitor and A1's primary classifier. Parallel bagged trees reduce variance well but XGBoost's sequential boosting corrects residual errors, giving better F1 on imbalanced datasets. RF also lacks native GPU support. Kept as a selectable fallback (one line in `core/registry.py`) for A1 reproducibility. |

**Rationale for XGBoost:**
- **GPU acceleration** — `device='cuda'` cuts CIC-IDS2017 training from ~4 hours to ~14 minutes on RTX A2000
- **Imbalanced data** — `scale_pos_weight` + early stopping handle the 80%+ BENIGN skew natively
- **Built-in XAI** — `feature_importances_` at zero extra cost satisfies the Stage 2 explainability contract
- **No feature scaling** — tree splits are threshold-based; `clf_scaler` is an identity passthrough, simplifying the dual-scaler contract (ADR-11)
- **Tabular SOTA** — consistently outperforms other algorithms on structured tabular classification

**Trade-off:** XGBoost weights are frozen after initial training (ADR-03). The classifier's knowledge of attack types does not grow over time — Stage 3 (AE) is the adaptive component.

**Not considered:** Isolation Forest was not evaluated for Stage 2 — it is an unsupervised anomaly detector with no concept of labelled classes. It cannot predict specific attack types (DDoS, PortScan, etc.) from labelled training data. It belongs in Stage 3 by definition (see ADR-22).

**A1 connection:** A1 used Random Forest as the primary classifier. XGBoost is the natural upgrade within the same ensemble-tree family, with better regularisation, GPU support, and handling of class imbalance.

---

### ADR-22 — MLP Autoencoder as Stage 3 anomaly detector

**Decision:** Use an MLP Autoencoder (trained on benign-only flows) as the Stage 3 anomaly detector; keep Isolation Forest as a selectable fallback.

**Alternatives considered:**

| Alternative | Why rejected |
|---|---|
| **Isolation Forest** | Already implemented as a selectable fallback. Uses random subsampling to isolate outliers — no reconstruction signal → no per-feature XAI. Not differentiable → cannot be fine-tuned with gradient descent; the HITL loop (ADR-06) requires encoder freeze + replay buffer, which depend on backpropagation. Eliminated by the fine-tune requirement. |
| **Variational Autoencoder (VAE)** | Same architecture family as MLP-AE; adds KL divergence loss and the reparameterisation trick for probabilistic latent space. Theoretically stronger generalisation, but in practice offers no measurable gain on tabular aggregated features over a standard AE — at the cost of training instability and implementation complexity. |
| **LSTM / GRU Autoencoder** | Designed for raw temporal sequences. CIC-IDS2017 flows are pre-aggregated statistics (duration, packet counts, byte totals) — temporal ordering within a flow is already encoded in the features. Sequence modelling adds architectural complexity with no benefit for aggregated tabular records. |
| **k-Means clustering** | Cluster normal traffic; flag high centroid-distance flows. Cannot fine-tune incrementally — re-clustering shifts all centroids, breaking prior threshold calibration. Sensitive to K selection. No reconstruction error → no XAI. |

**Rationale for MLP-AE:**
- **Differentiable — fine-tune requirement met** — gradient-based updates enable encoder freeze + replay buffer (ADR-06), which are the mechanism behind the HITL learning loop. Every non-differentiable alternative (Isolation Forest, OC-SVM, LOF, k-Means) is eliminated by this single requirement.
- **Free per-feature XAI** — reconstruction error `(x_i − x̂_i)²` per feature is an immediate anomaly explanation at zero extra cost (ADR-13)
- **Natural anomaly score** — reconstruction error is a continuous, interpretable signal; threshold set at p95 of validation-benign errors (ADR-05)
- **GPU-acceleratable** — PyTorch is already in the stack; same `device='cuda'` path used by the AE and XGBoost
- **No training data at inference** — only model weights are needed; unlike LOF, scales to production without memory growth

**Trade-off:** AE threshold requires recalibration after each fine-tune (recomputed on val-benign flows). False-alarm rate is sensitive to threshold choice — ADR-05 and ADR-06 address this with the regression gate.

**A1 connection:** A1 compared Random Forest (supervised) vs MLP Autoencoder (unsupervised) and concluded the AE generalises better to novel attack types by modelling normality rather than attack signatures. V2 implements that conclusion as Stage 3.

---

### ADR-23 — `runtime/` as the environment boundary layer

**Decision:** Isolate all host-environment discovery in a dedicated `runtime/` package. All source
packages (`core/`, `training/`, `inference/`) are environment-agnostic and receive resolved
configuration as parameters.

**What lives in `runtime/`:**
- `probe.py` — CUDA detection (`torch.cuda.is_available()`), Docker availability (`docker info`),
  image inventory (`docker images --format json`); returns a `HostEnvironment` dataclass
- `manifest.py` — `HostEnvironment` singleton, resolved once at startup and cached for the process
  lifetime; re-runs on demand via `GET /api/preflight`
- `preflight.py` — structured check results (`ok` / `warn` / `blocker`) served via
  `GET /api/preflight`

**Preflight check levels:**

| Level | Meaning | UI behaviour |
|-------|---------|--------------|
| `ok` | Capability confirmed | No indicator |
| `warn` | Capability absent but non-blocking (e.g., no CUDA → CPU fallback) | Banner shown |
| `blocker` | Capability required to continue (e.g., Docker not installed) | Prerequisites screen; flow halted |

**Rationale:**
- Without this boundary, environment checks scatter: `torch.cuda.is_available()` in one file,
  `subprocess.run("docker")` in another, `os.environ` reads everywhere.
- The existing `DEVICE: "auto"` config var (§7) is the seed of this pattern. `runtime/` is its
  generalisation to the full set of host capabilities.
- The `HostEnvironment` singleton enables `GET /api/preflight`, which the UI renders as a
  requirements screen on first load. Without a structured probe, blockers cannot be surfaced before
  the user tries to act.
- Docker image inventory lets the probe report reuse vs. pull cost upfront
  (e.g., "anomaly-flow:latest reused — saved 3 min pull").

**Rule:** `core/`, `training/`, and `inference/` never call `torch.cuda.is_available()`,
`subprocess.run(...)`, or read `os.environ` directly. They receive `device: str` and other
resolved values as parameters from the caller (the API startup sequence or the training job).

**Trade-off:** Probe runs at startup (~0.5 s: Docker info + image list via subprocess). Subprocess
calls are wrapped in `try/except` — if Docker is absent, `probe_docker()` returns
`DockerStatus(installed=False)` rather than raising. `HostEnvironment` is cached for the process
lifetime; `GET /api/preflight` re-runs probes on demand if the environment changes after startup.

---

*End of spec. All implementation should be traceable to a section above.*
