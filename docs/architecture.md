# AFD V2 — Architecture

## Overview

**Anomaly Flow Detector (AFD)** is a 3-stage Network Intrusion Detection System (NIDS)
that classifies network flows as SAFE, KNOWN_ATTACK, or NOVEL_ATTACK. It combines
deterministic rules, supervised classification (XGBoost), and unsupervised anomaly
detection (MLP Autoencoder) in a layered defence architecture.

Trained on the **CIC-IDS2017** dataset (2.8M flows, 15 attack classes).  
Day-based temporal split: Mon–Thu → train, Fri-AM → val, Fri-PM → test.

---

## 3-Stage Detection Pipeline

```
 Network Flow (feature dict)
          │
          ▼
 ┌─────────────────────────────────────────────────────────┐
 │  Stage 1: Rule Engine (deterministic — always first)    │
 │                                                         │
 │  CVE signatures, known-malicious IPs ──► KNOWN_ATTACK   │
 │  Allowlisted assets, trusted sources ──► SAFE           │
 │  Ambiguous flows ──────────────────────────────────────► │
 └─────────────────────────────────────────────────────────┘
          │ (ambiguous — no rule matched)
          ▼
 ┌─────────────────────────────────────────────────────────┐
 │  Stage 2: XGBoost Classifier                            │
 │                                                         │
 │  confidence ≥ CLF_CONFIDENCE_THRESHOLD ──► KNOWN_ATTACK │
 │  confidence < threshold ───────────────────────────────► │
 └─────────────────────────────────────────────────────────┘
          │ (all ambiguous flows — Stage 3 always runs)
          ▼
 ┌─────────────────────────────────────────────────────────┐
 │  Stage 3: MLP Autoencoder (benign-only trained)         │
 │                                                         │
 │  ae_score > ae_threshold ──► NOVEL_ATTACK               │
 │  ae_score ≤ ae_threshold ──► SAFE                       │
 └─────────────────────────────────────────────────────────┘
          │
          ▼
 Consensus verdict + severity (low / medium / high / critical)
```

**Precedence:** Stage 1 always wins. If Stage 2 fires KNOWN_ATTACK, the flow is
labelled KNOWN_ATTACK regardless of Stage 3's ae_score (though the score is still
recorded). Stage 3 only determines the final verdict when Stage 2 did not fire.

---

## Component Responsibilities

| Component | Responsibility |
|-----------|---------------|
| `core/rule_engine.py` | Deterministic rules — CVE signatures, IP blocklists, trusted ranges. Zero ML. |
| `core/classifiers/xgboost_clf.py` | Supervised XGBoost — detects known attack patterns from labelled CIC-IDS2017 data |
| `core/detectors/mlp_ae.py` | MLP Autoencoder — trained on benign-only traffic; high reconstruction error = anomalous |
| `core/store.py` | `ModelStore` singleton — in-memory active bundle; `save_artifacts()` / `load_into_memory()` |
| `training/pipelines/initial.py` | Full training pipeline: data load → feature engineering → XGBoost + AE fit → evaluate → save atomically → auto-promote |
| `training/pipelines/fine_tune.py` | AE-only retrain: load prev bundle → replay buffer → freeze encoder → recalibrate threshold → regression gate → copy classifier (I5) → promote |
| `training/feature_eng/` | Variance filter → correlation filter → optional SelectKBest → dual scalers (classifier + detector paths) |
| `inference/pipeline.py` | Batch 3-stage inference runner: rules → XGBoost → AE → consensus + severity |
| `inference/adapters/csv_adapter.py` | CSV → validated feature rows (I3 shape check) |
| `inference/adapters/pcap_adapter.py` | PCAP → flow features via nfstream (best-effort CIC-IDS2017 column mapping) |
| `api/` | FastAPI app — routers, schemas, jobs bridge, DB models |
| `api/jobs.py` | `ThreadPoolExecutor` + append-only SSE event log — training + predict background jobs |
| `runtime/` | Host environment probes — CUDA, Docker, image inventory; resolved once at startup |
| `training/train_offline.py` | Offline initial training entry point — trains classifier + AE on CIC-IDS2017, no server required (`make train`) |
| `inference/inference_offline.py` | Standalone CLI examiner — loads active model from SQLite, scores CSV/PCAP without HTTP overhead |

---

## Data Flow

```
                   ┌──────────────┐
  CSV / PCAP ─────►│  Adapter     │── valid_rows ──────────────────────┐
                   │  (I3 check)  │── rejected_rows                     │
                   └──────────────┘                                      │
                                                                          ▼
                                        ┌──────────────────────────────────────┐
   ModelStore.get_active() ────────────►│  run_batch_inference()               │
   (ArtifactBundle in memory)           │  Stage 1 → Stage 2 → Stage 3         │
                                        │  Returns List[FlowResult]             │
                                        └──────────────────────────────────────┘
                                                          │
                                                          ▼
                                               ┌──────────────────┐
                                               │  Alert DB (SQLite)│
                                               │  Analyst verdicts │
                                               └──────────────────┘
                                                          │
                                                          ▼
                                               ┌──────────────────┐
                                               │  Fine-tune gate  │
                                               │  (≥ 50 verdicts) │
                                               └──────────────────┘
                                                          │
                                                          ▼
                                               ┌──────────────────┐
                                               │  AE fine-tune    │
                                               │  pipeline        │
                                               └──────────────────┘
```

---

## Artifact Directory Layout

Each trained version is saved atomically to `models/<name>/`:

```
models/<name>/
  classifier.joblib     XGBoost model (frozen after initial training — ADR-03)
  detector.pt           MLP-AE state dict + threshold + n_features
  clf_scaler.joblib     StandardScaler for classifier input path
  det_scaler.joblib     StandardScaler for detector input path (fit on benign-only)
  variance_mask.npy     Boolean mask from VarianceThreshold step
  corr_mask.npy         Boolean mask from correlation pruning step
  selector_mask.npy     Boolean mask from SelectKBest step (all-True when k=0)
  feature_names.json    Ordered list of final selected feature names (I3 contract)
  metadata.json         Metrics + config snapshot at training time
```

Write is atomic: data goes to `models/tmp_<run_id>/`, then renamed to `models/<name>/` on
success. A partial write (crash mid-training) leaves only the `tmp_` dir which is ignored.

---

## Key Design Decisions (ADR Summary)

| ADR | Decision | Rationale |
|-----|----------|-----------|
| ADR-01 | 3-stage pipeline | Each stage covers a different threat class; complementary, not competitive |
| ADR-02 | Day-based temporal split | Prevents data leakage from future attack types into training |
| ADR-03 | XGBoost frozen after initial training | Analyst verdicts aren't a balanced labelled dataset for the classifier |
| ADR-04 | AE trained on benign-only | Anomaly = deviation from normal; no attack labels needed |
| ADR-05 | AE threshold at p95 val-benign | Principled FAR ≈ 5%; configurable via `AE_ANOMALY_THRESHOLD_PERCENTILE` |
| ADR-06 | Encoder freeze + replay buffer | Prevents catastrophic forgetting during AE fine-tune |
| ADR-07 | In-process `ThreadPoolExecutor` | Zero external infrastructure (no Celery, no Redis) |
| ADR-08 | SSE over WebSocket | Training progress is strictly server → client; SSE is the right primitive |

---

## Tech Stack

| Layer | Choice |
|-------|--------|
| Language | Python 3.12 |
| API | FastAPI + Pydantic v2 |
| DB | SQLite + SQLAlchemy async + aiosqlite |
| Migrations | Alembic |
| ML — Classifier | XGBoost (GPU via `device='cuda'`) |
| ML — Detector | PyTorch MLP-AE (GPU via `.to('cuda')`) |
| Feature engineering | scikit-learn + imbalanced-learn (SMOTE) |
| Data loading | DuckDB (10× faster than pandas for 2.8M rows) |
| PCAP parsing | nfstream |
| UI | React 18 + TypeScript + Vite 5 + Tailwind CSS |
| Containers | Docker Compose (api + ui/nginx) |
| Tests | pytest + httpx (async) |
| Linting | ruff |
