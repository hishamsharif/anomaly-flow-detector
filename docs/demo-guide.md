# AFD V2 — Demo Guide (6–8 min)

## Pre-demo checklist

- [ ] System running: `make start` (Docker) or `make start api` + `make start ui`
- [ ] Active model in DB — do **not** train live (takes ~20 min); verify on Training tab
- [ ] Two CSV files prepared:
  - `demo_benign.csv` — benign-only flows (Monday split)
  - `demo_mixed.csv` — mixed traffic with KNOWN and NOVEL verdicts (Fri-PM test split)
- [ ] Browser open to `http://localhost:5173`, full screen
- [ ] `docs/openapi.json` and `SPEC.md` open in a text editor (for the architecture credibility beat)

---

## Script

### 0:00 – 0:45 · Hook — About tab

Open the **About** tab. Point at the pipeline diagram.

> "AFD V2 is a 3-stage Network Intrusion Detection System. Every network flow
> passes through a deterministic rule engine, a supervised XGBoost classifier,
> and an unsupervised MLP Autoencoder. The system produces three verdicts —
> Safe, Known Attack, or Novel/Zero-day Attack. When analysts submit verdicts
> on flagged flows, the autoencoder fine-tunes on their feedback. That's the
> HITL loop."

**Key point:** name all three stages and all three verdicts. Examiners are looking for this.

---

### 0:45 – 2:00 · Dataset understanding — Dataset tab

Open the **Dataset** tab.

**Donut chart (class imbalance):**
> "80.3% of the 2.83 million flows are benign. A classifier that always
> predicts Benign scores 80% accuracy without catching a single attack.
> That's why we use F1-score, Precision/Recall, and ROC-AUC as primary
> metrics — and why SMOTE oversampling is applied to the training split only."

**Temporal split bar chart:**
> "Monday is 100% benign. Each subsequent day introduces attack types not
> seen before — DoS variants on Wednesday, web attacks Thursday, DDoS and
> PortScan on Friday afternoon. We split by day, not randomly. A random
> 80/20 split leaks examples of every attack class into training and
> inflates held-out performance. The temporal split is the same challenge
> the model faces in production."

**Why this matters:** justifies ADR-02 (temporal split) and ADR-04 (metric choice) without
opening the SPEC. Examiners notice when design decisions are grounded in data.

---

### 2:00 – 2:30 · Model metrics — Training tab

Open the **Training** tab. Point at the active model row.

> "Training ran offline. Here's the active model."

Hover each metric heading to show the tooltip:
- **F1** → harmonic mean of Precision and Recall
- **FAR** → "0.00022 — less than 0.03% of benign flows are falsely flagged as attacks"
- **Miss Rate ↓** → "this is the autoencoder's miss rate — the zero-day detector's threshold
  calibration improves as more analyst verdicts are collected"

**Do not start a training run live.**

---

### 2:30 – 5:30 · End-to-end inference — Inference tab ← *centrepiece*

Open the **Inference** tab. Upload `demo_mixed.csv`.

**Step 1 — summary bar loads:**
> "The pipeline processed all flows. Here's the breakdown: X Safe, Y Known Attack, Z Novel Attack."

**Step 2 — expand a KNOWN_ATTACK row:**
> "XGBoost flagged this flow at 94% confidence. The top contributing features
> are Fwd IAT Mean and Flow Duration — consistent with a DoS slow-rate pattern.
> Stage 1 (rule engine) passed it through; Stage 2 (classifier) caught it."

**Step 3 — expand a NOVEL_ATTACK row:**
> "XGBoost was not confident enough to call this an attack. The AE saw a
> reconstruction error well above the threshold — this flow looks unlike
> anything in the benign training data. Stage 3 caught what Stage 2 missed.
> That's the zero-day path."

**Step 4 — submit analyst verdicts:**
Click **Confirm** on a SAFE row and **Override** on a NOVEL row. Watch the buttons
disable immediately.
> "This is the optimistic UI update — the verdict is locked the moment it's
> submitted. Each submission increments the HITL counter."

---

### 5:30 – 6:30 · HITL loop — Dashboard tab

Open the **Dashboard** tab. Point at the verdict progress bar.

> "Once 50 analyst verdicts are collected, the Retrain AE button unlocks.
> The autoencoder fine-tunes on analyst-confirmed benign flows — the
> classifier weights are frozen, only the anomaly detector adapts. After
> fine-tuning, a regression gate checks the new FAR and recall against the
> previous version. If either degrades beyond the configured threshold,
> the new weights are rejected automatically."

Point at the 24-hour alert summary cards.
> "The dashboard gives the SOC analyst an at-a-glance view of the last 24
> hours — total flows, by verdict, by severity."

---

### 6:30 – 7:30 · Architecture credibility (optional, if time allows)

Briefly switch to `SPEC.md` or `docs/openapi.json` in the editor.

> "The spec is the source of truth. Every ADR, invariant, and API contract
> was written here before any code. The SDD approach means the OpenAPI spec,
> the test suite, and the UI client are all derivatives of this document.
> Any divergence between SPEC and code is treated as a bug."

Switch back to the browser. Optionally mention:
> "79 tests covering all three pipeline stages, the HITL threshold, and
> the temporal-split data loader."

---

### 7:30 – 8:00 · Limitations + close

> "Known limitations: the autoencoder miss rate reflects an uncalibrated
> threshold on a fresh model — it improves as the HITL loop collects verdicts.
> The temporal split means Friday-PM attack types appear only in the test set,
> so the classifier hasn't seen DDoS or PortScan during training.
>
> Future work: SHAP-based feature explanations for the AE, savepoint tracking
> in the HITL replay buffer, and live streaming PCAP ingestion via nfstream."

End on "here's what's next" — not on a limitation.

---

## Timing reference

| Segment | Duration | Tab |
|---------|----------|-----|
| Hook — pipeline overview | 0:45 | About |
| Dataset EDA | 1:15 | Dataset |
| Model metrics | 0:30 | Training |
| Inference — 3-stage + HITL verdicts | 3:00 | Inference |
| HITL loop + dashboard | 1:00 | Dashboard |
| Architecture credibility | 1:00 | Editor / SPEC |
| Limitations + close | 0:30 | — |
| **Total** | **~8:00** | |

---

## Do / Don't

| Do | Don't |
|----|-------|
| Narrate **why** each design decision exists | Read text from the UI |
| Let results load live — shows it's real | Start a training run live |
| Hover metric tooltips (FAR, AMR) to explain | Skip the Dataset tab |
| Submit verdicts and show optimistic UI disable | Over-explain the code |
| Mention SPEC as the living document | Apologise for AMR = 1.0 without context |
| State limitations confidently | Trail off at the end |

---

## If something goes wrong

| Problem | Recovery |
|---------|----------|
| API unreachable | `make logs api` in terminal; check `http://localhost:8000/api/health` |
| No active model | Go to Training tab → show "No active model" state → explain it triggers once setup is complete |
| Inference returns all SAFE | Wrong CSV — make sure `demo_mixed.csv` has the Fri-PM test flows |
| UI not updating | Hard refresh (`Ctrl+Shift+R`); dev server HMR may have stalled |
| Docker GPU not detected | `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up` and confirm `nvidia-smi` available |
