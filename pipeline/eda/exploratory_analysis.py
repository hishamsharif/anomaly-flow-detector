"""Offline EDA + preprocessing report for CIC-IDS2017.

Run via:  make eda
Or directly:  PYTHONPATH=. python eda/eda_offline.py

Scans data/CIC-IDS2017/, computes:
  - Class distribution across all flows
  - Per-day benign/attack breakdown with split assignment
  - Preprocessing pipeline feature counts (variance → correlation → selector)
  - Pearson correlation matrix for top-20 surviving features (benign sample)

Writes data/eda_summary.json and prints a terminal report.
The Dataset tab reads this file at runtime; re-run to refresh.
"""
from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── Day label mapping ──────────────────────────────────────────────────────────
# Maps fragments of CIC-IDS2017 filename → (display label, split)
_DAY_MAP: list[tuple[str, str, str]] = [
    ("Monday",                        "Monday",         "train"),
    ("Tuesday",                       "Tuesday",        "train"),
    ("Wednesday",                     "Wednesday",      "train"),
    ("Thursday-WorkingHours-Morning", "Thu AM",         "train"),
    ("Thursday-WorkingHours-Afternoon", "Thu PM",       "train"),
    ("Friday-WorkingHours-Morning",   "Fri AM",         "val"),
    ("Friday-WorkingHours-Afternoon-DDos", "Fri PM\n(DDoS)", "test"),
    ("Friday-WorkingHours-Afternoon-PortScan", "Fri PM\n(Scan)", "test"),
]

TOP_CORR_FEATURES = 20
BENIGN_SAMPLE_N   = 5_000
PREP_SAMPLE_N     = 10_000   # rows used for preprocessing pipeline counts


def _day_label(filename: str) -> tuple[str, str]:
    """Return (display_label, split) for a CIC-IDS2017 filename."""
    for fragment, label, split in _DAY_MAP:
        if fragment.lower() in filename.lower():
            return label, split
    return Path(filename).stem, "train"


def _load_csv_raw(path: Path) -> "pd.DataFrame":
    """Load a single CSV keeping the Label column, return raw DataFrame."""
    import duckdb
    db = duckdb.connect()
    df = db.execute(
        f"SELECT * FROM read_csv_auto('{path}', header=true, ignore_errors=true)"
    ).df()
    db.close()
    df.columns = [c.strip().replace(".", "_") for c in df.columns]
    if "Label" not in df.columns:
        # Try case-insensitive match
        for col in df.columns:
            if col.lower() == "label":
                df = df.rename(columns={col: "Label"})
                break
    return df


def _numeric_features(df: "pd.DataFrame") -> list[str]:
    return [c for c in df.select_dtypes(include=["number"]).columns if c != "label"]


# ── Main ───────────────────────────────────────────────────────────────────────

def run_eda(dataset_dir: str, output_path: str = "data/eda_summary.json") -> dict:
    import pandas as pd

    dataset_dir_path = Path(dataset_dir)
    csvs = sorted(dataset_dir_path.glob("*.csv"))
    if not csvs:
        log.error("No CSV files found in %s", dataset_dir)
        sys.exit(1)

    log.info("Dataset   : %s  (%d CSV files)", dataset_dir, len(csvs))
    log.info("")

    # ── 1. Load all CSVs, compute class counts + per-day breakdown ─────────────
    log.info("Scanning flows and labels …")
    all_class_counts: dict[str, int] = defaultdict(int)
    day_rows: list[dict] = []
    full_frames: list[pd.DataFrame] = []

    for csv_path in csvs:
        df = _load_csv_raw(csv_path)
        if "Label" not in df.columns:
            log.warning("  Skipping %s — no Label column", csv_path.name)
            continue

        label_col = df["Label"].str.strip()
        for lbl, cnt in label_col.value_counts().items():
            all_class_counts[lbl] += int(cnt)

        day_label, split = _day_label(csv_path.name)
        benign_count = int((label_col == "BENIGN").sum())
        attack_counts: dict[str, int] = {}
        for lbl, cnt in label_col.value_counts().items():
            if lbl != "BENIGN":
                attack_counts[lbl] = int(cnt)

        day_rows.append({
            "name": day_label,
            "split": split,
            "benign": benign_count,
            "attacks": attack_counts,
        })

        full_frames.append(df)

    full_df = pd.concat(full_frames, ignore_index=True) if full_frames else pd.DataFrame()

    # ── 2. Print class distribution ────────────────────────────────────────────
    total_flows = sum(all_class_counts.values())
    benign_total = all_class_counts.get("BENIGN", 0)
    attack_total = total_flows - benign_total

    log.info("─── Class distribution ─────────────────────────────────")
    log.info("  Total flows:     %s", f"{total_flows:,}")
    for lbl, cnt in sorted(all_class_counts.items(), key=lambda x: -x[1]):
        pct = cnt / total_flows * 100 if total_flows else 0
        log.info("  %-30s %8s  (%5.1f%%)", lbl, f"{cnt:,}", pct)
    log.info("")

    # ── 3. Print per-day split ─────────────────────────────────────────────────
    log.info("─── Per-day split ──────────────────────────────────────")
    for d in day_rows:
        atk_total = sum(d["attacks"].values())
        atk_names = ", ".join(sorted(d["attacks"].keys())[:4])
        atk_note = f"  ({atk_names})" if atk_names else ""
        log.info(
            "  %-20s [%-5s]  %7s benign  %7s attacks%s",
            d["name"].replace("\n", " "),
            d["split"],
            f"{d['benign']:,}",
            f"{atk_total:,}",
            atk_note,
        )
    log.info("")

    # ── 4. Preprocessing pipeline feature counts ───────────────────────────────
    log.info("─── Preprocessing pipeline ─────────────────────────────")
    feature_counts = _compute_feature_counts(full_df, benign_total)
    log.info("  Raw features:              %3d", feature_counts["raw"])
    log.info(
        "  After variance filter:     %3d  (dropped %2d, threshold=%.2f)",
        feature_counts["after_variance"],
        feature_counts["raw"] - feature_counts["after_variance"],
        feature_counts["variance_threshold"],
    )
    log.info(
        "  After correlation filter:  %3d  (dropped %2d, threshold=%.2f)",
        feature_counts["after_correlation"],
        feature_counts["after_variance"] - feature_counts["after_correlation"],
        feature_counts["correlation_threshold"],
    )
    log.info(
        "  After MI selector:         %3d  (selected %2d, method=%s)",
        feature_counts["after_selector"],
        feature_counts["after_selector"],
        feature_counts["selector_method"],
    )
    log.info("")

    # ── 5. Correlation matrix (top-N benign-sample features) ──────────────────
    log.info("─── Correlation analysis (benign sample, top %d features) ─", TOP_CORR_FEATURES)
    corr_features, corr_matrix = _compute_correlation(full_df, feature_counts)
    if corr_features:
        # Find strongest off-diagonal pairs
        mat = np.array(corr_matrix)
        triu = np.triu(np.abs(mat), k=1)
        if triu.max() > 0:
            idx = np.unravel_index(np.argmax(triu), triu.shape)
            log.info(
                "  Strongest pair: %s ↔ %s  ρ=%.2f",
                corr_features[idx[0]], corr_features[idx[1]], mat[idx[0], idx[1]]
            )
    log.info("")

    # ── 6. Build + write summary JSON ─────────────────────────────────────────
    class_counts_list = [
        {"label": k, "count": v}
        for k, v in sorted(all_class_counts.items(), key=lambda x: -x[1])
    ]

    summary = {
        "total_flows": total_flows,
        "class_counts": class_counts_list,
        "days": day_rows,
        "corr_features": corr_features,
        "corr_matrix": corr_matrix,
        "feature_counts": {
            "raw": feature_counts["raw"],
            "after_variance": feature_counts["after_variance"],
            "after_correlation": feature_counts["after_correlation"],
            "after_selector": feature_counts["after_selector"],
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    log.info("EDA summary → %s", output_path)
    log.info("Run 'make start' — Dataset tab now shows live numbers.")
    return summary


def _compute_feature_counts(full_df: "pd.DataFrame", benign_total: int) -> dict:
    """Run the preprocessing pipeline on a stratified sample, return feature counts."""
    import pandas as pd
    from config.settings import get_settings
    from pipeline.preprocessing.variance_filter import VarianceFilter
    from pipeline.preprocessing.correlation_filter import CorrelationFilter
    from pipeline.preprocessing.selector import FeatureSelector

    settings = get_settings()

    if full_df.empty:
        return {"raw": 0, "after_variance": 0, "after_correlation": 0, "after_selector": 0,
                "variance_threshold": settings.variance_threshold,
                "correlation_threshold": settings.correlation_threshold,
                "selector_method": settings.feature_selection_method}

    # Build a cleaned numeric DataFrame from the raw data
    df = full_df.copy()
    df.columns = [c.strip() for c in df.columns]
    import numpy as np
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "Label" in df.columns:
        y = (df["Label"].str.strip() != "BENIGN").astype(int).values
    else:
        y = np.zeros(len(df), dtype=int)
    X = df[num_cols].values.astype(np.float32)

    raw_count = X.shape[1]

    # Stratified sample
    rng = np.random.default_rng(42)
    benign_idx = np.where(y == 0)[0]
    attack_idx = np.where(y == 1)[0]
    n_benign = min(len(benign_idx), PREP_SAMPLE_N // 2)
    n_attack = min(len(attack_idx), PREP_SAMPLE_N // 2)
    sel_idx = np.concatenate([
        rng.choice(benign_idx, n_benign, replace=False),
        rng.choice(attack_idx, n_attack, replace=False) if n_attack > 0 else np.array([], dtype=int),
    ])
    X_s = X[sel_idx]
    y_s = y[sel_idx]

    # Step 1 — variance filter
    vf = VarianceFilter(threshold=settings.variance_threshold)
    X_v = vf.fit_transform(X_s)
    names_v = [num_cols[i] for i, keep in enumerate(vf.mask) if keep]

    # Step 2 — correlation filter
    cf = CorrelationFilter(threshold=settings.correlation_threshold)
    X_c = cf.fit_transform(X_v, names_v)
    names_c = [n for n, keep in zip(names_v, cf.mask) if keep]

    # Step 3 — MI selector
    fs = FeatureSelector(k=settings.feature_selection_k, method=settings.feature_selection_method)
    fs.fit_transform(X_c, y_s, names_c)

    return {
        "raw": raw_count,
        "after_variance": int(vf.mask.sum()),
        "after_correlation": int(cf.mask.sum()),
        "after_selector": len(fs.selected_names),
        "variance_threshold": settings.variance_threshold,
        "correlation_threshold": settings.correlation_threshold,
        "selector_method": settings.feature_selection_method,
        "_names_after_correlation": names_c,
        "_X_v": X_v,
        "_names_v": names_v,
        "_cf_mask": cf.mask,
    }


def _compute_correlation(full_df: "pd.DataFrame", feature_counts: dict) -> tuple[list[str], list[list[float]]]:
    """Compute Pearson correlation matrix for top-N surviving features on benign sample."""
    import numpy as np
    import pandas as pd

    names_c: list[str] = feature_counts.get("_names_after_correlation", [])
    if not names_c or full_df.empty:
        return [], []

    # Build benign-only subset from raw data
    df = full_df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    if "Label" not in df.columns:
        return [], []

    benign_df = df[df["Label"].str.strip() == "BENIGN"]

    # Keep only the surviving corr-filter features that exist in the DataFrame
    available = [n for n in names_c if n in benign_df.columns]
    if not available:
        return [], []

    X_benign = benign_df[available].values.astype(np.float32)

    # Sample
    rng = np.random.default_rng(42)
    n = min(len(X_benign), BENIGN_SAMPLE_N)
    idx = rng.choice(len(X_benign), n, replace=False)
    X_s = X_benign[idx]

    # Pearson correlation matrix for all surviving features
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.corrcoef(X_s.T)
    corr = np.nan_to_num(corr, nan=0.0)

    # Select top-N features by mean |ρ| (excluding self-correlation diagonal)
    n_feat = corr.shape[0]
    off_diag = np.abs(corr) - np.eye(n_feat)
    mean_abs = off_diag.mean(axis=1)
    top_n = min(TOP_CORR_FEATURES, n_feat)
    top_idx = np.argsort(mean_abs)[::-1][:top_n]
    top_idx_sorted = sorted(top_idx.tolist())

    sel_names = [available[i] for i in top_idx_sorted]
    sel_corr  = corr[np.ix_(top_idx_sorted, top_idx_sorted)]

    return sel_names, [[round(float(v), 3) for v in row] for row in sel_corr]


