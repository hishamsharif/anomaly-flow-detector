#!/usr/bin/env python3
"""
Standalone prediction examiner — no server required.

Loads the active model directly from the artifact database and batch-scores
a CSV or PCAP file through the full 3-stage pipeline.

Usage:
  python inference/inference_offline.py --csv path/to/flows.csv
  python inference/inference_offline.py --pcap path/to/capture.pcap
  python inference/inference_offline.py --csv flows.csv --output results.csv
  python inference/inference_offline.py --csv flows.csv --db data/afd.db
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path


def _load_active_bundle(db_path: str):
    """Query SQLite for the active ModelVersion, then load its artifacts."""
    import sqlite3

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT path FROM model_versions WHERE status = 'active' LIMIT 1"
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        print("ERROR: No active model found — run training first.", file=sys.stderr)
        sys.exit(1)

    artifact_path = row[0]
    if not artifact_path or not Path(artifact_path).exists():
        print(f"ERROR: Artifact path not found: {artifact_path}", file=sys.stderr)
        sys.exit(1)

    from core.store import ModelStore
    return ModelStore.get().load_into_memory(artifact_path)


def _parse_input(path: str, feature_names: list[str]):
    """Dispatch to CSV or PCAP adapter based on file extension."""
    p = Path(path)
    if not p.exists():
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(1)

    content = p.read_bytes()
    suffix = p.suffix.lower()

    if suffix == ".csv":
        from inference.adapters.csv_adapter import parse_csv
        return parse_csv(content, feature_names)
    elif suffix in (".pcap", ".cap"):
        from inference.adapters.pcap_adapter import parse_pcap
        return parse_pcap(content, feature_names)
    else:
        print(f"ERROR: Unsupported file type '{suffix}' — use .csv or .pcap", file=sys.stderr)
        sys.exit(1)


def _run(input_path: str, db_path: str, output_path: str | None) -> None:
    print(f"Loading active model from {db_path} …")
    bundle = _load_active_bundle(db_path)
    print(f"Model loaded: {len(bundle.feature_names)} features")

    print(f"Parsing {input_path} …")
    valid_rows, rejected = _parse_input(input_path, bundle.feature_names)
    print(f"  {len(valid_rows)} valid flows, {len(rejected)} rejected")

    if not valid_rows:
        print("No valid flows to score.")
        for r in rejected[:5]:
            print(f"  Row {r['row']}: {r['reason']}")
        return

    print("Running inference …")
    t0 = time.perf_counter()

    from inference.pipeline import run_batch_inference
    results = run_batch_inference(valid_rows, bundle)

    elapsed = time.perf_counter() - t0
    fps = len(results) / elapsed if elapsed > 0 else float("inf")
    print(f"  {len(results)} flows scored in {elapsed:.2f}s  ({fps:,.0f} flows/s)")

    # Tally
    safe = sum(1 for r in results if r.consensus == "SAFE")
    known = sum(1 for r in results if r.consensus == "KNOWN_ATTACK")
    novel = sum(1 for r in results if r.consensus == "NOVEL_ATTACK")
    print(f"  SAFE={safe}  KNOWN_ATTACK={known}  NOVEL_ATTACK={novel}")

    if output_path:
        fieldnames = [
            "flow_id", "consensus", "severity",
            "predicted_label", "classifier_confidence", "ae_score", "ae_threshold",
        ]
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "flow_id": r.flow_id,
                    "consensus": r.consensus,
                    "severity": r.severity,
                    "predicted_label": r.predicted_label,
                    "classifier_confidence": r.classifier_confidence,
                    "ae_score": r.ae_score,
                    "ae_threshold": r.ae_threshold,
                })
        print(f"Results saved → {output_path}")
    else:
        # Print first 20 non-SAFE rows to stdout
        attacks = [r for r in results if r.consensus != "SAFE"][:20]
        if attacks:
            print("\nTop alerts:")
            for r in attacks:
                print(
                    f"  [{r.severity.upper():8}] flow={r.flow_id}  "
                    f"{r.consensus}  ae={r.ae_score:.4f}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="AFD standalone examiner")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv", metavar="FILE", help="CSV file to score")
    group.add_argument("--pcap", metavar="FILE", help="PCAP file to score")
    parser.add_argument("--output", metavar="FILE", help="Save results to CSV")
    parser.add_argument(
        "--db",
        metavar="FILE",
        default="data/afd.db",
        help="SQLite DB path (default: data/afd.db)",
    )
    args = parser.parse_args()

    input_path = args.csv or args.pcap
    _run(input_path, args.db, args.output)


if __name__ == "__main__":
    main()
