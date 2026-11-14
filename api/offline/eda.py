"""CLI entry point for EDA + preprocessing report (make eda)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> None:
    parser = argparse.ArgumentParser(description="AFD V2 EDA + preprocessing report")
    parser.add_argument(
        "--dataset-dir", default=None, help="CIC-IDS2017 CSV directory (default: from settings)"
    )
    parser.add_argument(
        "--output",
        default="data/eda_summary.json",
        help="Output JSON path (default: data/eda_summary.json)",
    )
    args = parser.parse_args()

    from config.settings import get_settings
    from pipeline.eda.exploratory_analysis import run_eda

    settings = get_settings()
    dataset_dir = args.dataset_dir or settings.bundled_dataset_dir
    run_eda(dataset_dir, args.output)


if __name__ == "__main__":
    main()
