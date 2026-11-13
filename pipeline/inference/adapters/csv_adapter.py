from __future__ import annotations

import io

import numpy as np
import pandas as pd


def parse_csv(
    content: bytes,
    feature_names: list[str],
) -> tuple[list[dict[str, float]], list[dict]]:
    """
    Parse CSV bytes and validate each row against the model's feature set (I3).

    The model's `feature_names.json` lists the final selected features (after
    variance filter → correlation filter → feature selector).  Every row in the
    input CSV must supply all of these features; missing or non-numeric values
    produce a `rejected_rows` entry instead.

    Returns:
        valid_rows   — list of {feature_name: float} dicts, one per accepted row
        rejected_rows — list of {row: int, reason: str} for rows that failed I3
    """
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        return [], [{"row": -1, "reason": f"CSV parse error: {exc}"}]

    # CIC-IDS2017 column names often have a leading space — strip them.
    df.columns = [c.strip() for c in df.columns]

    # Quick check: are any required features completely absent?
    missing_global = [f for f in feature_names if f not in df.columns]
    if missing_global:
        sample = missing_global[:3]
        suffix = f" … and {len(missing_global) - 3} more" if len(missing_global) > 3 else ""
        return [], [
            {"row": i, "reason": f"missing feature: {sample[0]}{suffix}"}
            for i in range(len(df))
        ]

    sub = df[feature_names]
    valid_rows: list[dict[str, float]] = []
    rejected_rows: list[dict] = []

    # Vectorised NaN detection: find rows with at least one NaN.
    nan_row_mask = sub.isnull().any(axis=1)
    nan_indices = set(np.where(nan_row_mask)[0].tolist())

    for i in range(len(sub)):
        if i in nan_indices:
            # Identify the first offending column for the error message.
            bad_col = sub.columns[sub.iloc[i].isnull()][0]
            rejected_rows.append({"row": i, "reason": f"NaN in feature: {bad_col}"})
        else:
            row_values: dict[str, float] = {}
            bad = False
            for name in feature_names:
                val = sub.iat[i, sub.columns.get_loc(name)]
                try:
                    row_values[name] = float(val)
                except (ValueError, TypeError):
                    rejected_rows.append({"row": i, "reason": f"non-numeric value in feature: {name}"})
                    bad = True
                    break
            if not bad:
                valid_rows.append(row_values)

    return valid_rows, rejected_rows
