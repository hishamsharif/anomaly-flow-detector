from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


def _load_csvs(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame()
    quoted = ", ".join(f"'{p}'" for p in paths)
    db_connection = duckdb.connect()
    df = db_connection.execute(f"SELECT * FROM read_csv_auto([{quoted}], header=true, ignore_errors=true)").df()
    db_connection.close()
    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    df = df.drop_duplicates()
    # Drop non-numeric columns except Label
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    keep = num_cols + (["Label"] if "Label" in df.columns and "Label" not in num_cols else [])
    df = df[keep]
    if "Label" not in df.columns:
        raise ValueError("No 'Label' column found in dataset")
    df["label"] = (df["Label"] != "BENIGN").astype(int)
    df = df.drop(columns=["Label"])
    return df.reset_index(drop=True)


def load_dataset(dataset_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train_df, val_df, test_df); each has numeric features + 'label' (0/1)."""
    dataset_dir_path = Path(dataset_dir)
    all_csvs = sorted(dataset_dir_path.glob("*.csv"))
    if not all_csvs:
        raise FileNotFoundError(f"No CSV files found in {dataset_dir}")

    train_paths = [p for p in all_csvs if "Friday" not in p.name]
    val_paths   = [p for p in all_csvs if "Friday" in p.name and "Morning" in p.name]
    test_paths  = [p for p in all_csvs if "Friday" in p.name and "Afternoon" in p.name]

    # Fallback: if no Friday split, use 70/15/15 on train_paths
    if not val_paths and not test_paths:
        full_df = _clean(_load_csvs(train_paths))
        n = len(full_df)
        train_end_idx, val_end_idx = int(n * 0.7), int(n * 0.85)
        return full_df.iloc[:train_end_idx], full_df.iloc[train_end_idx:val_end_idx], full_df.iloc[val_end_idx:]

    train_df = _clean(_load_csvs(train_paths)) if train_paths else pd.DataFrame()
    val_df   = _clean(_load_csvs(val_paths))   if val_paths   else pd.DataFrame()
    test_df  = _clean(_load_csvs(test_paths))  if test_paths  else pd.DataFrame()

    return train_df, val_df, test_df
