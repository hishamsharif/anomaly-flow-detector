from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


def _load_csvs(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame()
    quoted = ", ".join(f"'{p}'" for p in paths)
    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM read_csv_auto([{quoted}], header=true, ignore_errors=true)").df()
    con.close()
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
    base = Path(dataset_dir)
    all_csvs = sorted(base.glob("*.csv"))
    if not all_csvs:
        raise FileNotFoundError(f"No CSV files found in {dataset_dir}")

    train_paths = [p for p in all_csvs if "Friday" not in p.name]
    val_paths   = [p for p in all_csvs if "Friday" in p.name and "Morning" in p.name]
    test_paths  = [p for p in all_csvs if "Friday" in p.name and "Afternoon" in p.name]

    # Fallback: if no Friday split, use 70/15/15 on train_paths
    if not val_paths and not test_paths:
        all_df = _clean(_load_csvs(train_paths))
        n = len(all_df)
        t, v = int(n * 0.7), int(n * 0.85)
        return all_df.iloc[:t], all_df.iloc[t:v], all_df.iloc[v:]

    train_df = _clean(_load_csvs(train_paths)) if train_paths else pd.DataFrame()
    val_df   = _clean(_load_csvs(val_paths))   if val_paths   else pd.DataFrame()
    test_df  = _clean(_load_csvs(test_paths))  if test_paths  else pd.DataFrame()

    return train_df, val_df, test_df
