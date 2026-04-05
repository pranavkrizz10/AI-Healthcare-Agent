"""
preprocess.py
=============
Preprocessing pipeline for the large symptom-disease dataset.

Expected CSV format
-------------------
  - First column  : 'diseases'  (string label, e.g. "Anxiety")
  - Remaining 377 : binary symptom columns (0 / 1), already one-hot encoded
  - ~246,000 rows, 773 unique diseases

What this script does
---------------------
  1. Loads the raw CSV
  2. Validates structure (checks for the 'diseases' column, binary values)
  3. Cleans column names (lowercase, strip whitespace)
  4. Renames 'diseases' -> 'disease' for consistency with the rest of the pipeline
  5. Drops rows where the disease label is null
  6. Reports class distribution so you can spot any imbalance
  7. Saves:
       data/processed_data.csv     -- cleaned dataset, ready for train.py
       models/symptom_list.json    -- ordered list of symptom column names

Usage
-----
    python src/preprocess.py
    python src/preprocess.py --input data/my_custom_file.csv
"""

import os
import json
import argparse
import pandas as pd
import numpy as np

# -- Paths ---------------------------------------------------------------------
RAW_PATH          = os.path.join("data",   "raw_sympt.csv")
PROCESSED_PATH    = os.path.join("data",   "processed_data.csv")
SYMPTOM_LIST_PATH = os.path.join("models", "symptom_list.json")

os.makedirs("data",   exist_ok=True)
os.makedirs("models", exist_ok=True)


# -- Load ----------------------------------------------------------------------
def load_raw(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Place your CSV there, or pass --input <path>."
        )
    print(f"[preprocess] Loading '{path}' ...")
    df = pd.read_csv(path, low_memory=False)
    print(f"[preprocess] Raw shape : {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


# -- Validate ------------------------------------------------------------------
def validate(df: pd.DataFrame) -> None:
    """Raise clear errors for common structural problems."""
    label_candidates = [c for c in df.columns
                        if c.strip().lower() in ("disease", "diseases", "prognosis", "label")]
    if not label_candidates:
        raise ValueError(
            "No disease label column found.\n"
            "Expected a column named 'diseases', 'disease', or 'prognosis'."
        )

    label_col = label_candidates[0]
    sym_cols  = [c for c in df.columns if c != label_col]
    sample    = df[sym_cols].head(500)
    unique_vals = set(sample.values.ravel())
    non_binary  = unique_vals - {0, 1, 0.0, 1.0, "0", "1"}
    if non_binary:
        print(f"[preprocess] WARNING: Unexpected values in symptom columns: "
              f"{list(non_binary)[:10]}  -- will coerce to int.")


# -- Clean ---------------------------------------------------------------------
def clean(df: pd.DataFrame) -> tuple:
    # Normalise ALL column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Rename label column to canonical 'disease'
    for candidate in ("diseases", "prognosis", "label"):
        if candidate in df.columns:
            df = df.rename(columns={candidate: "disease"})
            break

    # Clean disease label strings
    df["disease"] = (
        df["disease"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )

    # Drop rows with null/empty disease label
    before = len(df)
    df = df[df["disease"].notna() & (df["disease"] != "") & (df["disease"] != "nan")]
    if len(df) < before:
        print(f"[preprocess] Dropped {before - len(df):,} rows with null disease label.")

    # Coerce symptom columns to integer 0/1
    sym_cols = [c for c in df.columns if c != "disease"]
    df[sym_cols] = df[sym_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.int8)

    # Drop symptom columns that are all zeros (carry no information)
    all_zero = [c for c in sym_cols if df[c].sum() == 0]
    if all_zero:
        df = df.drop(columns=all_zero)
        print(f"[preprocess] Dropped {len(all_zero)} all-zero symptom columns.")

    sym_cols = [c for c in df.columns if c != "disease"]

    print(f"[preprocess] Clean shape : {df.shape[0]:,} rows x "
          f"{len(sym_cols)} symptoms x {df['disease'].nunique()} diseases")
    return df, sym_cols


# -- Class Distribution Report -------------------------------------------------
def report_distribution(df: pd.DataFrame) -> None:
    counts = df["disease"].value_counts()
    print(f"\n[preprocess] Class distribution summary:")
    print(f"  Total diseases  : {len(counts)}")
    print(f"  Rows per disease: min={counts.min()}, max={counts.max()}, median={counts.median():.0f}")

    very_rare = counts[counts < 10]
    if len(very_rare):
        print(f"  WARNING: {len(very_rare)} disease(s) have fewer than 10 rows -- "
              "consider dropping or oversampling them.")

    print(f"\n  Top 5 most common:")
    for disease, n in counts.head(5).items():
        print(f"    {disease:<45} {n:>6,} rows")
    print(f"\n  Top 5 rarest:")
    for disease, n in counts.tail(5).items():
        print(f"    {disease:<45} {n:>6,} rows")


# -- Save ----------------------------------------------------------------------
def save(df: pd.DataFrame, sym_cols: list) -> None:
    col_order = sym_cols + ["disease"]
    df[col_order].to_csv(PROCESSED_PATH, index=False)
    print(f"\n[preprocess] Saved processed data --> {PROCESSED_PATH}")

    with open(SYMPTOM_LIST_PATH, "w") as f:
        json.dump(sym_cols, f, indent=2)
    print(f"[preprocess] Saved symptom list   --> {SYMPTOM_LIST_PATH}  ({len(sym_cols)} symptoms)")


# -- Pipeline ------------------------------------------------------------------
def run_pipeline(raw_path: str = RAW_PATH) -> pd.DataFrame:
    df = load_raw(raw_path)
    validate(df)
    df, sym_cols = clean(df)
    report_distribution(df)
    save(df, sym_cols)
    print("\n[preprocess] Done. Run  python src/train.py  next.")
    return df


# -- Entry Point ---------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess the symptom-disease CSV.")
    parser.add_argument("--input", default=RAW_PATH,
                        help=f"Path to raw CSV (default: {RAW_PATH})")
    args = parser.parse_args()
    run_pipeline(args.input)