"""
preprocessing.py
================
Loads and prepares fake_reviews_dataset_2022.csv.

Columns: category, rating, label (OR/CG), text_
Label mapping: OR -> 0 (real/human), CG -> 1 (computer-generated)

Cleaning strategy:
  - PRESERVE all punctuation and special characters — they are features
  - Only strip HTML tags, null bytes, and collapse excessive whitespace
  - Do NOT lowercase for TF-IDF (ALL CAPS is a feature)
  - Do NOT strip punctuation (!, ?, repeated punct are features)
  - Do NOT remove non-ASCII (non-ASCII ratio is a feature)
"""

import re
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# ──────────────────────────────────────────────────────────────────────── #
# Cleaning
# ──────────────────────────────────────────────────────────────────────── #

def preserve_special_clean(text: str) -> str:
    """
    Minimal cleaning that preserves ALL special characters.
    Only removes HTML, null bytes, and collapses whitespace.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)   # strip HTML tags
    text = text.replace("\x00", "")          # null bytes
    text = re.sub(r"[ \t]+", " ", text)      # collapse spaces/tabs
    text = re.sub(r"\n{3,}", "\n\n", text)   # collapse excessive newlines
    return text.strip()


# ──────────────────────────────────────────────────────────────────────── #
# Dataset loader
# ──────────────────────────────────────────────────────────────────────── #

def load_dataset(
    csv_path: str = None,
    test_size: float = 0.20,
    val_size:  float = 0.10,
    random_state: int = 42,
    category_filter: str = None,   # e.g. "Books_5" to train on one category
) -> dict:
    """
    Load fake_reviews_dataset_2022.csv and split into train/val/test.

    Parameters
    ----------
    csv_path         : path to CSV; defaults to data/fake_reviews_dataset_2022.csv
    test_size        : fraction held out for testing
    val_size         : fraction of training data held out for validation
    random_state     : reproducibility seed
    category_filter  : if set, keep only this product category

    Returns
    -------
    dict with keys: train_df, val_df, test_df, full_df, label_map
    """
    # ── Locate CSV ──────────────────────────────────────────────────── #
    if csv_path is None:
        candidates = [
            "data/fake_reviews_dataset_2022.csv",
            "/mnt/user-data/uploads/fake_reviews_dataset_2022.csv",
            os.path.join(os.path.dirname(__file__), "..", "data", "fake_reviews_dataset_2022.csv"),
        ]
        for c in candidates:
            if os.path.exists(c):
                csv_path = c
                break
        if csv_path is None:
            raise FileNotFoundError(
                "Dataset not found. Pass csv_path= explicitly, or place the file at "
                "data/fake_reviews_dataset_2022.csv"
            )

    df = pd.read_csv(csv_path)
    print(f"[load_dataset] Loaded {len(df):,} rows from {csv_path}")

    # ── Validate ────────────────────────────────────────────────────── #
    required = {"category", "rating", "label", "text_"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found: {df.columns.tolist()}")

    # ── Optional category filter ─────────────────────────────────────── #
    if category_filter:
        df = df[df["category"] == category_filter].copy()
        print(f"[load_dataset] Filtered to category '{category_filter}': {len(df):,} rows")

    # ── Clean & remap labels ─────────────────────────────────────────── #
    df = df.dropna(subset=["text_"]).copy()
    df["text_clean"] = df["text_"].apply(preserve_special_clean)

    # OR -> 0 (real/human), CG -> 1 (computer-generated)
    label_map = {"OR": 0, "CG": 1}
    df["label_int"] = df["label"].map(label_map)

    if df["label_int"].isna().any():
        unique_labels = df["label"].unique()
        raise ValueError(
            f"Unexpected label values: {unique_labels}. Expected 'OR' and 'CG'."
        )

    # Keep useful metadata columns
    df = df[["text_", "text_clean", "label", "label_int", "category", "rating"]].copy()
    df = df.rename(columns={"label_int": "label_num"})

    # ── Label distribution ───────────────────────────────────────────── #
    dist = df["label"].value_counts().to_dict()
    print(f"[load_dataset] Label distribution: {dist}  "
          f"(OR=real/human → 0, CG=generated → 1)")

    # ── Train / val / test split ─────────────────────────────────────── #
    train_df, test_df = train_test_split(
        df, test_size=test_size,
        random_state=random_state, stratify=df["label_num"]
    )
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_size / (1.0 - test_size),
        random_state=random_state,
        stratify=train_df["label_num"]
    )

    print(f"[load_dataset] Split → "
          f"train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

    return {
        "train_df":  train_df.reset_index(drop=True),
        "val_df":    val_df.reset_index(drop=True),
        "test_df":   test_df.reset_index(drop=True),
        "full_df":   df.reset_index(drop=True),
        "label_map": label_map,      # {"OR": 0, "CG": 1}
        "label_names": ["OR (real)", "CG (generated)"],
    }


# ──────────────────────────────────────────────────────────────────────── #
# CLI
# ──────────────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    splits = load_dataset()
    print("\nSample training rows:")
    print(splits["train_df"][["text_", "label", "label_num"]].head(6).to_string(index=False))
    print("\nCategory breakdown:")
    print(splits["full_df"]["category"].value_counts())