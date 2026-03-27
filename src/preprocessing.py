"""
preprocessing.py
================
Text preprocessing that deliberately PRESERVES special characters
and emoji. Unlike standard NLP pipelines, we do NOT strip punctuation
because it is a core signal for AI-generated vs. human text.

Two modes:
  - 'preserve_special'  (default) — keep all punct/emoji, normalize only
  - 'full_clean'        — traditional NLP cleaning (for ablation studies)
"""

import re
import string
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

try:
    import emoji as emoji_lib
    EMOJI_AVAILABLE = True
except ImportError:
    EMOJI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Sample dataset generator (used when no CSV is available)
# ---------------------------------------------------------------------------

def generate_sample_dataset(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic dataset for demonstration.
    Label 0 = human/real, Label 1 = fake/AI-generated.
    """
    rng = np.random.default_rng(random_state)

    human_templates = [
        "omg this is literally the BEST thing i've ever bought!! 😍😍",
        "ok so i was skeptical at first but... WOW. total game changer 🙌",
        "tbh i bought this on a whim and i'm SO glad i did!! worth every penny!!",
        "not gonna lie, i've had better... but for the price? it's fine i guess 🤷",
        "LOVE LOVE LOVE this product!! bought 3 already lol 😂",
        "my mom got this for me and honestly?? it slaps. 10/10 no notes ✨",
        "shipping took forever but the product is great!! 5 stars ⭐⭐⭐⭐⭐",
        "meh. it's ok i guess. nothing special tbh... expected more for the price 😐",
        "UPDATE: it broke after 2 weeks. DO NOT BUY 😡😡",
        "yoooo this actually works?? i'm shook fr fr 💀",
    ]

    ai_templates = [
        "This product exceeded all my expectations. The quality is outstanding and "
        "the attention to detail is commendable. I would highly recommend it.",
        "I am thoroughly impressed with this purchase. The craftsmanship is superb—"
        "truly a testament to modern engineering and design principles.",
        "After extensive use, I can confidently say this product delivers on all its "
        "promises. The build quality is exceptional and the performance is remarkable.",
        "This is an excellent product that offers great value for money. The customer "
        "service was also very responsive and helpful throughout the process.",
        "I have been using this product for several weeks now, and I am pleased to "
        "report that it continues to perform admirably in all respects.",
        "The product arrived promptly and was well-packaged. Upon first use, it became "
        "clear that considerable thought had gone into its design and functionality.",
        "Overall, this is a well-crafted product that I would recommend to anyone "
        "seeking a reliable and high-quality solution for their needs.",
        "My experience with this product has been uniformly positive. It performs "
        "exactly as described and the quality justifies the price point.",
    ]

    records = []
    for _ in range(n_samples // 2):
        idx = rng.integers(0, len(human_templates))
        # Add random variation
        text = human_templates[idx]
        if rng.random() > 0.5:
            text += " " + rng.choice(["!!!", "😊", "would recommend!", "great buy!"])
        records.append({"text": text, "label": 0})

    for _ in range(n_samples // 2):
        idx = rng.integers(0, len(ai_templates))
        text = ai_templates[idx]
        records.append({"text": text, "label": 1})

    df = pd.DataFrame(records).sample(frac=1, random_state=random_state).reset_index(drop=True)
    print(f"[generate_sample_dataset] Created {len(df)} samples  "
          f"(real={sum(df.label==0)}, fake/AI={sum(df.label==1)})")
    return df


# ---------------------------------------------------------------------------
# Cleaning functions
# ---------------------------------------------------------------------------

def preserve_special_clean(text: str) -> str:
    """
    Minimal cleaning that PRESERVES:
      - Punctuation (!, ?, ..., —, etc.)
      - Emoji
      - Mixed case
      - Non-ASCII characters

    Only removes:
      - Leading/trailing whitespace
      - Duplicate spaces
      - HTML tags
      - Null bytes
    """
    if not isinstance(text, str):
        return ""

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove null bytes
    text = text.replace('\x00', '')
    # Normalize whitespace (collapse multiple spaces/newlines)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    return text


def full_clean(text: str) -> str:
    """
    Traditional NLP cleaning — strips special chars, lowercases, etc.
    Use this ONLY for ablation / baseline comparison.
    NOT recommended for AI-detection tasks.
    """
    if not isinstance(text, str):
        return ""

    # Remove HTML
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' URL ', text)
    # Remove emoji
    if EMOJI_AVAILABLE:
        text = emoji_lib.replace_emoji(text, replace='')
    else:
        text = re.sub(u'[\U0001F300-\U0001FAFF]', '', text)
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove non-ASCII
    text = text.encode('ascii', errors='ignore').decode()
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_dataset(
    csv_path: str = None,
    text_col: str = "text",
    label_col: str = "label",
    cleaning_mode: str = "preserve_special",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
):
    """
    Load and split dataset into train/val/test sets.

    Parameters
    ----------
    csv_path       : path to CSV file. If None, generates a synthetic dataset.
    text_col       : name of the text column
    label_col      : name of the label column (0=real, 1=fake/AI)
    cleaning_mode  : 'preserve_special' (recommended) or 'full_clean'
    test_size      : fraction for test set
    val_size       : fraction for validation set (from training data)
    random_state   : seed for reproducibility

    Returns
    -------
    dict with keys: train_df, val_df, test_df, full_df
    """
    if csv_path:
        df = pd.read_csv(csv_path)
        print(f"[load_dataset] Loaded {len(df)} rows from {csv_path}")
    else:
        print("[load_dataset] No CSV path provided — generating synthetic dataset.")
        df = generate_sample_dataset(random_state=random_state)

    # Validate columns
    assert text_col in df.columns, f"Column '{text_col}' not found. Available: {df.columns.tolist()}"
    assert label_col in df.columns, f"Column '{label_col}' not found. Available: {df.columns.tolist()}"

    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    df = df.dropna(subset=["text"]).reset_index(drop=True)

    # Apply cleaning
    clean_fn = preserve_special_clean if cleaning_mode == "preserve_special" else full_clean
    df["text_clean"] = df["text"].apply(clean_fn)

    # Label distribution
    dist = df["label"].value_counts().to_dict()
    print(f"[load_dataset] Label distribution: {dist}")

    # Split
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["label"]
    )
    train_df, val_df = train_test_split(
        train_df, test_size=val_size / (1 - test_size),
        random_state=random_state, stratify=train_df["label"]
    )

    print(f"[load_dataset] Split → train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    return {
        "train_df": train_df.reset_index(drop=True),
        "val_df":   val_df.reset_index(drop=True),
        "test_df":  test_df.reset_index(drop=True),
        "full_df":  df,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    splits = load_dataset()
    print("\nSample rows from training set:")
    print(splits["train_df"][["text", "label"]].head(5).to_string(index=False))
