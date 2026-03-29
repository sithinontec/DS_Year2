"""
train_classical.py
==================
Trains two models for OR vs CG (computer-generated) review detection:

  1. Logistic Regression  — fast, interpretable, strong baseline
  2. Random Forest        — captures non-linear feature interactions

Both are trained on the same feature matrix:
  - Word TF-IDF  (preserves punctuation tokens like !, !!, ...)
  - Char TF-IDF  (captures n-grams like .T, !., !! directly)
  - SpecialCharFeatureExtractor  (39 numeric features)
"""

import os
import sys
import joblib
import numpy as np
import scipy.sparse as sp
from scipy.sparse import hstack, csr_matrix

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.preprocessing import load_dataset
from src.feature_engineering import SpecialCharFeatureExtractor


# ── Feature transformer ──────────────────────────────────────────────── #

class SpecialCharTransformer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible wrapper around SpecialCharFeatureExtractor."""
    def __init__(self):
        self.extractor = SpecialCharFeatureExtractor()
        self.scaler    = MinMaxScaler()

    def fit(self, X, y=None):
        self.scaler.fit(self.extractor.transform(X).values)
        return self

    def transform(self, X):
        return self.scaler.transform(self.extractor.transform(X).values)

    def get_feature_names_out(self):
        return self.extractor.feature_names


def build_feature_matrix(texts, transformer_bundle=None, fit=True):
    """
    Build combined sparse feature matrix: word TF-IDF + char TF-IDF + special chars.

    Parameters
    ----------
    texts              : list of text strings
    transformer_bundle : fitted bundle dict (required when fit=False)
    fit                : True = fit and return bundle, False = transform only

    Returns
    -------
    X : scipy sparse matrix  (n_samples, n_features)
    bundle : dict of fitted transformers
    """
    texts = list(texts)

    if fit:
        tfidf_word = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 2),
            max_features=30000,
            sublinear_tf=True,
            min_df=2,
            token_pattern=r'(?u)\b\w+\b|[!?.]{1,3}|[^\w\s]',
        )
        tfidf_char = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(2, 4),
            max_features=20000,
            sublinear_tf=True,
            min_df=3,
        )
        special_transformer = SpecialCharTransformer()

        X_word    = tfidf_word.fit_transform(texts)
        X_char    = tfidf_char.fit_transform(texts)
        X_special = csr_matrix(special_transformer.fit_transform(texts))
        bundle    = {
            'tfidf_word':          tfidf_word,
            'tfidf_char':          tfidf_char,
            'special_transformer': special_transformer,
        }
    else:
        X_word    = transformer_bundle['tfidf_word'].transform(texts)
        X_char    = transformer_bundle['tfidf_char'].transform(texts)
        X_special = csr_matrix(transformer_bundle['special_transformer'].transform(texts))
        bundle    = transformer_bundle

    X = hstack([X_word, X_char, X_special])
    print(f"  Feature matrix shape: {X.shape}")
    return X, bundle


# ── Training ─────────────────────────────────────────────────────────── #

def train_models(splits: dict, save_dir: str = "models"):
    """
    Train Logistic Regression and Random Forest on splits from load_dataset().
    Saves models and feature bundle to save_dir. Returns (results dict, bundle).
    """
    os.makedirs(save_dir, exist_ok=True)

    train_texts = splits["train_df"]["text_clean"].tolist()
    val_texts   = splits["val_df"]["text_clean"].tolist()
    test_texts  = splits["test_df"]["text_clean"].tolist()

    y_train = splits["train_df"]["label_num"].values
    y_val   = splits["val_df"]["label_num"].values
    y_test  = splits["test_df"]["label_num"].values

    print("\n" + "=" * 60)
    print("BUILDING FEATURE MATRICES")
    print("=" * 60)
    print("  Training set:")
    X_train, bundle = build_feature_matrix(train_texts, fit=True)
    print("  Validation set:")
    X_val,   _      = build_feature_matrix(val_texts,  transformer_bundle=bundle, fit=False)
    print("  Test set:")
    X_test,  _      = build_feature_matrix(test_texts, transformer_bundle=bundle, fit=False)

    joblib.dump(bundle, os.path.join(save_dir, "feature_bundle.pkl"))
    print(f"\n  ✅ Feature bundle saved → {save_dir}/feature_bundle.pkl")

    results = {}

    # ── Logistic Regression ──────────────────────────────────────────── #
    print("\n" + "=" * 60)
    print("TRAINING: Logistic Regression")
    print("=" * 60)
    lr = LogisticRegression(C=1.0, max_iter=1000, solver='saga', class_weight='balanced')
    lr.fit(X_train, y_train)
    _eval_and_save(lr, "LogisticRegression", X_val, y_val, X_test, y_test,
                   save_dir, results, has_proba=True)

    # ── Random Forest ────────────────────────────────────────────────── #
    # Trained on the 39 special-char features only — avoids the 5+ GB
    # memory cost of converting the full sparse TF-IDF matrix to dense.
    # The special-char features are the most interpretable signals anyway.
    print("\n" + "=" * 60)
    print("TRAINING: Random Forest (special-char features)")
    print("=" * 60)
    _sc = MinMaxScaler()
    _ext = SpecialCharFeatureExtractor()
    X_train_sc = _sc.fit_transform(_ext.transform(train_texts).values)
    X_val_sc   = _sc.transform(_ext.transform(val_texts).values)
    X_test_sc  = _sc.transform(_ext.transform(test_texts).values)
    print(f"  Feature matrix shape: {X_train_sc.shape}")
    joblib.dump(_sc,  os.path.join(save_dir, 'RandomForest_scaler.pkl'))
    joblib.dump(_ext, os.path.join(save_dir, 'RandomForest_extractor.pkl'))
    rf = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train_sc, y_train)
    _eval_and_save(rf, "RandomForest", X_val_sc, y_val, X_test_sc, y_test,
                   save_dir, results, has_proba=True)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, r in results.items():
        auc_str = f"  auc={r['test_auc']:.4f}" if r.get('test_auc') else ""
        print(f"  {name:<22} val={r['val_acc']:.4f}  test={r['test_acc']:.4f}{auc_str}")

    return results, bundle


def _eval_and_save(model, name, X_val, y_val, X_test, y_test,
                   save_dir, results, has_proba=True):
    val_acc  = model.score(X_val,  y_val)
    test_acc = model.score(X_test, y_test)
    test_auc = None

    if has_proba:
        try:
            test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        except Exception:
            pass

    print(f"  Val Accuracy : {val_acc:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    if test_auc:
        print(f"  Test ROC-AUC : {test_auc:.4f}")
    print("\n  Classification Report (Test):")
    print(classification_report(y_test, model.predict(X_test),
                                target_names=["OR (real)", "CG (generated)"]))

    path = os.path.join(save_dir, f"{name}.pkl")
    joblib.dump(model, path)
    print(f"  💾 Saved → {path}")

    results[name] = {"val_acc": val_acc, "test_acc": test_acc, "test_auc": test_auc}


# ── CLI ──────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    splits = load_dataset()
    train_models(splits)