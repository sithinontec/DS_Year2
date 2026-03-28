"""
train_classical.py
==================
Classical ML pipeline:
  TF-IDF (char + word n-grams, special chars preserved)
  + Special character feature vector
  → Logistic Regression / SVM / Naive Bayes

The key design:
  We use a FeatureUnion of:
    1. Word-level TF-IDF (captures semantic fake patterns)
    2. Char-level TF-IDF (captures punctuation n-grams like '!!', '...')
    3. SpecialCharFeatureExtractor (numeric punctuation/emoji features)
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.preprocessing import load_dataset
from src.feature_engineering import SpecialCharFeatureExtractor


# ---------------------------------------------------------------------------
# Custom transformer to wrap SpecialCharFeatureExtractor for sklearn
# ---------------------------------------------------------------------------

class SpecialCharTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible wrapper around SpecialCharFeatureExtractor.
    Returns a dense numpy array of special char features.
    """
    def __init__(self):
        self.extractor = SpecialCharFeatureExtractor()
        self.scaler    = MinMaxScaler()

    def fit(self, X, y=None):
        features = self.extractor.transform(X).values
        self.scaler.fit(features)
        return self

    def transform(self, X):
        features = self.extractor.transform(X).values
        return self.scaler.transform(features)

    def get_feature_names_out(self):
        return self.extractor.feature_names


class TextSelector(BaseEstimator, TransformerMixin):
    """Passes text through unchanged (needed for FeatureUnion)."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X


# ---------------------------------------------------------------------------
# Combined feature builder
# ---------------------------------------------------------------------------

def build_feature_matrix(texts, transformer_bundle=None, fit=True):
    """
    Build combined feature matrix from TF-IDF + SpecialChar features.

    Parameters
    ----------
    texts              : list/Series of raw text strings
    transformer_bundle : dict of fitted transformers (for transform-only)
    fit                : if True, fit transformers; else transform only

    Returns
    -------
    X_combined : scipy sparse matrix
    bundle     : dict of transformers (save for inference)
    """
    texts = list(texts)

    if fit:
        # Word TF-IDF (preserve special chars via analyzer='word')
        tfidf_word = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 2),
            max_features=30000,
            sublinear_tf=True,
            min_df=2,
            token_pattern=r'(?u)\b\w+\b|[!?.]{1,3}|[^\w\s]',  # keeps punctuation tokens
        )
        # Char TF-IDF (captures !! ..  -- patterns directly)
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

        bundle = {
            'tfidf_word':  tfidf_word,
            'tfidf_char':  tfidf_char,
            'special_transformer': special_transformer,
        }
    else:
        tfidf_word  = transformer_bundle['tfidf_word']
        tfidf_char  = transformer_bundle['tfidf_char']
        special_transformer = transformer_bundle['special_transformer']

        X_word    = tfidf_word.transform(texts)
        X_char    = tfidf_char.transform(texts)
        X_special = csr_matrix(special_transformer.transform(texts))
        bundle    = transformer_bundle

    X_combined = hstack([X_word, X_char, X_special])
    print(f"  Feature matrix shape: {X_combined.shape}")
    return X_combined, bundle


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

CLASSICAL_MODELS = {
    "LogisticRegression": LogisticRegression(
        C=1.0, max_iter=1000, solver='lbfgs', class_weight='balanced'
    ),
    "LinearSVC": LinearSVC(
        C=1.0, max_iter=2000, class_weight='balanced'
    ),
    "SVM_RBF": SVC(
        C=1.0, kernel='rbf', gamma='scale',
        class_weight='balanced', probability=True
    ),
    "ComplementNB": None,  # Built separately (needs non-negative)
}


def train_classical_models(
    splits: dict,
    save_dir: str = "models",
    verbose: bool = True,
):
    """
    Train all classical models on the splits dict from load_dataset().
    Saves models to save_dir. Returns results dict.
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
    X_val, _        = build_feature_matrix(val_texts, transformer_bundle=bundle, fit=False)
    print("  Test set:")
    X_test, _       = build_feature_matrix(test_texts, transformer_bundle=bundle, fit=False)

    # Save transformer bundle
    joblib.dump(bundle, os.path.join(save_dir, "feature_bundle.pkl"))
    print(f"\n  ✅ Feature bundle saved to {save_dir}/feature_bundle.pkl")

    results = {}

    # --- Logistic Regression ---
    print("\n" + "=" * 60)
    print("TRAINING: Logistic Regression")
    print("=" * 60)
    lr = LogisticRegression(C=1.0, max_iter=1000, solver='saga',
                            class_weight='balanced', n_jobs=-1)
    lr.fit(X_train, y_train)
    _eval_and_save(lr, "LogisticRegression", X_val, y_val, X_test, y_test,
                   save_dir, results)

    # --- LinearSVC ---
    print("\n" + "=" * 60)
    print("TRAINING: LinearSVC")
    print("=" * 60)
    svc = LinearSVC(C=1.0, max_iter=3000, class_weight='balanced')
    svc.fit(X_train, y_train)
    _eval_and_save(svc, "LinearSVC", X_val, y_val, X_test, y_test,
                   save_dir, results, has_proba=False)

    # --- ComplementNB (needs non-negative, use char TF-IDF only) ---
    print("\n" + "=" * 60)
    print("TRAINING: ComplementNB (char TF-IDF only)")
    print("=" * 60)
    tfidf_char_nb = TfidfVectorizer(
        analyzer='char_wb', ngram_range=(2, 4),
        max_features=20000, sublinear_tf=True, min_df=3
    )
    X_train_nb = tfidf_char_nb.fit_transform(train_texts)
    X_val_nb   = tfidf_char_nb.transform(val_texts)
    X_test_nb  = tfidf_char_nb.transform(test_texts)
    cnb = ComplementNB(alpha=0.1)
    cnb.fit(X_train_nb, y_train)
    # Save the vectorizer alongside the model so evaluate.py can use it
    joblib.dump(tfidf_char_nb, os.path.join(save_dir, 'ComplementNB_vectorizer.pkl'))
    _eval_and_save(cnb, "ComplementNB", X_val_nb, y_val, X_test_nb, y_test,
                   save_dir, results)

    print("\n" + "=" * 60)
    print("CLASSICAL MODELS — SUMMARY")
    print("=" * 60)
    for name, r in results.items():
        print(f"  {name:<25} val_acc={r['val_acc']:.4f}  test_acc={r['test_acc']:.4f}", end="")
        if r.get('test_auc'):
            print(f"  test_auc={r['test_auc']:.4f}")
        else:
            print()

    return results, bundle


def _eval_and_save(model, name, X_val, y_val, X_test, y_test,
                   save_dir, results, has_proba=True):
    val_acc  = model.score(X_val, y_val)
    test_acc = model.score(X_test, y_test)

    test_auc = None
    if has_proba:
        try:
            proba = model.predict_proba(X_test)[:, 1]
            test_auc = roc_auc_score(y_test, proba)
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

    results[name] = {
        "val_acc": val_acc,
        "test_acc": test_acc,
        "test_auc": test_auc,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    splits = load_dataset()
    train_classical_models(splits)