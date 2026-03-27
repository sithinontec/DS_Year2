"""
train_ensemble.py
=================
Ensemble ML pipeline:
  Random Forest + XGBoost + Gradient Boosting
  All trained on the FULL feature set:
    TF-IDF (word + char) + SpecialChar numeric features

Ensemble models are particularly good at capturing non-linear interactions
between features — e.g., "many emojis AND short text AND no caps" = human.
"""

import os
import sys
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, roc_auc_score

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("[WARNING] xgboost not installed. Skipping XGBoost model.")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.preprocessing import load_dataset
from src.train_classical import build_feature_matrix


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_ensemble_models(
    splits: dict,
    bundle: dict = None,
    save_dir: str = "models",
):
    """
    Train ensemble models. Reuses feature bundle from classical training if provided.
    """
    os.makedirs(save_dir, exist_ok=True)

    train_texts = splits["train_df"]["text_clean"].tolist()
    val_texts   = splits["val_df"]["text_clean"].tolist()
    test_texts  = splits["test_df"]["text_clean"].tolist()

    y_train = splits["train_df"]["label"].values
    y_val   = splits["val_df"]["label"].values
    y_test  = splits["test_df"]["label"].values

    # Try to load existing bundle
    if bundle is None:
        bundle_path = os.path.join(save_dir, "feature_bundle.pkl")
        if os.path.exists(bundle_path):
            bundle = joblib.load(bundle_path)
            print(f"[train_ensemble] Loaded feature bundle from {bundle_path}")
        else:
            print("[train_ensemble] No bundle found — building feature matrices from scratch.")

    if bundle:
        X_train, _ = build_feature_matrix(train_texts, transformer_bundle=bundle, fit=False)
        X_val, _   = build_feature_matrix(val_texts, transformer_bundle=bundle, fit=False)
        X_test, _  = build_feature_matrix(test_texts, transformer_bundle=bundle, fit=False)
    else:
        X_train, bundle = build_feature_matrix(train_texts, fit=True)
        X_val, _        = build_feature_matrix(val_texts, transformer_bundle=bundle, fit=False)
        X_test, _       = build_feature_matrix(test_texts, transformer_bundle=bundle, fit=False)

    # Convert sparse to dense for tree-based models
    # Use toarray() only if memory allows; otherwise use CSR directly (XGBoost supports it)
    print(f"  Feature matrix (train): {X_train.shape}")

    results = {}

    # --- Random Forest ---
    print("\n" + "=" * 60)
    print("TRAINING: Random Forest")
    print("=" * 60)
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    _eval_and_save(rf, "RandomForest", X_val, y_val, X_test, y_test, save_dir, results)

    # --- XGBoost ---
    if XGB_AVAILABLE:
        print("\n" + "=" * 60)
        print("TRAINING: XGBoost")
        print("=" * 60)

        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric='logloss',
            tree_method='hist',  # fast, supports sparse
            random_state=42,
            n_jobs=-1,
        )
        xgb.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        _eval_and_save(xgb, "XGBoost", X_val, y_val, X_test, y_test, save_dir, results)

    # --- Gradient Boosting (sklearn) ---
    # Note: sklearn GB does not support sparse input natively for all methods
    # We use a subset for speed
    print("\n" + "=" * 60)
    print("TRAINING: Gradient Boosting (sklearn) — using special char features only")
    print("=" * 60)

    from src.feature_engineering import SpecialCharFeatureExtractor
    from sklearn.preprocessing import MinMaxScaler

    extractor = SpecialCharFeatureExtractor()
    scaler    = MinMaxScaler()

    X_train_sc = scaler.fit_transform(extractor.transform(train_texts).values)
    X_val_sc   = scaler.transform(extractor.transform(val_texts).values)
    X_test_sc  = scaler.transform(extractor.transform(test_texts).values)

    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    gb.fit(X_train_sc, y_train)
    _eval_and_save(gb, "GradientBoosting_SpecialChar", X_val_sc, y_val,
                   X_test_sc, y_test, save_dir, results)

    # Feature importance for GradientBoosting on special char features
    feat_names = extractor.feature_names
    importances = gb.feature_importances_
    top_idx = np.argsort(importances)[::-1][:15]
    print("\n  Top 15 Special Character Features (GradientBoosting):")
    print(f"  {'Feature':<30} {'Importance':>10}")
    print("  " + "-" * 42)
    for i in top_idx:
        print(f"  {feat_names[i]:<30} {importances[i]:>10.4f}")

    print("\n" + "=" * 60)
    print("ENSEMBLE MODELS — SUMMARY")
    print("=" * 60)
    for name, r in results.items():
        print(f"  {name:<40} val_acc={r['val_acc']:.4f}  test_acc={r['test_acc']:.4f}  test_auc={r.get('test_auc', 'N/A')}")

    return results


def _eval_and_save(model, name, X_val, y_val, X_test, y_test, save_dir, results):
    val_acc  = model.score(X_val, y_val)
    test_acc = model.score(X_test, y_test)

    test_auc = None
    try:
        proba    = model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, proba)
    except Exception:
        pass

    print(f"  Val Accuracy : {val_acc:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    if test_auc:
        print(f"  Test ROC-AUC : {test_auc:.4f}")
    print("\n  Classification Report (Test):")
    print(classification_report(y_test, model.predict(X_test),
                                target_names=["Real", "Fake/AI"]))

    path = os.path.join(save_dir, f"{name}.pkl")
    joblib.dump(model, path)
    print(f"  💾 Saved → {path}")

    results[name] = {"val_acc": val_acc, "test_acc": test_acc, "test_auc": test_auc}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    splits = load_dataset()
    train_ensemble_models(splits)
