"""
evaluate.py
===========
Unified evaluation, visualization, and reporting.

- Loads all saved models
- Generates comparison plots: accuracy, ROC curves, confusion matrices
- Feature importance analysis for special char features
- Single-text inference function for deployment
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.preprocessing import load_dataset
from src.feature_engineering import SpecialCharFeatureExtractor
from src.train_classical import build_feature_matrix


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def predict_single(text: str, model_path: str, bundle_path: str) -> dict:
    """
    Predict a single review text.
    Returns dict with label, confidence, and feature breakdown.
    """
    bundle    = joblib.load(bundle_path)
    model     = joblib.load(model_path)
    extractor = SpecialCharFeatureExtractor()

    X, _ = build_feature_matrix([text], transformer_bundle=bundle, fit=False)

    pred = model.predict(X)[0]
    conf = None
    if hasattr(model, 'predict_proba'):
        conf = model.predict_proba(X)[0][1]

    feats = extractor.extract(text)

    return {
        "text":           text,
        "prediction":     "Fake/AI" if pred == 1 else "Real/Human",
        "confidence":     round(conf, 4) if conf is not None else None,
        "ai_signal_score": round(feats["ai_signal_score"], 3),
        "n_emojis":       feats["n_emojis"],
        "n_exclaim":      feats["n_exclaim"],
        "n_emdash":       feats["n_emdash"],
        "n_ellipsis":     feats["n_ellipsis"],
        "n_repeated_punct": feats["n_repeated_punct"],
        "non_ascii_ratio": round(feats["non_ascii_ratio"], 4),
        "char_entropy":   round(feats["char_entropy"], 3),
    }


# ---------------------------------------------------------------------------
# Batch evaluation & plots
# ---------------------------------------------------------------------------

def evaluate_all_models(
    splits: dict,
    models_dir: str = "models",
    output_dir: str = "outputs",
):
    os.makedirs(output_dir, exist_ok=True)

    test_texts = splits["test_df"]["text_clean"].tolist()
    y_test     = splits["test_df"]["label"].values

    bundle_path = os.path.join(models_dir, "feature_bundle.pkl")
    if not os.path.exists(bundle_path):
        print("[evaluate] No feature bundle found. Run train_classical.py first.")
        return

    bundle = joblib.load(bundle_path)
    X_test, _ = build_feature_matrix(test_texts, transformer_bundle=bundle, fit=False)

    # Discover classical/ensemble models
    model_files = {
        name.replace(".pkl", ""): os.path.join(models_dir, name)
        for name in os.listdir(models_dir)
        if name.endswith(".pkl") and name not in (
            "feature_bundle.pkl", "bilstm_tokenizer.pkl",
            "bilstm_scaler.pkl", "bert_scaler.pkl"
        )
    }

    results = {}
    roc_data = {}

    for name, path in sorted(model_files.items()):
        try:
            model = joblib.load(path)
            preds = model.predict(X_test)
            acc   = (preds == y_test).mean()
            report= classification_report(y_test, preds, target_names=["Real", "Fake/AI"],
                                          output_dict=True)
            results[name] = {
                "accuracy": acc,
                "precision_fake": report["Fake/AI"]["precision"],
                "recall_fake":    report["Fake/AI"]["recall"],
                "f1_fake":        report["Fake/AI"]["f1-score"],
            }

            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_test)[:, 1]
                auc   = roc_auc_score(y_test, proba)
                fpr, tpr, _ = roc_curve(y_test, proba)
                results[name]["auc"] = auc
                roc_data[name] = (fpr, tpr, auc)

            print(f"  ✅ {name}: acc={acc:.4f}")
        except Exception as e:
            print(f"  ⚠️  {name}: {e}")

    if not results:
        print("[evaluate] No models found.")
        return

    results_df = pd.DataFrame(results).T.sort_values("accuracy", ascending=False)
    results_df.to_csv(os.path.join(output_dir, "model_comparison.csv"))
    print(f"\n  📄 Results saved → {output_dir}/model_comparison.csv")
    print("\n" + results_df.round(4).to_string())

    # --- Plot 1: Model accuracy comparison ---
    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = plt.cm.viridis(np.linspace(0.2, 0.85, len(results_df)))
    bars    = ax.barh(results_df.index, results_df["accuracy"], color=colors)
    ax.set_xlabel("Accuracy", fontsize=12)
    ax.set_title("Model Accuracy Comparison (Test Set)", fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    for bar, val in zip(bars, results_df["accuracy"]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_accuracy.png"), dpi=150)
    plt.close()

    # --- Plot 2: ROC curves ---
    if roc_data:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        cmap = plt.cm.tab10
        for i, (name, (fpr, tpr, auc)) in enumerate(roc_data.items()):
            ax.plot(fpr, tpr, color=cmap(i), lw=2, label=f"{name} (AUC={auc:.3f})")
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves — Fake/AI Review Detection", fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=150)
        plt.close()

    # --- Plot 3: Confusion matrix for best model ---
    best_name = results_df.index[0]
    best_path = model_files.get(best_name)
    if best_path:
        best_model = joblib.load(best_path)
        cm         = confusion_matrix(y_test, best_model.predict(X_test))
        fig, ax    = plt.subplots(figsize=(5, 4))
        disp       = ConfusionMatrixDisplay(cm, display_labels=["Real", "Fake/AI"])
        disp.plot(ax=ax, colorbar=False, cmap='Blues')
        ax.set_title(f"Confusion Matrix — {best_name}", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
        plt.close()

    # --- Plot 4: Special char feature distribution ---
    _plot_special_char_analysis(splits["test_df"], output_dir)

    print(f"\n  📊 Plots saved to {output_dir}/")
    return results_df


def _plot_special_char_analysis(test_df, output_dir):
    """Visualize special char feature distributions by class."""
    extractor = SpecialCharFeatureExtractor()
    feats_df  = extractor.transform(test_df["text"].tolist())
    feats_df["label"] = test_df["label"].values

    key_features = [
        "n_emojis", "n_exclaim", "n_emdash", "n_ellipsis",
        "n_repeated_punct", "non_ascii_ratio", "char_entropy",
        "ai_signal_score", "punct_density", "emoji_word_ratio"
    ]

    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    axes = axes.flatten()
    palette = {0: "#2196F3", 1: "#F44336"}
    label_names = {0: "Real/Human", 1: "Fake/AI"}

    for i, feat in enumerate(key_features):
        if feat not in feats_df.columns:
            continue
        for lbl, grp in feats_df.groupby("label"):
            axes[i].hist(grp[feat].clip(upper=grp[feat].quantile(0.95)),
                         bins=30, alpha=0.6, color=palette[lbl],
                         label=label_names[lbl], density=True)
        axes[i].set_title(feat, fontsize=10, fontweight='bold')
        axes[i].legend(fontsize=7)
        axes[i].set_yticks([])

    fig.suptitle("Special Character Feature Distributions: Real vs Fake/AI",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "special_char_distributions.png"), dpi=150)
    plt.close()
    print(f"  📊 Special char distributions → {output_dir}/special_char_distributions.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    splits = load_dataset()
    evaluate_all_models(splits)

    # Demo single prediction
    print("\n" + "=" * 60)
    print("DEMO: Single-text prediction")
    print("=" * 60)
    bundle_path = "models/feature_bundle.pkl"
    model_path  = "models/LogisticRegression.pkl"

    if os.path.exists(bundle_path) and os.path.exists(model_path):
        examples = [
            "omg this is the BEST thing ever!! 😍😍 totally obsessed!!",
            "This product exceeded all my expectations. The craftsmanship is "
            "truly exceptional—a testament to thoughtful engineering.",
        ]
        for ex in examples:
            result = predict_single(ex, model_path, bundle_path)
            print(f"\nText: {ex[:70]}...")
            print(f"  → {result['prediction']}  (confidence={result['confidence']})")
            print(f"     emojis={result['n_emojis']}, exclaim={result['n_exclaim']}, "
                  f"emdash={result['n_emdash']}, ai_signal={result['ai_signal_score']}")
