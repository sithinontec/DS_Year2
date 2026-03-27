"""
evaluate.py
===========
Evaluates models trained on fake_reviews_dataset_2022.csv against:
  1. The held-out test split (CG vs OR)
  2. The AI dataset (fake_reviews_AI.csv) — out-of-distribution test

The AI dataset has NO labels (it's entirely AI-generated).
We treat it as a 100% positive class and measure detection rate.

Also produces:
  - Model comparison bar chart
  - ROC curves
  - Confusion matrix (best model)
  - Feature distribution plots for all three text types
  - Score histogram: how well does each model separate OR / CG / AI?
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.preprocessing import load_dataset, preserve_special_clean
from src.feature_engineering import SpecialCharFeatureExtractor
from src.train_classical import build_feature_matrix


# ── Inference helper ─────────────────────────────────────────────────── #

def predict_single(text: str, model_path: str, bundle_path: str) -> dict:
    """Predict a single review. Returns prediction, confidence, and key features."""
    bundle = joblib.load(bundle_path)
    model  = joblib.load(model_path)
    extractor = SpecialCharFeatureExtractor()

    cleaned = preserve_special_clean(text)
    X, _    = build_feature_matrix([cleaned], transformer_bundle=bundle, fit=False)
    pred    = model.predict(X)[0]
    conf    = model.predict_proba(X)[0][1] if hasattr(model, "predict_proba") else None
    feats   = extractor.extract(cleaned)

    return {
        "text":              text[:100] + ("..." if len(text) > 100 else ""),
        "prediction":        "Fake/Generated" if pred == 1 else "Real/Human",
        "confidence":        round(conf, 4) if conf is not None else None,
        "cg_signal_score":   round(feats["cg_signal_score"], 3),
        "ai_signal_score":   round(feats["ai_signal_score"], 3),
        "is_truncated":      feats["is_truncated"],
        "n_glued_sents":     feats["n_glued_sents"],
        "n_emojis":          feats["n_emojis"],
        "n_words":           feats["n_words"],
        "type_token_ratio":  round(feats["type_token_ratio"], 3),
        "n_contractions":    feats["n_contractions"],
    }


# ── Main evaluation ───────────────────────────────────────────────────── #

def evaluate_all_models(
    splits: dict,
    ai_csv_path: str = None,
    models_dir:  str = "models",
    output_dir:  str = "outputs",
):
    os.makedirs(output_dir, exist_ok=True)

    test_texts = splits["test_df"]["text_clean"].tolist()
    y_test     = splits["test_df"]["label_num"].values
    label_names= splits["label_names"]   # ["OR (real)", "CG (generated)"]

    bundle_path = os.path.join(models_dir, "feature_bundle.pkl")
    if not os.path.exists(bundle_path):
        print("[evaluate] No feature bundle found. Run train_classical.py first.")
        return None

    bundle = joblib.load(bundle_path)

    print("Building test feature matrix...")
    X_test, _ = build_feature_matrix(test_texts, transformer_bundle=bundle, fit=False)

    # ── Load AI dataset ─────────────────────────────────────────────── #
    ai_df     = None
    X_ai      = None
    ai_texts  = []

    ai_candidates = [
        ai_csv_path,
        "data/fake_reviews_AI.csv",
        "/mnt/user-data/uploads/fake_reviews_AI.csv",
    ]
    for c in ai_candidates:
        if c and os.path.exists(c):
            ai_df = pd.read_csv(c)
            print(f"[evaluate] Loaded AI dataset: {len(ai_df):,} rows from {c}")
            ai_df["text_clean"] = ai_df["text"].apply(preserve_special_clean)
            ai_texts = ai_df["text_clean"].tolist()
            print("Building AI feature matrix...")
            X_ai, _ = build_feature_matrix(ai_texts, transformer_bundle=bundle, fit=False)
            break

    if ai_df is None:
        print("[evaluate] AI dataset not found — skipping AI evaluation.")

    # ── Discover saved models ────────────────────────────────────────── #
    skip = {"feature_bundle.pkl", "bilstm_tokenizer.pkl", "bilstm_scaler.pkl", "bert_scaler.pkl"}
    model_files = {
        name.replace(".pkl", ""): os.path.join(models_dir, name)
        for name in sorted(os.listdir(models_dir))
        if name.endswith(".pkl") and name not in skip
    }

    if not model_files:
        print("[evaluate] No .pkl model files found in", models_dir)
        return None

    results = {}
    roc_data = {}

    print(f"\n{'Model':<35} {'Test acc':>9} {'Test AUC':>9}", end="")
    if X_ai is not None:
        print(f"  {'AI detect%':>10}", end="")
    print()
    print("-" * 70)

    for name, path in model_files.items():
        try:
            model = joblib.load(path)
            preds = model.predict(X_test)
            acc   = (preds == y_test).mean()
            auc   = None
            ai_detect = None

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test)[:, 1]
                auc   = roc_auc_score(y_test, proba)
                fpr, tpr, _ = roc_curve(y_test, proba)
                roc_data[name] = (fpr, tpr, auc)

            if X_ai is not None:
                ai_preds  = model.predict(X_ai)
                ai_detect = ai_preds.mean() * 100  # % flagged as generated

            results[name] = {
                "accuracy":   acc,
                "auc":        auc,
                "ai_detect%": ai_detect,
                "precision_cg": None,
                "recall_cg":    None,
                "f1_cg":        None,
            }

            rep = classification_report(y_test, preds, output_dict=True)
            results[name].update({
                "precision_cg": rep.get("1", {}).get("precision"),
                "recall_cg":    rep.get("1", {}).get("recall"),
                "f1_cg":        rep.get("1", {}).get("f1-score"),
            })

            print(f"  {name:<33} {acc:>9.4f} {str(round(auc,4)) if auc else 'N/A':>9}", end="")
            if X_ai is not None:
                print(f"  {ai_detect:>9.1f}%", end="")
            print()

        except Exception as e:
            print(f"  ⚠️  {name}: {e}")

    # ── Best model full report ───────────────────────────────────────── #
    results_df = pd.DataFrame(results).T
    best_name  = results_df["accuracy"].idxmax()
    best_path  = model_files[best_name]
    best_model = joblib.load(best_path)

    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_name}")
    print(f"{'='*60}")
    print(classification_report(
        y_test, best_model.predict(X_test),
        target_names=label_names
    ))

    if X_ai is not None and hasattr(best_model, "predict_proba"):
        ai_proba  = best_model.predict_proba(X_ai)[:, 1]
        ai_preds  = best_model.predict(X_ai)
        print(f"AI dataset detection:  {ai_preds.mean()*100:.1f}% flagged as generated")
        print(f"AI dataset avg confidence: {ai_proba.mean():.4f}")
        print(f"AI dataset confidence ≥ 0.7: {(ai_proba >= 0.7).mean()*100:.1f}%")
        print(f"AI dataset confidence ≥ 0.9: {(ai_proba >= 0.9).mean()*100:.1f}%")

    # ── Save results CSV ─────────────────────────────────────────────── #
    results_df.to_csv(os.path.join(output_dir, "model_comparison.csv"))

    # ── Plots ─────────────────────────────────────────────────────────── #
    _plot_accuracy_bars(results_df, output_dir, has_ai=(X_ai is not None))
    if roc_data:
        _plot_roc(roc_data, output_dir)
    _plot_confusion(best_model, X_test, y_test, best_name, label_names, output_dir)
    _plot_feature_distributions(splits, ai_df, output_dir)

    if X_ai is not None:
        _plot_score_histograms(best_model, X_test, y_test, X_ai, best_name, output_dir)

    print(f"\n✅ All plots saved to {output_dir}/")
    return results_df


# ── Plot helpers ─────────────────────────────────────────────────────── #

def _plot_accuracy_bars(results_df, output_dir, has_ai=False):
    fig, axes = plt.subplots(1, 2 if has_ai else 1,
                              figsize=(14 if has_ai else 8, 5))
    if not has_ai:
        axes = [axes]

    # Test accuracy
    ax = axes[0]
    df = results_df.sort_values("accuracy", ascending=True)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df)))
    bars = ax.barh(df.index, df["accuracy"], color=colors)
    ax.set_xlabel("Accuracy"); ax.set_title("Test Accuracy (CG vs OR)", fontweight="bold")
    ax.set_xlim(0, 1)
    for bar, v in zip(bars, df["accuracy"]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f"{v:.4f}", va="center", fontsize=9)

    if has_ai and "ai_detect%" in results_df.columns:
        ax2 = axes[1]
        df2 = results_df.dropna(subset=["ai_detect%"]).sort_values("ai_detect%", ascending=True)
        bars2 = ax2.barh(df2.index, df2["ai_detect%"], color=plt.cm.plasma(np.linspace(0.3, 0.9, len(df2))))
        ax2.set_xlabel("% flagged as generated")
        ax2.set_title("AI Dataset Detection Rate\n(% of modern AI reviews caught)", fontweight="bold")
        ax2.set_xlim(0, 100)
        for bar, v in zip(bars2, df2["ai_detect%"]):
            ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                     f"{v:.1f}%", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()


def _plot_roc(roc_data, output_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0,1],[0,1], "k--", alpha=0.4, label="Random")
    for i, (name, (fpr, tpr, auc)) in enumerate(roc_data.items()):
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc:.3f})")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — CG Detection (test set)", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=150)
    plt.close()


def _plot_confusion(model, X_test, y_test, name, label_names, output_dir):
    cm   = confusion_matrix(y_test, model.predict(X_test))
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=label_names).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {name}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close()


def _plot_feature_distributions(splits, ai_df, output_dir):
    """Compare OR / CG / AI feature distributions side by side."""
    extractor = SpecialCharFeatureExtractor()

    or_df = splits["full_df"][splits["full_df"]["label"] == "OR"].copy()
    cg_df = splits["full_df"][splits["full_df"]["label"] == "CG"].copy()

    or_feats = extractor.transform(or_df["text_"].astype(str).tolist())
    cg_feats = extractor.transform(cg_df["text_"].astype(str).tolist())

    datasets = [("OR (real)", or_feats, "#2196F3"),
                ("CG (generated)", cg_feats, "#FF9800")]

    if ai_df is not None:
        ai_feats = extractor.transform(ai_df["text"].astype(str).tolist())
        datasets.append(("AI (modern)", ai_feats, "#9C27B0"))

    key_features = [
        "is_truncated", "n_glued_sents", "has_emoji",
        "n_words", "type_token_ratio", "burstiness",
        "n_caps_words", "n_exclaim", "contraction_ratio",
        "bigram_repetition", "cg_signal_score", "ai_signal_score",
    ]

    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    axes = axes.flatten()

    for i, feat in enumerate(key_features):
        ax = axes[i]
        for label, feats, color in datasets:
            if feat in feats.columns:
                data = feats[feat].clip(upper=feats[feat].quantile(0.97))
                ax.hist(data, bins=25, alpha=0.55, color=color, label=label, density=True)
        ax.set_title(feat, fontsize=10, fontweight="bold")
        ax.legend(fontsize=7)
        ax.set_yticks([])

    fig.suptitle("Feature Distributions: OR vs CG vs AI",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_distributions.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Feature distributions → {output_dir}/feature_distributions.png")


def _plot_score_histograms(model, X_test, y_test, X_ai, model_name, output_dir):
    """Show the model's confidence scores for OR, CG, and AI separately."""
    if not hasattr(model, "predict_proba"):
        return

    or_proba  = model.predict_proba(X_test[y_test == 0])[:, 1]
    cg_proba  = model.predict_proba(X_test[y_test == 1])[:, 1]
    ai_proba  = model.predict_proba(X_ai)[:, 1]

    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, 1, 40)
    ax.hist(or_proba, bins=bins, alpha=0.6, color="#2196F3",  label=f"OR/Real   (n={len(or_proba):,})",  density=True)
    ax.hist(cg_proba, bins=bins, alpha=0.6, color="#FF9800",  label=f"CG 2022   (n={len(cg_proba):,})",  density=True)
    ax.hist(ai_proba, bins=bins, alpha=0.6, color="#9C27B0",  label=f"AI modern (n={len(ai_proba):,})",  density=True)

    ax.axvline(0.5, color="red", linestyle="--", alpha=0.7, label="Decision boundary (0.5)")
    ax.set_xlabel("Model confidence score (probability of being generated)", fontsize=11)
    ax.set_ylabel("Density")
    ax.set_title(f"Score Distribution — {model_name}\n"
                 f"CG detection: {(cg_proba>=0.5).mean()*100:.1f}%  |  "
                 f"AI detection: {(ai_proba>=0.5).mean()*100:.1f}%  |  "
                 f"OR false-positive: {(or_proba>=0.5).mean()*100:.1f}%",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "score_distributions.png"), dpi=150)
    plt.close()
    print(f"  📊 Score distributions → {output_dir}/score_distributions.png")


# ── CLI ──────────────────────────────────────────────────────────────── #
if __name__ == "__main__":
    splits = load_dataset()
    evaluate_all_models(
        splits,
        ai_csv_path="/mnt/user-data/uploads/fake_reviews_AI.csv",
    )

    # Single prediction demo
    bundle_path = "models/feature_bundle.pkl"
    model_path  = "models/LogisticRegression.pkl"
    if os.path.exists(bundle_path) and os.path.exists(model_path):
        print("\n" + "="*60)
        print("SINGLE-REVIEW INFERENCE DEMO")
        print("="*60)
        demos = [
            "My dog LOVES these! It's not too tall (I'm 5'7\") -- great for the price!!!",
            "This is a great bag. I love the look and feel of it. I had to get a size down, as I wear a 6",
            "Got this mini fridge for my Silom condo. Does exactly what it needs to do. 🧴❄️",
        ]
        for d in demos:
            r = predict_single(d, model_path, bundle_path)
            print(f"\n→ {r['prediction']}  (conf={r['confidence']})")
            print(f"  emojis={r['n_emojis']}, words={r['n_words']}, trunc={r['is_truncated']}, glued={r['n_glued_sents']}")
            print(f"  CG score={r['cg_signal_score']}  AI score={r['ai_signal_score']}")
            print(f"  Text: {r['text']}")