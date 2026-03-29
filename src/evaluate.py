"""
evaluate.py
===========
Binary classification evaluation: OR (real/human) vs CG (computer-generated).

Both the 2022 CG dataset and the modern AI dataset are treated as the same CG class.
The AI dataset (fake_reviews_AI.csv) is used as an additional CG test set — all
reviews in it are generated, so we expect the model to flag all of them as CG.

Produces:
  - Model accuracy comparison bar chart
  - ROC curves
  - Confusion matrix (best model)
  - Feature distribution plots (OR vs CG 2022 vs AI/CG modern)
  - Score distribution histogram
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.preprocessing import load_dataset, preserve_special_clean
try:
    from src.preprocessing import strip_emoji
except ImportError:
    # Fallback if local preprocessing.py is an older version
    import re as _re
    def strip_emoji(text: str) -> str:
        return _re.sub(
            r'[\U0001F300-\U0001F5FF\U0001F600-\U0001F64F'
            r'\U0001F680-\U0001F6FF\U0001F900-\U0001FAFF\u2702-\u27B0]+',
            '', str(text), flags=_re.UNICODE).strip()
from src.feature_engineering import SpecialCharFeatureExtractor
from src.train_classical import build_feature_matrix


# ── Inference ────────────────────────────────────────────────────────── #

def predict_single(text: str, model_path: str, bundle_path: str) -> dict:
    """
    Predict whether a single review is OR (real) or CG (generated).

    Parameters
    ----------
    text        : raw review text (emoji already stripped by caller if needed)
    model_path  : path to a saved .pkl model
    bundle_path : path to the feature_bundle.pkl from train_classical

    Returns
    -------
    dict with prediction, confidence, cg_signal_score, and key feature values
    """
    bundle    = joblib.load(bundle_path)
    model     = joblib.load(model_path)
    extractor = SpecialCharFeatureExtractor()

    cleaned = preserve_special_clean(text)
    X, _    = build_feature_matrix([cleaned], transformer_bundle=bundle, fit=False)
    pred    = model.predict(X)[0]
    conf    = model.predict_proba(X)[0][1] if hasattr(model, "predict_proba") else None
    feats   = extractor.extract(cleaned)

    return {
        "text":             text[:100] + ("..." if len(text) > 100 else ""),
        "prediction":       "CG (generated)" if pred == 1 else "OR (real)",
        "confidence":       round(conf, 4) if conf is not None else None,
        "cg_signal_score":  round(feats["cg_signal_score"], 3),
        "is_truncated":     feats["is_truncated"],
        "n_glued_sents":    feats["n_glued_sents"],
        "n_words":          feats["n_words"],
        "type_token_ratio": round(feats["type_token_ratio"], 3),
        "n_contractions":   feats["n_contractions"],
        "n_caps_words":     feats["n_caps_words"],
        "burstiness":       round(feats["burstiness"], 3),
    }


# ── Main evaluation ───────────────────────────────────────────────────── #

def evaluate_all_models(
    splits: dict,
    ai_csv_path: str = None,
    models_dir:  str = "models",
    output_dir:  str = "outputs",
):
    """
    Evaluate all saved models on the held-out 2022 test split (OR vs CG).
    If an AI CSV is provided, also report how many of those reviews are
    correctly flagged as CG — since all are generated, higher = better.
    """
    os.makedirs(output_dir, exist_ok=True)

    test_texts  = splits["test_df"]["text_clean"].tolist()
    y_test      = splits["test_df"]["label_num"].values
    label_names = splits["label_names"]    # ["OR (real)", "CG (generated)"]

    bundle_path = os.path.join(models_dir, "feature_bundle.pkl")
    if not os.path.exists(bundle_path):
        print("[evaluate] No feature bundle found. Run train_classical.py first.")
        return None

    bundle = joblib.load(bundle_path)
    print("Building test feature matrix...")
    X_test, _ = build_feature_matrix(test_texts, transformer_bundle=bundle, fit=False)

    # ── Load AI/CG dataset ───────────────────────────────────────────── #
    ai_df    = None
    X_ai     = None
    ai_texts = []

    for c in [ai_csv_path, "data/fake_reviews_AI.csv",
               "/mnt/user-data/uploads/fake_reviews_AI.csv"]:
        if c and os.path.exists(c):
            ai_df = pd.read_csv(c)
            print(f"[evaluate] Loaded AI/CG dataset: {len(ai_df):,} rows from {c}")
            # Strip emoji — prevents shortcut learning from dataset imbalance
            ai_df["text_clean"] = ai_df["text"].apply(
                lambda t: preserve_special_clean(strip_emoji(str(t)))
            )
            print("Building AI/CG feature matrix (emoji stripped)...")
            X_ai, _ = build_feature_matrix(
                ai_df["text_clean"].tolist(), transformer_bundle=bundle, fit=False
            )
            break

    if ai_df is None:
        print("[evaluate] AI/CG dataset not found — skipping.")

    # ── Load only the two expected models by explicit name ──────────── #
    # Avoids accidentally picking up stale .pkl files from old training runs
    # (scalers, vectorizers, extractors, or models from removed experiments).
    model_files = {}
    for name in ["LogisticRegression", "RandomForest"]:
        path = os.path.join(models_dir, f"{name}.pkl")
        if os.path.exists(path):
            model_files[name] = path
        else:
            print(f"  ⚠️  {name}.pkl not found in {models_dir} — run train_models() first")
    if not model_files:
        print("[evaluate] No .pkl model files found in", models_dir)
        return None

    results  = {}
    roc_data = {}

    header = f"{'Model':<35} {'Test acc':>9} {'Test AUC':>9}"
    if X_ai is not None:
        header += f"  {'AI/CG catch%':>12}"
    print("\n" + header)
    print("-" * (len(header) + 2))

    for name, path in model_files.items():
        try:
            model = joblib.load(path)

            # RandomForest was trained on 39 special-char features (not TF-IDF)
            # so we rebuild its feature matrix using the saved scaler + extractor.
            if name == "RandomForest":
                sc_path  = os.path.join(models_dir, 'RandomForest_scaler.pkl')
                ext_path = os.path.join(models_dir, 'RandomForest_extractor.pkl')
                if not os.path.exists(sc_path):
                    print(f"  ⚠️  {name}: scaler not found — retrain to fix")
                    continue
                _sc  = joblib.load(sc_path)
                _ext = joblib.load(ext_path)
                X_model_test = _sc.transform(_ext.transform(test_texts).values)
                X_model_ai   = _sc.transform(_ext.transform(ai_texts).values) if ai_texts else None
            else:
                X_model_test = X_test
                X_model_ai   = X_ai

            preds    = model.predict(X_model_test)
            acc      = (preds == y_test).mean()
            auc      = None
            ai_catch = None

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_model_test)[:, 1]
                auc   = roc_auc_score(y_test, proba)
                fpr, tpr, _ = roc_curve(y_test, proba)
                roc_data[name] = (fpr, tpr, auc)

            if X_model_ai is not None:
                ai_catch = model.predict(X_model_ai).mean() * 100

            rep = classification_report(y_test, preds, output_dict=True)
            results[name] = {
                "accuracy":     acc,
                "auc":          auc,
                "ai_cg_catch%": ai_catch,
                "precision_cg": rep.get("1", {}).get("precision"),
                "recall_cg":    rep.get("1", {}).get("recall"),
                "f1_cg":        rep.get("1", {}).get("f1-score"),
            }

            row = f"  {name:<33} {acc:>9.4f} {str(round(auc, 4)) if auc else 'N/A':>9}"
            if X_ai is not None:
                row += f"  {ai_catch:>11.1f}%"
            print(row)

        except Exception as e:
            print(f"  ⚠️  {name}: {e}")

    # ── Best model full report ───────────────────────────────────────── #
    results_df = pd.DataFrame(results).T
    best_name  = results_df["accuracy"].idxmax()
    best_model = joblib.load(model_files[best_name])

    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_name}")
    print(f"{'='*60}")
    print(classification_report(y_test, best_model.predict(X_test),
                                 target_names=label_names))

    if X_ai is not None and hasattr(best_model, "predict_proba"):
        ai_proba = best_model.predict_proba(X_ai)[:, 1]
        print(f"AI/CG dataset — flagged as CG:")
        print(f"  ≥ 0.5 confidence:  {(ai_proba >= 0.5).mean()*100:.1f}%")
        print(f"  ≥ 0.7 confidence:  {(ai_proba >= 0.7).mean()*100:.1f}%")
        print(f"  ≥ 0.9 confidence:  {(ai_proba >= 0.9).mean()*100:.1f}%")

    # ── Save & plot ──────────────────────────────────────────────────── #
    results_df.to_csv(os.path.join(output_dir, "model_comparison.csv"))
    _plot_accuracy_bars(results_df, output_dir, has_ai=(X_ai is not None))
    if roc_data:
        _plot_roc(roc_data, output_dir)
    _plot_confusion(best_model, X_test, y_test, best_name, label_names, output_dir)
    _plot_feature_distributions(splits, ai_df, output_dir)
    if X_ai is not None:
        _plot_score_histograms(best_model, X_test, y_test, X_ai, best_name, output_dir)

    print(f"\n✅ All outputs saved to {output_dir}/")
    return results_df


# ── Plot helpers ─────────────────────────────────────────────────────── #

def _plot_accuracy_bars(results_df, output_dir, has_ai=False):
    ncols  = 2 if has_ai else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    df     = results_df.sort_values("accuracy", ascending=True)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df)))
    bars   = axes[0].barh(df.index, df["accuracy"], color=colors)
    axes[0].set_xlabel("Accuracy")
    axes[0].set_title("Test Accuracy — OR vs CG", fontweight="bold")
    axes[0].set_xlim(0, 1)
    for bar, v in zip(bars, df["accuracy"]):
        axes[0].text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                     f"{v:.4f}", va="center", fontsize=9)

    if has_ai and "ai_cg_catch%" in results_df.columns:
        df2   = results_df.dropna(subset=["ai_cg_catch%"]).sort_values("ai_cg_catch%", ascending=True)
        bars2 = axes[1].barh(df2.index, df2["ai_cg_catch%"],
                              color=plt.cm.plasma(np.linspace(0.3, 0.9, len(df2))))
        axes[1].set_xlabel("% flagged as CG")
        axes[1].set_title("AI/CG Dataset Catch Rate\n(all are generated — higher is better)",
                          fontweight="bold")
        axes[1].set_xlim(0, 100)
        for bar, v in zip(bars2, df2["ai_cg_catch%"]):
            axes[1].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                         f"{v:.1f}%", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()


def _plot_roc(roc_data, output_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    for name, (fpr, tpr, auc) in roc_data.items():
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc:.3f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — OR vs CG", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=150)
    plt.close()


def _plot_confusion(model, X_test, y_test, name, label_names, output_dir):
    cm  = confusion_matrix(y_test, model.predict(X_test))
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=label_names).plot(
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title(f"Confusion Matrix — {name}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close()


def _plot_feature_distributions(splits, ai_df, output_dir):
    """Feature distributions: OR vs CG 2022 vs AI/CG modern (emoji stripped)."""
    extractor = SpecialCharFeatureExtractor()

    or_feats = extractor.transform(
        splits["full_df"][splits["full_df"]["label"] == "OR"]["text_"].astype(str).tolist()
    )
    cg_feats = extractor.transform(
        splits["full_df"][splits["full_df"]["label"] == "CG"]["text_"].astype(str).tolist()
    )

    datasets = [
        ("OR (real)",    or_feats, "#2196F3"),
        ("CG 2022",      cg_feats, "#FF9800"),
    ]
    if ai_df is not None:
        ai_feats = extractor.transform(ai_df["text_clean"].astype(str).tolist())
        datasets.append(("AI / CG modern", ai_feats, "#E53935"))

    key_features = [
        "is_truncated", "n_glued_sents", "n_words",
        "type_token_ratio", "burstiness", "n_caps_words",
        "n_exclaim", "contraction_ratio", "bigram_repetition", "cg_signal_score",
    ]

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
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

    fig.suptitle("Feature Distributions — OR (real) vs CG (all types)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_distributions.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


def _plot_score_histograms(model, X_test, y_test, X_ai, model_name, output_dir):
    """P(CG) score distributions for OR, CG 2022, and AI/CG modern."""
    if not hasattr(model, "predict_proba"):
        return

    or_proba = model.predict_proba(X_test[y_test == 0])[:, 1]
    cg_proba = model.predict_proba(X_test[y_test == 1])[:, 1]
    ai_proba = model.predict_proba(X_ai)[:, 1]

    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, 1, 40)
    ax.hist(or_proba, bins=bins, alpha=0.6, color="#2196F3",
            label=f"OR / real  (n={len(or_proba):,})", density=True)
    ax.hist(cg_proba, bins=bins, alpha=0.6, color="#FF9800",
            label=f"CG 2022   (n={len(cg_proba):,})", density=True)
    ax.hist(ai_proba, bins=bins, alpha=0.6, color="#E53935",
            label=f"AI / CG modern (n={len(ai_proba):,})", density=True)
    ax.axvline(0.5, color="black", linestyle="--", lw=1.5, label="Threshold 0.5")
    ax.set_xlabel("P(CG / generated)", fontsize=11)
    ax.set_ylabel("Density")
    ax.set_title(
        f"Score Distribution — {model_name}\n"
        f"CG 2022 caught: {(cg_proba>=0.5).mean()*100:.1f}%  |  "
        f"AI/CG caught: {(ai_proba>=0.5).mean()*100:.1f}%  |  "
        f"OR false-positive: {(or_proba>=0.5).mean()*100:.1f}%",
        fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "score_distributions.png"), dpi=150)
    plt.close()


# ── CLI ──────────────────────────────────────────────────────────────── #
if __name__ == "__main__":
    from src.preprocessing import strip_emoji

    splits = load_dataset()
    evaluate_all_models(
        splits,
        ai_csv_path="/mnt/user-data/uploads/fake_reviews_AI.csv",
    )

    bundle_path = "models/feature_bundle.pkl"
    model_path  = "models/LogisticRegression_augmented.pkl"
    if os.path.exists(bundle_path) and os.path.exists(model_path):
        print("\n" + "=" * 60)
        print("SINGLE-REVIEW INFERENCE DEMO")
        print("=" * 60)
        demos = [
            "My dog LOVES these! It's not too tall -- great for the price!!!",
            "This is a great bag. I love the look and feel of it. I had to get a size down, as I wear a 6",
            strip_emoji("Got this mini fridge for my Silom condo. Does exactly what it needs to do. 🧴❄️"),
        ]
        for d in demos:
            r = predict_single(d, model_path, bundle_path)
            print(f"\n→ {r['prediction']}  (conf={r['confidence']})")
            print(f"  words={r['n_words']}, trunc={r['is_truncated']}, "
                  f"glued={r['n_glued_sents']}, cg_score={r['cg_signal_score']}")
            print(f"  Text: {r['text']}")