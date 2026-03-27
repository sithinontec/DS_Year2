# 🕵️ Fake & AI-Generated Review Detector

A full machine learning pipeline for detecting **fake reviews** and **AI-generated reviews**, with a special focus on preserving and leveraging **special characters, punctuation patterns, and emoji signals** as features.

---

## 📁 Project Structure

```
fake_review_detector/
├── data/                        # Place your dataset CSV here
│   └── reviews.csv              # Expected columns: text, label (0=real, 1=fake/AI)
├── src/
│   ├── feature_engineering.py   # Special char + emoji feature extractor
│   ├── preprocessing.py         # Text cleaning (preserves special chars)
│   ├── train_classical.py       # TF-IDF + Logistic Regression / SVM
│   ├── train_ensemble.py        # XGBoost + Random Forest
│   ├── train_deep.py            # LSTM + BERT fine-tuning
│   └── evaluate.py              # Unified evaluation & reporting
├── models/                      # Saved model artifacts
├── outputs/                     # Reports, confusion matrices, feature plots
├── notebook.ipynb               # End-to-end walkthrough notebook
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt

# 1. Extract features
python src/feature_engineering.py

# 2. Train classical models
python src/train_classical.py

# 3. Train ensemble models
python src/train_ensemble.py

# 4. Train deep learning models
python src/train_deep.py

# 5. Evaluate all
python src/evaluate.py
```

Or run the full notebook: `jupyter notebook notebook.ipynb`

---

## 🔑 Key Features

### Special Character & Emoji Signals (preserved, not stripped!)
| Feature | Why it matters for AI detection |
|---|---|
| `!` frequency | AI often uses moderate punctuation; humans overuse `!!!` |
| `?` frequency | AI rarely ends with uncertain questions |
| `...` ellipsis count | AI loves ellipses for stylistic flow |
| `—` em-dash count | Very common in LLM output |
| Emoji count & diversity | Humans use emojis contextually; AI rarely or uniformly |
| Emoji-to-word ratio | Low ratio = AI signal |
| ALL_CAPS word count | Humans shout; AI stays calm |
| Repeated punctuation | `!!`, `???` almost never from AI |
| Special char entropy | AI punctuation is uniform; humans are chaotic |
| Non-ASCII char ratio | AI sticks to ASCII; humans include ñ, é, curly quotes etc. |

---

## 📊 Models Included

- **Classical:** Logistic Regression, SVM (linear + RBF), Naive Bayes
- **Ensemble:** Random Forest, XGBoost, Gradient Boosting
- **Deep Learning:** Bi-LSTM, BERT (`bert-base-uncased` fine-tuned)

---

## 📋 Dataset Format

```csv
text,label
"This product is amazing!!!",0
"The product exceeded all my expectations in every way.",1
```

Labels: `0` = Real/Human review, `1` = Fake or AI-generated review

A sample synthetic dataset is auto-generated if no CSV is found.
# DS_Year2
