"""
train_deep.py
=============
Deep Learning pipeline:

1. Bi-LSTM
   - Custom tokenizer that keeps special chars/emoji as tokens
   - Special char features concatenated to LSTM final state
   - Good baseline, fast to train

2. BERT fine-tuning (bert-base-uncased)
   - Uses HuggingFace transformers
   - Special char features fed into a secondary head
   - Final classification: BERT_CLS + SpecialCharFeatures → Linear → Sigmoid

Both models preserve and use special character signals.
"""

import os
import sys
import numpy as np
import joblib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.preprocessing import load_dataset
from src.feature_engineering import SpecialCharFeatureExtractor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[train_deep] Using device: {DEVICE}")


# ---------------------------------------------------------------------------
# Bi-LSTM
# ---------------------------------------------------------------------------

class CharPreservingTokenizer:
    """
    Tokenizer that keeps special chars and emoji as individual tokens.
    Builds vocabulary from training data.
    """
    import re
    # Token pattern: word | emoji block | punctuation cluster | single non-space char
    TOKEN_RE = re.compile(
        r'[\U0001F300-\U0001FAFF]+|[!?\.]{1,3}|[—–\-]{1,2}|\w+|[^\s\w]',
        flags=re.UNICODE
    )

    def __init__(self, max_vocab: int = 20000, max_len: int = 200):
        self.max_vocab = max_vocab
        self.max_len   = max_len
        self.word2idx  = {"<PAD>": 0, "<UNK>": 1}
        self.vocab_built = False

    def tokenize(self, text: str):
        return self.TOKEN_RE.findall(text.lower())

    def build_vocab(self, texts):
        from collections import Counter
        counter = Counter()
        for t in texts:
            counter.update(self.tokenize(t))
        for word, _ in counter.most_common(self.max_vocab - 2):
            if word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)
        self.vocab_built = True
        print(f"  [Tokenizer] Vocab size: {len(self.word2idx)}")

    def encode(self, text: str):
        tokens = self.tokenize(text)[:self.max_len]
        ids    = [self.word2idx.get(t, 1) for t in tokens]  # 1 = UNK
        # Pad
        ids   += [0] * (self.max_len - len(ids))
        return ids

    @property
    def vocab_size(self):
        return len(self.word2idx)


class ReviewDatasetLSTM(Dataset):
    def __init__(self, texts, special_feats, labels, tokenizer):
        self.ids      = [torch.tensor(tokenizer.encode(t), dtype=torch.long) for t in texts]
        self.special  = torch.tensor(special_feats, dtype=torch.float32)
        self.labels   = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.ids[idx], self.special[idx], self.labels[idx]


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_special_feats,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.dropout     = nn.Dropout(dropout)
        self.attn        = nn.Linear(hidden_dim * 2, 1)  # attention over timesteps
        self.special_fc  = nn.Sequential(
            nn.Linear(n_special_feats, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        combined_dim = hidden_dim * 2 + 64
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, token_ids, special_feats):
        x = self.dropout(self.embedding(token_ids))          # (B, T, E)
        lstm_out, _ = self.lstm(x)                           # (B, T, 2H)
        # Attention pooling
        attn_w = torch.softmax(self.attn(lstm_out), dim=1)   # (B, T, 1)
        context = (attn_w * lstm_out).sum(dim=1)             # (B, 2H)
        special_out = self.special_fc(special_feats)         # (B, 64)
        combined    = torch.cat([context, special_out], dim=1)
        return self.classifier(combined)


def train_bilstm(splits, save_dir="models",
                 embed_dim=128, hidden_dim=256,
                 batch_size=64, epochs=10, lr=1e-3):
    os.makedirs(save_dir, exist_ok=True)

    extractor = SpecialCharFeatureExtractor()
    scaler    = MinMaxScaler()

    train_texts = splits["train_df"]["text_clean"].tolist()
    val_texts   = splits["val_df"]["text_clean"].tolist()
    test_texts  = splits["test_df"]["text_clean"].tolist()

    y_train = splits["train_df"]["label"].values
    y_val   = splits["val_df"]["label"].values
    y_test  = splits["test_df"]["label"].values

    # Special char features
    X_sc_train = scaler.fit_transform(extractor.transform(train_texts).values)
    X_sc_val   = scaler.transform(extractor.transform(val_texts).values)
    X_sc_test  = scaler.transform(extractor.transform(test_texts).values)
    n_special  = X_sc_train.shape[1]

    # Tokenizer
    tokenizer = CharPreservingTokenizer(max_vocab=20000, max_len=200)
    tokenizer.build_vocab(train_texts)

    # Datasets
    train_ds = ReviewDatasetLSTM(train_texts, X_sc_train, y_train, tokenizer)
    val_ds   = ReviewDatasetLSTM(val_texts,   X_sc_val,   y_val,   tokenizer)
    test_ds  = ReviewDatasetLSTM(test_texts,  X_sc_test,  y_test,  tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    model = BiLSTMClassifier(
        vocab_size=tokenizer.vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        n_special_feats=n_special,
    ).to(DEVICE)

    optimizer  = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler  = OneCycleLR(optimizer, max_lr=lr,
                            steps_per_epoch=len(train_loader), epochs=epochs)
    criterion  = nn.CrossEntropyLoss()

    print("\n" + "=" * 60)
    print(f"TRAINING: Bi-LSTM  (device={DEVICE})")
    print("=" * 60)

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for ids, special, labels in train_loader:
            ids, special, labels = ids.to(DEVICE), special.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(ids, special)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * len(labels)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += len(labels)

        train_acc  = correct / total
        val_acc    = _eval_loader(model, val_loader)
        print(f"  Epoch {epoch:02d}/{epochs}  "
              f"loss={total_loss/total:.4f}  "
              f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "bilstm_best.pt"))

    # Load best and evaluate on test
    model.load_state_dict(torch.load(os.path.join(save_dir, "bilstm_best.pt"),
                                     map_location=DEVICE))
    test_acc, test_report, test_auc = _full_eval(model, test_loader, y_test)
    print(f"\n  Test Accuracy: {test_acc:.4f}  ROC-AUC: {test_auc:.4f}")
    print("  " + test_report)

    # Save supporting artifacts
    joblib.dump(tokenizer, os.path.join(save_dir, "bilstm_tokenizer.pkl"))
    joblib.dump(scaler,    os.path.join(save_dir, "bilstm_scaler.pkl"))
    print(f"  💾 Saved → {save_dir}/bilstm_best.pt")

    return {"test_acc": test_acc, "test_auc": test_auc}


def _eval_loader(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for ids, special, labels in loader:
            ids, special, labels = ids.to(DEVICE), special.to(DEVICE), labels.to(DEVICE)
            preds   = model(ids, special).argmax(1)
            correct += (preds == labels).sum().item()
            total   += len(labels)
    return correct / total


def _full_eval(model, loader, y_true):
    model.eval()
    all_preds, all_proba = [], []
    with torch.no_grad():
        for ids, special, _ in loader:
            ids, special = ids.to(DEVICE), special.to(DEVICE)
            logits = model(ids, special)
            proba  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds  = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_proba.extend(proba)
    acc     = (np.array(all_preds) == y_true).mean()
    report  = classification_report(y_true, all_preds, target_names=["Real", "Fake/AI"])
    auc     = roc_auc_score(y_true, all_proba)
    return acc, report, auc


# ---------------------------------------------------------------------------
# BERT fine-tuning
# ---------------------------------------------------------------------------

class ReviewDatasetBERT(Dataset):
    def __init__(self, texts, special_feats, labels, tokenizer, max_len=256):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_len,
            return_tensors='pt',
        )
        self.special = torch.tensor(special_feats, dtype=torch.float32)
        self.labels  = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids':      self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'special_feats':  self.special[idx],
            'labels':         self.labels[idx],
        }


class BERTWithSpecialChar(nn.Module):
    """
    BERT base + a side branch for special char features.
    Final logits = concat(BERT_CLS, SpecialCharMLP) → Linear(2)
    """
    def __init__(self, bert_model, n_special_feats, dropout=0.3):
        super().__init__()
        self.bert    = bert_model
        hidden_size  = bert_model.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.special_fc = nn.Sequential(
            nn.Linear(n_special_feats, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_size + 64, 2)

    def forward(self, input_ids, attention_mask, special_feats):
        out      = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb  = self.dropout(out.last_hidden_state[:, 0, :])   # CLS token
        sp_out   = self.special_fc(special_feats)
        combined = torch.cat([cls_emb, sp_out], dim=1)
        return self.classifier(combined)


def train_bert(splits, save_dir="models",
               model_name="bert-base-uncased",
               batch_size=16, epochs=3, lr=2e-5):
    try:
        from transformers import BertTokenizerFast, BertModel
    except ImportError:
        print("[train_bert] transformers not installed. Skipping BERT.")
        return None

    os.makedirs(save_dir, exist_ok=True)

    extractor = SpecialCharFeatureExtractor()
    scaler    = MinMaxScaler()

    train_texts = splits["train_df"]["text_clean"].tolist()
    val_texts   = splits["val_df"]["text_clean"].tolist()
    test_texts  = splits["test_df"]["text_clean"].tolist()

    y_train = splits["train_df"]["label"].values
    y_val   = splits["val_df"]["label"].values
    y_test  = splits["test_df"]["label"].values

    X_sc_train = scaler.fit_transform(extractor.transform(train_texts).values)
    X_sc_val   = scaler.transform(extractor.transform(val_texts).values)
    X_sc_test  = scaler.transform(extractor.transform(test_texts).values)
    n_special  = X_sc_train.shape[1]

    print(f"\n[train_bert] Loading {model_name} tokenizer & model...")
    bert_tokenizer = BertTokenizerFast.from_pretrained(model_name)
    bert_base      = BertModel.from_pretrained(model_name)

    train_ds = ReviewDatasetBERT(train_texts, X_sc_train, y_train, bert_tokenizer)
    val_ds   = ReviewDatasetBERT(val_texts,   X_sc_val,   y_val,   bert_tokenizer)
    test_ds  = ReviewDatasetBERT(test_texts,  X_sc_test,  y_test,  bert_tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    model     = BERTWithSpecialChar(bert_base, n_special).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    print("\n" + "=" * 60)
    print(f"TRAINING: BERT + SpecialChar  (device={DEVICE})")
    print("=" * 60)

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for batch in train_loader:
            input_ids     = batch['input_ids'].to(DEVICE)
            attention_mask= batch['attention_mask'].to(DEVICE)
            special_feats = batch['special_feats'].to(DEVICE)
            labels        = batch['labels'].to(DEVICE)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, special_feats)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * len(labels)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += len(labels)

        train_acc = correct / total
        val_acc   = _eval_loader_bert(model, val_loader)
        print(f"  Epoch {epoch}/{epochs}  loss={total_loss/total:.4f}  "
              f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "bert_best.pt"))

    model.load_state_dict(torch.load(os.path.join(save_dir, "bert_best.pt"),
                                     map_location=DEVICE))
    test_acc, test_report, test_auc = _full_eval_bert(model, test_loader, y_test)
    print(f"\n  Test Accuracy: {test_acc:.4f}  ROC-AUC: {test_auc:.4f}")
    print("  " + test_report)

    joblib.dump(scaler, os.path.join(save_dir, "bert_scaler.pkl"))
    print(f"  💾 Saved → {save_dir}/bert_best.pt")
    return {"test_acc": test_acc, "test_auc": test_auc}


def _eval_loader_bert(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            logits  = model(batch['input_ids'].to(DEVICE),
                            batch['attention_mask'].to(DEVICE),
                            batch['special_feats'].to(DEVICE))
            preds   = logits.argmax(1).cpu()
            correct += (preds == batch['labels']).sum().item()
            total   += len(batch['labels'])
    return correct / total


def _full_eval_bert(model, loader, y_true):
    model.eval()
    all_preds, all_proba = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch['input_ids'].to(DEVICE),
                           batch['attention_mask'].to(DEVICE),
                           batch['special_feats'].to(DEVICE))
            proba  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds  = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_proba.extend(proba)
    acc    = (np.array(all_preds) == y_true).mean()
    report = classification_report(y_true, all_preds, target_names=["Real", "Fake/AI"])
    auc    = roc_auc_score(y_true, all_proba)
    return acc, report, auc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    splits = load_dataset()
    train_bilstm(splits, epochs=5)
    train_bert(splits, epochs=3)
