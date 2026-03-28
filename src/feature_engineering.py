"""
feature_engineering.py
======================
Extracts features that distinguish OR (original/human) from CG (computer-generated).
Both the 2022 CG dataset and the modern AI dataset are the same CG class.

CG signals:
  - Truncation mid-sentence        (64.8% CG 2022 / 18% AI)
  - Glued sentences "love it.This" (12.4% CG 2022 only)
  - Low vocab richness / TTR       (CG 2022)
  - Uniform sentence lengths       (low burstiness)
  - Bigram/trigram repetition      (CG 2022)
  - Fewer ALL CAPS, fewer !        (both CG types)
  - Low contraction ratio          (AI/modern CG)
  - Short review length            (AI/modern CG ~30 words)

Total features: 41
"""

import re
import math
import unicodedata
import numpy as np
import pandas as pd
from collections import Counter


class SpecialCharFeatureExtractor:

    # ── Patterns ──────────────────────────────────────────────────────── #
    _WORD_RE           = re.compile(r"\b\w+\b")
    _SENT_SPLIT_RE     = re.compile(r"[.!?]+")
    _EXCLAIM_RE        = re.compile(r"!")
    _QUESTION_RE       = re.compile(r"\?")
    _REPEATED_PUNCT    = re.compile(r"([!?.,'\"\-])\1{1,}")
    _ALL_CAPS_RE       = re.compile(r"\b[A-Z]{2,}\b")
    _NON_ASCII_RE      = re.compile(r"[^\x00-\x7F]")
    _URL_RE            = re.compile(r"https?://\S+|www\.\S+")
    _ELLIPSIS_RE       = re.compile(r"\.{2,}|…")
    _TERMINAL_PUNCT    = re.compile(r"[.!?\"')\]]+\s*$")
    _GLUED_SENT_RE     = re.compile(r"[a-z][.!?][A-Z]")   # CG 2022 artifact

    _CONTRACTION_RE    = re.compile(
        r"\b(i'm|you're|he's|she's|it's|we're|they're|i've|you've|we've|they've|"
        r"i'd|you'd|he'd|she'd|we'd|they'd|i'll|you'll|he'll|she'll|we'll|they'll|"
        r"isn't|aren't|wasn't|weren't|don't|doesn't|didn't|won't|wouldn't|can't|"
        r"couldn't|shouldn't|haven't|hadn't|hasn't|that's|there's|here's|what's|"
        r"who's|how's|let's|tbh|omg|imo|lol|btw|ngl|idk)\b",
        re.IGNORECASE,
    )
    _FORMAL_RE         = re.compile(
        r"\b(furthermore|moreover|consequently|therefore|thus|hence|additionally|"
        r"subsequently|commendable|noteworthy|exceptional|remarkable|outstanding|"
        r"superb|testament|craftsmanship|seamlessly|meticulously|delve|utilize|"
        r"robust|well-written|well-developed|well-told|well-crafted|well-done|"
        r"well-made|well-built|well-designed|well-thought)\b",
        re.IGNORECASE,
    )
    _GENERIC_PRAISE_RE = re.compile(
        r"\b(great product|great read|great book|great item|great purchase|"
        r"highly recommend|very happy|very pleased|very satisfied|works great|"
        r"works well|works perfectly|works as expected|as described|as advertised|"
        r"love it|love this|would recommend|would buy again)\b",
        re.IGNORECASE,
    )

    # ── Helpers ───────────────────────────────────────────────────────── #
    def _count(self, pattern, text): return len(pattern.findall(text))
    def _safe_ratio(self, n, d): return n / d if d > 0 else 0.0

    def _char_entropy(self, text):
        if not text: return 0.0
        freq = Counter(text); total = len(text)
        return -sum((c / total) * math.log2(c / total) for c in freq.values())

    def _punct_entropy(self, text):
        punct = [c for c in text if unicodedata.category(c).startswith("P")]
        if not punct: return 0.0
        freq = Counter(punct); total = len(punct)
        return -sum((c / total) * math.log2(c / total) for c in freq.values())

    def _bigram_rep(self, words):
        bg = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
        if not bg: return 0.0
        freq = Counter(b.lower() for b in bg)
        return sum(c - 1 for c in freq.values() if c > 1) / len(bg)

    def _trigram_rep(self, words):
        tg = [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words) - 2)]
        if not tg: return 0.0
        freq = Counter(t.lower() for t in tg)
        return sum(c - 1 for c in freq.values() if c > 1) / len(tg)

    # ── Main ──────────────────────────────────────────────────────────── #
    def extract(self, text: str) -> dict:
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        t = text.strip()
        n_chars = max(len(t), 1)

        words   = self._WORD_RE.findall(t)
        n_words = max(len(words), 1)
        sents   = [s.strip() for s in self._SENT_SPLIT_RE.split(t) if s.strip()]
        n_sents = max(len(sents), 1)
        slens   = [len(s.split()) for s in sents]

        # ── Punctuation ───────────────────────────────────────────────── #
        n_exclaim  = self._count(self._EXCLAIM_RE, t)
        n_question = self._count(self._QUESTION_RE, t)
        n_ellipsis = self._count(self._ELLIPSIS_RE, t)
        n_repeat_p = self._count(self._REPEATED_PUNCT, t)
        n_caps     = self._count(self._ALL_CAPS_RE, t)
        caps_ratio = sum(1 for c in t if c.isupper()) / n_chars
        n_non_ascii= len(self._NON_ASCII_RE.findall(t))

        # ── CG truncation artifact ────────────────────────────────────── #
        is_truncated = int(not bool(self._TERMINAL_PUNCT.search(t)) and len(t) > 0)
        last_term    = max((i for i, c in enumerate(t) if c in ".!?"), default=0)
        chars_tail   = max(n_chars - last_term - 1, 0)

        # ── CG glued sentences artifact ───────────────────────────────── #
        n_glued     = self._count(self._GLUED_SENT_RE, t)
        glued_ratio = self._safe_ratio(n_glued, n_sents)

        # ── Sentence structure / burstiness ───────────────────────────── #
        slen_mean  = float(np.mean(slens)) if slens else 0.0
        slen_std   = float(np.std(slens))  if len(slens) > 1 else 0.0
        slen_range = float(max(slens) - min(slens)) if slens else 0.0
        burstiness = self._safe_ratio(slen_std, slen_mean + 1e-6)

        # ── Vocabulary richness ───────────────────────────────────────── #
        wl     = [w.lower() for w in words]
        wfreq  = Counter(wl)
        ttr    = self._safe_ratio(len(set(wl)), n_words)
        hapax_r= self._safe_ratio(sum(1 for c in wfreq.values() if c == 1), n_words)
        wlens  = [len(w) for w in words]
        wlen_mean = float(np.mean(wlens)) if wlens else 0.0
        wlen_std  = float(np.std(wlens))  if len(wlens) > 1 else 0.0

        # ── Repetition ───────────────────────────────────────────────── #
        bigram_rep  = self._bigram_rep(words)
        trigram_rep = self._trigram_rep(words)
        n_generic   = self._count(self._GENERIC_PRAISE_RE, t)
        generic_r   = self._safe_ratio(n_generic, n_words)

        # ── Human vs CG vocabulary ────────────────────────────────────── #
        n_contr  = self._count(self._CONTRACTION_RE, t)
        contr_r  = self._safe_ratio(n_contr, n_words)
        n_formal = self._count(self._FORMAL_RE, t)
        formal_r = self._safe_ratio(n_formal, n_words)

        # ── Structural ───────────────────────────────────────────────── #
        punct_d  = self._safe_ratio(n_exclaim + n_question + n_repeat_p, n_chars)
        char_ent = self._char_entropy(t)
        punct_ent= self._punct_entropy(t)

        # ── Composite CG score (higher = more likely CG) ─────────────── #
        cg_score = (
            is_truncated                                * 0.20 +
            (1.0 if n_glued > 0        else 0.0)       * 0.20 +
            (1.0 if burstiness < 0.25  else 0.0)        * 0.15 +
            (1.0 if ttr < 0.75         else 0.0)        * 0.10 +
            (1.0 if bigram_rep > 0.02  else 0.0)        * 0.10 +
            (1.0 if n_caps == 0        else 0.0)        * 0.10 +
            (1.0 if n_exclaim == 0     else 0.0)        * 0.08 +
            (1.0 if contr_r < 0.005    else 0.0)        * 0.07
        )

        return {
            # ── CG truncation ──
            "is_truncated":         is_truncated,
            "chars_tail":           chars_tail,
            # ── CG glued sentences ──
            "n_glued_sents":        n_glued,
            "glued_ratio":          glued_ratio,
            # ── Punctuation ──
            "n_exclaim":            n_exclaim,
            "exclaim_per_sent":     self._safe_ratio(n_exclaim, n_sents),
            "n_question":           n_question,
            "question_per_sent":    self._safe_ratio(n_question, n_sents),
            "n_ellipsis":           n_ellipsis,
            "n_repeated_punct":     n_repeat_p,
            "punct_density":        punct_d,
            "punct_entropy":        punct_ent,
            # ── Casing ──
            "n_caps_words":         n_caps,
            "caps_char_ratio":      caps_ratio,
            # ── Non-ASCII ──
            "n_non_ascii":          n_non_ascii,
            "non_ascii_ratio":      self._safe_ratio(n_non_ascii, n_chars),
            # ── Length / structure ──
            "n_words":              n_words,
            "n_chars":              n_chars,
            "n_sentences":          n_sents,
            "sent_len_mean":        slen_mean,
            "sent_len_std":         slen_std,
            "sent_len_range":       slen_range,
            "burstiness":           burstiness,
            # ── Vocabulary richness ──
            "type_token_ratio":     ttr,
            "hapax_ratio":          hapax_r,
            "avg_word_len":         wlen_mean,
            "word_len_std":         wlen_std,
            # ── Repetition ──
            "bigram_repetition":    bigram_rep,
            "trigram_repetition":   trigram_rep,
            "n_generic_phrases":    n_generic,
            "generic_phrase_ratio": generic_r,
            # ── Human vs CG vocabulary ──
            "n_contractions":       n_contr,
            "contraction_ratio":    contr_r,
            "n_formal_words":       n_formal,
            "formal_ratio":         formal_r,
            # ── Structural ──
            "has_url":              int(bool(self._URL_RE.search(t))),
            "n_urls":               self._count(self._URL_RE, t),
            "char_entropy":         char_ent,
            # ── Composite CG score ──
            "cg_signal_score":      cg_score,
        }

    def transform(self, texts) -> pd.DataFrame:
        return pd.DataFrame([self.extract(t) for t in texts])

    @property
    def feature_names(self):
        return list(self.extract("sample.").keys())


# ── CLI demo ──────────────────────────────────────────────────────────── #
if __name__ == "__main__":
    extractor = SpecialCharFeatureExtractor()
    cases = [
        ("OR — human",
         "My dog LOVES these things! All I have to say is \"greenie\" and he starts "
         "jumping up and down. It's not too expensive either."),
        ("CG 2022 — truncated",
         "This is a great bag. I love the look and feel of it, and the size is "
         "perfect. I had to get a size down, as I wear a 6"),
        ("CG 2022 — glued sentences",
         "A great read. The story is well told. The characters are "
         "well-developed.This is a great book to"),
        ("AI / CG modern — short",
         "Got this mini fridge for my Silom condo. Does exactly what it needs to do."),
        ("AI / CG modern — polished",
         "This milk frother is changing my life. Makes perfect foam every morning. "
         "Takes 5 seconds to rinse."),
    ]
    print(f"{'Type':<30} {'trunc':>5} {'glued':>5} {'words':>5} {'ttr':>5} "
          f"{'caps':>5} {'contr':>5} {'cg_score':>9}")
    print("-" * 70)
    for label, text in cases:
        f = extractor.extract(text)
        print(f"{label:<30} {f['is_truncated']:>5} {f['n_glued_sents']:>5} "
              f"{f['n_words']:>5} {f['type_token_ratio']:>5.2f} "
              f"{f['n_caps_words']:>5} {f['n_contractions']:>5} "
              f"{f['cg_signal_score']:>9.3f}")
    print(f"\nTotal features: {len(extractor.feature_names)}")