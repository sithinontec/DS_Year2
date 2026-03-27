"""
feature_engineering.py
=======================
Extracts special character, punctuation, and emoji features from review text.
These features are PRESERVED (not stripped) as they are strong signals for
AI-generated vs. human-written text.

Key insight:
  - AI-generated text tends to be uniform, punctuation-balanced, emoji-free
  - Human text is noisy: repeated !!!, slang, emojis, typos, non-ASCII chars
"""

import re
import math
import unicodedata
import numpy as np
import pandas as pd
from collections import Counter

try:
    import emoji as emoji_lib
    EMOJI_AVAILABLE = True
except ImportError:
    EMOJI_AVAILABLE = False
    print("[WARNING] 'emoji' package not installed. Emoji features will be zeros.")


# ---------------------------------------------------------------------------
# Core extractor
# ---------------------------------------------------------------------------

class SpecialCharFeatureExtractor:
    """
    Extracts a rich set of special character and emoji features from text.
    All features are numeric and can be concatenated with TF-IDF vectors.
    """

    # Patterns compiled once for speed
    _EXCLAIM_RE      = re.compile(r'!')
    _QUESTION_RE     = re.compile(r'\?')
    _ELLIPSIS_RE     = re.compile(r'\.{2,}|…')
    _EMDASH_RE       = re.compile(r'—|--|–')
    _COMMA_RE        = re.compile(r',')
    _SEMICOLON_RE    = re.compile(r';')
    _COLON_RE        = re.compile(r':(?!//)')   # colon but not URL ://
    _REPEATED_PUNCT  = re.compile(r'([!?.])\1{1,}')  # !! ?? ..
    _ALL_CAPS_RE     = re.compile(r'\b[A-Z]{2,}\b')
    _WORD_RE         = re.compile(r'\b\w+\b')
    _SENT_RE         = re.compile(r'[.!?]+')
    _NON_ASCII_RE    = re.compile(r'[^\x00-\x7F]')
    _WHITESPACE_RE   = re.compile(r'\s+')
    _URL_RE          = re.compile(r'https?://\S+|www\.\S+')
    _HASHTAG_RE      = re.compile(r'#\w+')
    _MENTION_RE      = re.compile(r'@\w+')
    _DOLLAR_RE       = re.compile(r'\$')
    _PERCENT_RE      = re.compile(r'%')
    _STAR_RE         = re.compile(r'\*')
    _QUOTE_RE        = re.compile(r'["\u201c\u201d\u2018\u2019]')

    # Common emoji categories
    _FACE_EMOJI_RE   = re.compile(
        u'[\U0001F600-\U0001F64F]', flags=re.UNICODE)
    _SYMBOL_EMOJI_RE = re.compile(
        u'[\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF'
        u'\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F'
        u'\U0001FA70-\U0001FAFF\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251]+', flags=re.UNICODE)

    def _count(self, pattern, text):
        return len(pattern.findall(text))

    def _safe_ratio(self, numerator, denominator):
        return numerator / denominator if denominator > 0 else 0.0

    def _char_entropy(self, text):
        """Shannon entropy of character distribution — low = AI (uniform), high = human."""
        if not text:
            return 0.0
        freq = Counter(text)
        total = len(text)
        return -sum((c / total) * math.log2(c / total) for c in freq.values())

    def _punct_entropy(self, text):
        """Entropy over punctuation characters only."""
        punct = [c for c in text if unicodedata.category(c).startswith('P')]
        if not punct:
            return 0.0
        freq = Counter(punct)
        total = len(punct)
        return -sum((c / total) * math.log2(c / total) for c in freq.values())

    def _get_emojis(self, text):
        if EMOJI_AVAILABLE:
            # emoji_lib.emoji_list returns list of dicts with 'emoji' key
            return [e['emoji'] for e in emoji_lib.emoji_list(text)]
        else:
            # Fallback: use regex
            return self._FACE_EMOJI_RE.findall(text) + self._SYMBOL_EMOJI_RE.findall(text)

    def extract(self, text: str) -> dict:
        """
        Extract all features from a single text string.
        Returns a flat dict of feature_name -> float.
        """
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        n_chars  = max(len(text), 1)
        words    = self._WORD_RE.findall(text)
        n_words  = max(len(words), 1)
        sentences = [s for s in self._SENT_RE.split(text) if s.strip()]
        n_sents  = max(len(sentences), 1)

        # --- Punctuation counts ---
        n_exclaim    = self._count(self._EXCLAIM_RE, text)
        n_question   = self._count(self._QUESTION_RE, text)
        n_ellipsis   = self._count(self._ELLIPSIS_RE, text)
        n_emdash     = self._count(self._EMDASH_RE, text)
        n_comma      = self._count(self._COMMA_RE, text)
        n_semicolon  = self._count(self._SEMICOLON_RE, text)
        n_colon      = self._count(self._COLON_RE, text)
        n_repeat_p   = self._count(self._REPEATED_PUNCT, text)
        n_quote      = self._count(self._QUOTE_RE, text)
        n_dollar     = self._count(self._DOLLAR_RE, text)
        n_percent    = self._count(self._PERCENT_RE, text)
        n_star       = self._count(self._STAR_RE, text)
        n_hashtag    = self._count(self._HASHTAG_RE, text)
        n_mention    = self._count(self._MENTION_RE, text)

        # --- Casing ---
        n_caps_words = self._count(self._ALL_CAPS_RE, text)
        n_upper_chars = sum(1 for c in text if c.isupper())
        n_lower_chars = sum(1 for c in text if c.islower())

        # --- Non-ASCII / Unicode ---
        non_ascii    = self._NON_ASCII_RE.findall(text)
        n_non_ascii  = len(non_ascii)
        # Unicode categories
        n_unicode_letter = sum(1 for c in text if unicodedata.category(c).startswith('L') and ord(c) > 127)
        n_modifier_chars = sum(1 for c in text if unicodedata.category(c) in ('Lm', 'Sk'))

        # --- Emoji features ---
        emojis      = self._get_emojis(text)
        n_emojis    = len(emojis)
        n_unique_emojis = len(set(emojis))
        # Emoji diversity: unique / total  (high = human variety, low = repetitive)
        emoji_diversity = self._safe_ratio(n_unique_emojis, n_emojis)
        emoji_word_ratio = self._safe_ratio(n_emojis, n_words)
        face_emojis  = self._FACE_EMOJI_RE.findall(text)
        n_face_emoji = len(face_emojis)
        n_symb_emoji = n_emojis - n_face_emoji

        # --- Ratios ---
        exclaim_per_word  = self._safe_ratio(n_exclaim, n_words)
        exclaim_per_sent  = self._safe_ratio(n_exclaim, n_sents)
        question_per_sent = self._safe_ratio(n_question, n_sents)
        emdash_per_sent   = self._safe_ratio(n_emdash, n_sents)
        ellipsis_per_sent = self._safe_ratio(n_ellipsis, n_sents)
        comma_per_sent    = self._safe_ratio(n_comma, n_sents)
        caps_ratio        = self._safe_ratio(n_upper_chars, n_chars)
        non_ascii_ratio   = self._safe_ratio(n_non_ascii, n_chars)
        punct_density     = self._safe_ratio(
            n_exclaim + n_question + n_comma + n_colon + n_semicolon, n_chars)

        # --- Entropy features ---
        char_entropy  = self._char_entropy(text)
        punct_entropy = self._punct_entropy(text)

        # --- Structural ---
        avg_word_len  = np.mean([len(w) for w in words]) if words else 0.0
        avg_sent_len  = self._safe_ratio(n_words, n_sents)
        has_url       = int(bool(self._URL_RE.search(text)))
        n_urls        = self._count(self._URL_RE, text)

        # --- AI-style signals (composite) ---
        # AI tends to: no emojis, no repeated punct, low non-ASCII, balanced punctuation
        ai_signal_score = (
            (1.0 if n_emojis == 0 else 0.0) * 0.3 +
            (1.0 if n_repeat_p == 0 else 0.0) * 0.2 +
            (1.0 if n_non_ascii == 0 else 0.0) * 0.2 +
            (1.0 if n_caps_words == 0 else 0.0) * 0.15 +
            (1.0 if emdash_per_sent > 0.1 else 0.0) * 0.15
        )

        return {
            # --- Exclamation ---
            "n_exclaim": n_exclaim,
            "exclaim_per_word": exclaim_per_word,
            "exclaim_per_sent": exclaim_per_sent,
            # --- Question ---
            "n_question": n_question,
            "question_per_sent": question_per_sent,
            # --- Ellipsis & em-dash (AI loves these) ---
            "n_ellipsis": n_ellipsis,
            "ellipsis_per_sent": ellipsis_per_sent,
            "n_emdash": n_emdash,
            "emdash_per_sent": emdash_per_sent,
            # --- Other punctuation ---
            "n_comma": n_comma,
            "comma_per_sent": comma_per_sent,
            "n_semicolon": n_semicolon,
            "n_colon": n_colon,
            "n_repeated_punct": n_repeat_p,
            "n_quote": n_quote,
            "n_star": n_star,
            "n_dollar": n_dollar,
            "n_percent": n_percent,
            "punct_density": punct_density,
            # --- Casing ---
            "n_caps_words": n_caps_words,
            "caps_char_ratio": caps_ratio,
            "n_upper_chars": n_upper_chars,
            # --- Non-ASCII / Unicode ---
            "n_non_ascii": n_non_ascii,
            "non_ascii_ratio": non_ascii_ratio,
            "n_unicode_letter": n_unicode_letter,
            "n_modifier_chars": n_modifier_chars,
            # --- Emoji ---
            "n_emojis": n_emojis,
            "n_unique_emojis": n_unique_emojis,
            "emoji_diversity": emoji_diversity,
            "emoji_word_ratio": emoji_word_ratio,
            "n_face_emoji": n_face_emoji,
            "n_symbol_emoji": n_symb_emoji,
            # --- Social markers ---
            "n_hashtags": n_hashtag,
            "n_mentions": n_mention,
            "n_urls": n_urls,
            "has_url": has_url,
            # --- Entropy ---
            "char_entropy": char_entropy,
            "punct_entropy": punct_entropy,
            # --- Structure ---
            "avg_word_len": avg_word_len,
            "avg_sent_len": avg_sent_len,
            "n_words": n_words,
            "n_sentences": n_sents,
            "n_chars": n_chars,
            # --- Composite AI signal ---
            "ai_signal_score": ai_signal_score,
        }

    def transform(self, texts) -> pd.DataFrame:
        """
        Transform a list/Series of texts into a DataFrame of features.
        Use this to get a feature matrix for model training.
        """
        return pd.DataFrame([self.extract(t) for t in texts])

    @property
    def feature_names(self):
        return list(self.extract("sample text!").keys())


# ---------------------------------------------------------------------------
# CLI / standalone usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    extractor = SpecialCharFeatureExtractor()

    test_cases = [
        ("AI-style review",
         "This product exceeded all my expectations. The quality is remarkable, "
         "and the customer service was truly exceptional. I would highly recommend "
         "it to anyone looking for a reliable solution."),

        ("Human review with emojis",
         "omg i LOVE this!! 😍😍 best purchase ever tbh... "
         "kinda pricey but worth it 100%!! would def buy again 🛒✨"),

        ("Fake/spam review",
         "BEST PRODUCT EVER!!! BUY NOW!!! 5 STARS!!! AMAZING QUALITY!!! "
         "FAST SHIPPING!!! HIGHLY RECOMMEND!!!"),

        ("AI with em-dashes",
         "The craftsmanship is superb—truly a testament to modern engineering. "
         "Every detail has been considered—from the packaging to the finish—"
         "making this an outstanding purchase."),
    ]

    print("=" * 70)
    print("SPECIAL CHARACTER FEATURE EXTRACTION — DEMO")
    print("=" * 70)

    for label, text in test_cases:
        feats = extractor.extract(text)
        print(f"\n[{label}]")
        print(f"  Text: {text[:80]}...")
        print(f"  n_emojis={feats['n_emojis']}, n_exclaim={feats['n_exclaim']}, "
              f"n_emdash={feats['n_emdash']}, n_ellipsis={feats['n_ellipsis']}")
        print(f"  n_repeated_punct={feats['n_repeated_punct']}, "
              f"non_ascii_ratio={feats['non_ascii_ratio']:.3f}, "
              f"caps_ratio={feats['caps_char_ratio']:.3f}")
        print(f"  char_entropy={feats['char_entropy']:.3f}, "
              f"punct_entropy={feats['punct_entropy']:.3f}")
        print(f"  ⚡ ai_signal_score={feats['ai_signal_score']:.2f}")

    print("\n✅ Feature extractor working. Total features:", len(extractor.feature_names))
