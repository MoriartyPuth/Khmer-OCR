"""
Khmer character-level language model (n-gram, stupid back-off)
─────────────────────────────────────────────────────────────────────────────
Why this exists:
    The CRNN reads each glyph in near-isolation, so it confuses visually similar
    Khmer clusters (្ត ↔ ្ទ, ន ↔ ណ) and occasionally drops trailing characters.
    These mistakes produce character sequences that are rare or impossible in
    real Khmer. A character n-gram LM scores how "Khmer-like" a sequence is, so
    we can re-rank the OCR's top hypotheses and pick the linguistically plausible
    one — exactly what production OCR engines do.

Design:
    • Pure Python, no external deps → deploys anywhere (Streamlit Cloud).
    • Stupid back-off (Brants et al. 2007): cheap, robust, no normalisation.
    • Counts pruned on save to keep the file small (a few MB).

Usage:
    lm = KhmerCharLM.load("outputs/khmer_lm.pkl")
    lm.logprob_text("សម្តេច")   # higher (more Khmer-like)
    lm.logprob_text("សម្ទេច")   # lower
"""

from __future__ import annotations

import math
import pickle
from collections import Counter
from typing import Dict, Iterable, List


BACKOFF = 0.4          # stupid-backoff discount per order dropped
FLOOR_LOGP = -18.0     # log-prob assigned to a fully unseen character


class KhmerCharLM:
    """Character n-gram model with stupid back-off scoring."""

    def __init__(self, n: int = 4):
        self.n = n
        # grams[k] : dict mapping a length-k character string -> count
        self.grams: Dict[int, Counter] = {k: Counter() for k in range(1, n + 1)}
        self.total = 0  # total characters seen (for the unigram denominator)

    # ── Training ────────────────────────────────────────────────────────────
    def add_text(self, text: str) -> None:
        """Accumulate n-gram counts from one line of text.

        A start marker (\\x02) is prepended so the model also learns which
        characters legitimately begin a line."""
        s = "\x02" + text
        L = len(s)
        for i in range(1, L):           # skip the marker itself as a unigram target
            self.total += 1
            for k in range(1, self.n + 1):
                if i - k + 1 >= 0:
                    self.grams[k][s[i - k + 1:i + 1]] += 1

    def build(self, texts: Iterable[str]) -> "KhmerCharLM":
        for t in texts:
            if t:
                self.add_text(t)
        return self

    def prune(self, min_count_by_order: Dict[int, int] | None = None) -> "KhmerCharLM":
        """Drop low-count high-order n-grams to shrink the saved model.
        Unigrams/bigrams are always kept (they back-stop every query)."""
        if min_count_by_order is None:
            min_count_by_order = {3: 2, 4: 2, 5: 3}
        for k, mc in min_count_by_order.items():
            if k in self.grams:
                self.grams[k] = Counter(
                    {g: c for g, c in self.grams[k].items() if c >= mc}
                )
        return self

    # ── Scoring ─────────────────────────────────────────────────────────────
    def logprob_char(self, context: str, ch: str) -> float:
        """log P(ch | context) via stupid back-off."""
        max_k = min(self.n - 1, len(context))
        penalty = 0.0
        for k in range(max_k, -1, -1):
            ctx = context[len(context) - k:] if k > 0 else ""
            higher = self.grams[k + 1].get(ctx + ch, 0)
            if higher > 0:
                denom = self.total if k == 0 else self.grams[k].get(ctx, 0)
                if denom > 0:
                    return penalty + math.log(higher / denom)
            penalty += math.log(BACKOFF)
        return FLOOR_LOGP

    def logprob_text(self, text: str) -> float:
        """Total log-likelihood of a string (sum over characters).

        Length-normalised scoring is the caller's job — return the raw sum so
        callers can add a length bonus when comparing hypotheses of different
        lengths."""
        if not text:
            return FLOOR_LOGP
        s = "\x02" + text
        total = 0.0
        for i in range(1, len(s)):
            total += self.logprob_char(s[1:i], s[i])
        return total

    # ── I/O ─────────────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {"n": self.n, "total": self.total,
                 "grams": {k: dict(v) for k, v in self.grams.items()}},
                f, protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: str) -> "KhmerCharLM":
        with open(path, "rb") as f:
            d = pickle.load(f)
        lm = cls(n=d["n"])
        lm.total = d["total"]
        lm.grams = {k: Counter(v) for k, v in d["grams"].items()}
        return lm

    def size_info(self) -> str:
        parts = [f"{k}-gram: {len(v):,}" for k, v in self.grams.items()]
        return f"n={self.n}, total_chars={self.total:,}, " + ", ".join(parts)


# ── Hypothesis re-ranking ─────────────────────────────────────────────────────

def rescore(
    hypotheses: List[tuple[str, float]],
    lm: KhmerCharLM | None,
    alpha: float = 0.5,
    beta: float = 0.6,
) -> str:
    """
    Pick the best hypothesis by combining the OCR (acoustic) score with the LM.

        score = am_logp + alpha * lm_logp + beta * len(text)

    alpha : LM weight. 0 disables the LM (returns the OCR's own best).
    beta  : per-character bonus that offsets the LM's bias toward shorter
            strings (otherwise dropped-character hypotheses win unfairly).

    `hypotheses` is the list of (text, am_logp) from ctc_beam_search_nbest,
    already sorted best-first, so with no LM we simply return the first.
    """
    if not hypotheses:
        return ""
    if lm is None or alpha <= 0:
        return hypotheses[0][0]

    best_text, best_score = hypotheses[0][0], float("-inf")
    for text, am_logp in hypotheses:
        score = am_logp + alpha * lm.logprob_text(text) + beta * len(text)
        if score > best_score:
            best_score, best_text = score, text
    return best_text
