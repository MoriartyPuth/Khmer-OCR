"""
Khmer-only vocabulary.
CTC requires a blank token (index 0 by convention).
Covers the full Khmer Unicode block (U+1780–U+17FF) plus space.
"""
import math

# Full Khmer Unicode block — every character the model can predict
KHMER_CHARS = (
    # Consonants
    "កខគឃងចឆជឈញដឋឌឍណតថទធនបផពភមយរលវឝឞសហឡអ"
    # Independent vowels
    "ឣឤឥឦឧឩឪឫឬឭឮឯឰឱឲឳ"
    # Dependent vowels
    "ាិីឹឺុូួើឿៀេែៃោៅ"
    # Diacritics & signs
    "ំះៈ័៉៊់៌៍៎៏័៑្"
    # Punctuation
    "។៕៖ៗ៘៙៚៛៝"
    # Khmer digits
    "០១២៣៤៥៦៧៨៩"
    # Space
    " "
)

# Build vocab: blank=0, then all chars (deduplicated)
BLANK_TOKEN = 0
VOCAB = ["<blank>"] + list(dict.fromkeys(KHMER_CHARS))

CHAR2IDX = {ch: i for i, ch in enumerate(VOCAB)}
IDX2CHAR  = {i: ch for ch, i in CHAR2IDX.items()}

NUM_CLASSES = len(VOCAB)  # includes blank


def encode(text: str) -> list[int]:
    """Convert a text string to a list of integer indices."""
    return [CHAR2IDX[ch] for ch in text if ch in CHAR2IDX]


def decode(indices: list[int]) -> str:
    """Convert indices back to a string, skipping blank tokens."""
    return "".join(IDX2CHAR[i] for i in indices if i != BLANK_TOKEN)


def ctc_decode(indices: list[int]) -> str:
    """
    Greedy CTC decode:
    1. Collapse consecutive repeated tokens
    2. Remove blank tokens
    """
    if not indices:
        return ""
    collapsed = [indices[0]] + [
        indices[i] for i in range(1, len(indices))
        if indices[i] != indices[i - 1]
    ]
    return decode([i for i in collapsed if i != BLANK_TOKEN])


def _log_add(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    NEG_INF = float('-inf')
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    return max(a, b) + math.log1p(math.exp(-abs(a - b)))


def ctc_beam_search(log_probs, beam_width: int = 10, blank: int = BLANK_TOKEN) -> str:
    """
    CTC beam search decoder — pure Python, no extra dependencies.

    Explores `beam_width` candidate sequences simultaneously and returns the
    one with the highest total probability. Significantly outperforms greedy
    decoding on ambiguous Khmer clusters and short lines.

    Args:
        log_probs  : (T, C) array of log-softmax probabilities
        beam_width : number of beams to keep per timestep (10 is a good default)
        blank      : blank token index (0)

    Returns:
        Decoded text string
    """
    NEG_INF = float('-inf')
    T = len(log_probs)

    # beams: prefix (tuple of int) → [log_p_blank, log_p_non_blank]
    # Start with empty prefix, probability 1.0 via blank path
    beams: dict = {(): [0.0, NEG_INF]}

    for t in range(T):
        lp = log_probs[t]
        new_beams: dict = {}

        for prefix, (p_b, p_nb) in beams.items():
            p_total = _log_add(p_b, p_nb)

            # ── Extend with blank (prefix stays the same) ─────────────────────
            lp_blank = float(lp[blank])
            new_p_b = p_total + lp_blank
            if prefix not in new_beams:
                new_beams[prefix] = [NEG_INF, NEG_INF]
            new_beams[prefix][0] = _log_add(new_beams[prefix][0], new_p_b)

            # ── Extend with each non-blank character ──────────────────────────
            for c in range(len(lp)):
                if c == blank:
                    continue
                lp_c = float(lp[c])

                if prefix and prefix[-1] == c:
                    # Same char as end of prefix:
                    #   Path A — c follows blank → prefix grows (e.g. "aa")
                    ext = prefix + (c,)
                    if ext not in new_beams:
                        new_beams[ext] = [NEG_INF, NEG_INF]
                    new_beams[ext][1] = _log_add(new_beams[ext][1], p_b + lp_c)
                    #   Path B — c follows non-blank → CTC collapse, prefix stays
                    if prefix not in new_beams:
                        new_beams[prefix] = [NEG_INF, NEG_INF]
                    new_beams[prefix][1] = _log_add(new_beams[prefix][1], p_nb + lp_c)
                else:
                    # Different char: always extends to a new prefix
                    ext = prefix + (c,)
                    if ext not in new_beams:
                        new_beams[ext] = [NEG_INF, NEG_INF]
                    new_beams[ext][1] = _log_add(new_beams[ext][1], p_total + lp_c)

        # Prune to top beam_width by total log probability
        beams = dict(
            sorted(
                new_beams.items(),
                key=lambda kv: _log_add(kv[1][0], kv[1][1]),
                reverse=True,
            )[:beam_width]
        )

    if not beams:
        return ""
    best = max(beams, key=lambda p: _log_add(beams[p][0], beams[p][1]))
    return decode(list(best))