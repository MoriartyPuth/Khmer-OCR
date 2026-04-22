"""
Khmer-only vocabulary.
CTC requires a blank token (index 0 by convention).
Covers the full Khmer Unicode block (U+1780–U+17FF) plus space.
"""

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