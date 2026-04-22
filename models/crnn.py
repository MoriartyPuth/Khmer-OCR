"""
CNN + BiLSTM OCR Model (CRNN-style)
─────────────────────────────────────────────────────────
Flow:
  Image (1, H, W)
    → CNN backbone  → feature map (C, H', W')
    → collapse H'   → sequence  (W', C*H')
    → BiLSTM        → contextual sequence  (W', hidden*2)
    → Linear        → logits  (W', num_classes)
    → CTC decode    → text

Why this works for Khmer:
  - CNN captures local glyph shapes (subscript consonants, vowels stacked
    above/below, diacritics).
  - BiLSTM reads the sequence in both directions so each position is aware
    of its full context — important for Khmer's complex ligatures.
  - CTC loss doesn't need character-level alignment: just feed the whole
    sequence and let the model figure out boundaries.
"""

import torch
import torch.nn as nn


# ── CNN Backbone ──────────────────────────────────────────────────────────────

class CNNBackbone(nn.Module):
    """
    Four conv blocks.  Each block: Conv → BN → ReLU → MaxPool (height only).

    Input : (B, 1, 32, W)
    Output: (B, 256, 1, W') where W' ≈ W/4
    """

    def __init__(self):
        super().__init__()

        def block(in_c, out_c, pool_h=True):
            layers = [
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            ]
            # Pool only height so width (= time steps) shrinks more gently
            if pool_h:
                layers.append(nn.MaxPool2d(kernel_size=(2, 1)))  # halve height only
            return nn.Sequential(*layers)

        self.net = nn.Sequential(
            block(1,   64),   # 32 → 16 height
            block(64, 128),   # 16 →  8 height
            block(128, 256),  #  8 →  4 height
            block(256, 256, pool_h=False),  # keep height at 4
            # Final: collapse remaining height dimension
            nn.Conv2d(256, 256, kernel_size=(4, 1)),  # 4 → 1 height
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)   # (B, 256, 1, W')


# ── Full Model ────────────────────────────────────────────────────────────────

class KhmerOCR(nn.Module):
    """
    Args:
        num_classes : size of vocabulary including CTC blank (index 0)
        rnn_hidden  : hidden units per direction in BiLSTM
        rnn_layers  : number of stacked BiLSTM layers
        dropout     : dropout between RNN layers
    """

    def __init__(
        self,
        num_classes: int,
        rnn_hidden: int = 256,
        rnn_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.cnn = CNNBackbone()

        # After CNN: feature size per time step = 256 channels × 1 height
        cnn_out_features = 256

        self.rnn = nn.LSTM(
            input_size=cnn_out_features,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
            batch_first=False,   # expects (T, B, F) — standard for CTC
        )

        self.classifier = nn.Linear(rnn_hidden * 2, num_classes)

    def forward(self, x):
        """
        x : (B, 1, H, W)  — batch of grayscale images
        returns logits : (T, B, num_classes)  where T = sequence length
        """
        # ── CNN ──────────────────────────────────────────────────────────────
        feat = self.cnn(x)           # (B, 256, 1, W')
        B, C, H, W = feat.shape
        assert H == 1, f"Expected height=1 after CNN, got {H}"

        feat = feat.squeeze(2)       # (B, 256, W')
        feat = feat.permute(2, 0, 1) # (W', B, 256) — time-first for RNN

        # ── BiLSTM ───────────────────────────────────────────────────────────
        rnn_out, _ = self.rnn(feat)  # (W', B, hidden*2)

        # ── Classifier ───────────────────────────────────────────────────────
        logits = self.classifier(rnn_out)  # (W', B, num_classes)
        return logits


# ── CTC Loss wrapper ──────────────────────────────────────────────────────────

class CTCLoss(nn.Module):
    """Thin wrapper around PyTorch's built-in CTCLoss."""

    def __init__(self, blank: int = 0):
        super().__init__()
        self.ctc = nn.CTCLoss(blank=blank, reduction="mean", zero_infinity=True)

    def forward(self, logits, labels, label_lengths):
        """
        logits        : (T, B, C) — raw model output
        labels        : (sum of label_lengths,) flat label tensor
        label_lengths : (B,) length of each label
        """
        log_probs = logits.log_softmax(dim=2)
        input_lengths = torch.full(
            (logits.size(1),), logits.size(0), dtype=torch.long
        )
        return self.ctc(log_probs, labels, input_lengths, label_lengths)


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    from utils.vocab import NUM_CLASSES

    model = KhmerOCR(num_classes=NUM_CLASSES)
    dummy = torch.randn(4, 1, 32, 200)   # batch=4, grayscale, h=32, w=200
    out   = model(dummy)
    print(f"Input : {dummy.shape}")
    print(f"Output: {out.shape}")   # expect (T, 4, NUM_CLASSES)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {total_params:,}")
