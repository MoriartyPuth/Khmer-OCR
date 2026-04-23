# 🔤 Khmer OCR

A deep learning OCR system for recognising **Khmer (Cambodian) script** using a CNN + BiLSTM architecture trained with CTC loss. Includes a full Streamlit web GUI for single-line and full-document recognition.

Live Demo : [Khmer OCR](https://khmer-ocr.streamlit.app/)

---

## 📊 Model Performance

| Metric | Value |
| --- | --- |
| Architecture | CNN + BiLSTM + CTC |
| Vocabulary size | 101 classes |
| Val CER (Character Error Rate) | **1.93%** |
| Training epochs | 63 |
| Device | CPU / CUDA |

---

## 🗂️ Project Structure

```text
Khmer-OCR/
├── 📁 data/
│   └── dataset.py                     # Dataset loader, transforms, Khmer filtering
├── 📁 models/
│   └── crnn.py                        # CNN backbone + BiLSTM + CTC loss
├── 📁 utils/
│   ├── vocab.py                       # Khmer character vocabulary & CTC decode
│   └── improved_line_segmentation.py  # Line detection, padding, deskewing
├── 📁 outputs/
│   └── best_model.pth                 # Trained model checkpoint (not in repo — see below)
├── 📁 testing_image/                  # Sample document images
├── app.py                             # ✨ Streamlit GUI
├── predict.py                         # Inference: single image, folder, parquet
├── improved_document_predict.py       # Document OCR with confidence scoring
├── interactive_predict.py             # Interactive CLI (line-by-line)
├── interactive_improved_predict.py    # Interactive CLI (document, improved)
├── train.py                           # Training script
├── evaluate.py                        # Evaluation: CER + WER metrics
├── validate_preprocessing.py          # Diagnostic tool for preprocessing comparison
└── requirements.txt
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/MoriartyPuth/Khmer-OCR.git
cd Khmer-OCR
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the model checkpoint

The trained model (`outputs/best_model.pth`) is not stored in the repository due to file size. Download it and place it at `outputs/best_model.pth`.

> If you want to train from scratch, see the Training section below.

---

## 🖥️ Streamlit GUI

The easiest way to use the OCR system is through the web interface.

```bash
streamlit run app.py
```

Then open <http://localhost:8501> in your browser.

### Features

| Feature | Description |
| --- | --- |
| 🌙 / ☀️ Light & Dark mode | Toggle in the sidebar |
| 📝 Single line mode | Upload one cropped line of Khmer text |
| 📄 Full document mode | Upload a full page — auto-segments into lines |
| 🔍 Confidence scores | Per-line confidence with 🟢 / 🔴 indicators |
| 📥 Download result | Export recognised text as `.txt` |
| 🔧 Adjustable settings | Threshold, padding, deskew, confidence threshold |

---

## 🔮 Inference

### Single image (command line)

```bash
python predict.py \
    --checkpoint outputs/best_model.pth \
    --image path/to/line.png
```

### Full document

```bash
python predict.py \
    --checkpoint outputs/best_model.pth \
    --document path/to/document.png
```

### Folder of images

```bash
python predict.py \
    --checkpoint outputs/best_model.pth \
    --image_dir path/to/folder/
```

### From a Parquet file

```bash
python predict.py \
    --checkpoint outputs/best_model.pth \
    --parquet data/unlabeled.parquet
```

### Interactive CLI

```bash
# Line-by-line
python interactive_predict.py --checkpoint outputs/best_model.pth

# Document mode with confidence
python interactive_improved_predict.py \
    --checkpoint outputs/best_model.pth \
    --padding_tb 8 \
    --deskew \
    --confidence_threshold 0.60
```

---

## 🏋️ Training

### Prepare your data

Training data should be a Parquet file with two columns:

- `image` — PIL Image or raw bytes
- `text` — Ground-truth Khmer Unicode string

### Run training

```bash
python train.py \
    --train_parquet data/train.parquet \
    --val_parquet   data/val.parquet \
    --epochs        50 \
    --batch_size    32 \
    --img_height    32 \
    --lr            0.001 \
    --rnn_hidden    256 \
    --rnn_layers    2 \
    --output_dir    outputs/
```

### Resume training

```bash
python train.py \
    --train_parquet data/train.parquet \
    --val_parquet   data/val.parquet \
    --resume
```

The best checkpoint (lowest val CER) is saved automatically to `outputs/best_model.pth`.

---

## 📈 Evaluation

```bash
python evaluate.py \
    --checkpoint        outputs/best_model.pth \
    --val_full_parquet  data/val.parquet \
    --batch_size        32 \
    --num_samples       10
```

Reports **CER** (Character Error Rate) and **WER** (Word Error Rate).

---

## 🔬 Validate Preprocessing

Compare the impact of different preprocessing strategies (padding, deskewing) on a test image:

```bash
python validate_preprocessing.py \
    --image      testing_image/document.png \
    --checkpoint outputs/best_model.pth
```

Outputs a side-by-side comparison of 3 pipelines — original, padded, and padded + deskewed — with confidence scores for each.

---

## 🧠 Architecture

```text
Input Image (H=32, variable W)
        │
        ▼
┌─────────────────────┐
│   CNN Backbone      │  4 conv blocks → (B, 256, 1, W')
│   (feature extract) │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   BiLSTM            │  2 layers, hidden=256 → (W', B, 512)
│   (sequence model)  │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   Linear + CTC      │  → (W', B, num_classes=101)
│   (decoder)         │
└─────────────────────┘
        │
        ▼
   Khmer text string
```

### CNN Backbone detail

| Block | Channels | Pooling | Output height |
| --- | --- | --- | --- |
| Block 1 | 1 → 64 | MaxPool(2,1) | 32 → 16 |
| Block 2 | 64 → 128 | MaxPool(2,1) | 16 → 8 |
| Block 3 | 128 → 256 | MaxPool(2,1) | 8 → 4 |
| Block 4 | 256 → 256 | Conv(4,1) | 4 → 1 |

### Hyperparameter defaults

| Parameter | Default | Description |
| --- | --- | --- |
| `img_height` | 32 | Input height (aspect ratio preserved) |
| `rnn_hidden` | 256 | Hidden units per LSTM direction |
| `rnn_layers` | 2 | Stacked LSTM layers |
| `dropout` | 0.2 | Regularization between LSTM layers |
| `batch_size` | 32 | Training batch size |
| `lr` | 1e-3 | Adam learning rate |

---

## 📄 Document Segmentation

Full-document OCR works in 7 steps:

1. 🎨 **Grayscale + binary threshold** — converts the image to black & white
2. 📊 **Horizontal projection** — sums black pixels per row to find text regions
3. 📏 **Line boundary detection** — finds gaps between text rows
4. 🔲 **Padding** — adds 8 px above/below each line (critical for Khmer diacritics)
5. 📐 **Deskewing** — straightens slightly rotated lines using `minAreaRect`
6. 🔮 **Per-line OCR** — each line is resized to 32 px height and passed through the model
7. 📉 **Confidence filtering** — lines below the threshold are flagged

### Segmentation parameters

| Parameter | Default | Effect |
| --- | --- | --- |
| `threshold` | 127 | Binary cutoff for line detection |
| `min_gap` | 3 px | Min blank rows between two lines |
| `min_height` | 5 px | Min line height (shorter = noise) |
| `padding_top_bottom` | 8 px | Vertical padding per line |
| `padding_left_right` | 2 px | Horizontal padding per line |
| `deskew` | True | Straighten tilted lines |
| `confidence_threshold` | 0.60 | Flag lines below this confidence |

---

## 📦 Requirements

```text
torch
torchvision
streamlit
pandas
pyarrow
Pillow
opencv-python
numpy
scipy
```

Install all with:

```bash
pip install -r requirements.txt
```

---

## 📜 License

This project is for educational and research purposes.

---

## 🙏 Acknowledgements

- Khmer Unicode Standard — [unicode.org](https://unicode.org)
- PyTorch CTC Loss — [pytorch.org](https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html)
- Streamlit — [streamlit.io](https://streamlit.io)
- Contributor - [MariyaThorn](https://github.com/MariyaThorn)
