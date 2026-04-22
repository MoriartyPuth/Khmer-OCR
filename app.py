"""
Khmer OCR — Streamlit GUI
"""

import io
import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predict import load_model, predict_image, predict_document
from improved_document_predict import predict_document_improved
from utils.improved_line_segmentation import segment_document_improved

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Khmer OCR",
    page_icon="🔤",
    layout="wide",
)

# ── Theme ──────────────────────────────────────────────────────────────────────

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

if st.session_state.dark_mode:
    bg        = "#0e1117"
    sidebar   = "#262730"
    surface   = "#1e2130"
    text      = "#fafafa"
    text_muted = "#a0aab8"
    border    = "#3a3f55"
    input_bg  = "#1e2130"
else:
    bg        = "#ffffff"
    sidebar   = "#f0f2f6"
    surface   = "#f8f9fd"
    text      = "#31333f"
    text_muted = "#6b7280"
    border    = "#e0e3eb"
    input_bg  = "#ffffff"

st.markdown(f"""
<style>
/* ── Main background ── */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"] {{
    background-color: {bg} !important;
}}

/* ── Top header bar ── */
[data-testid="stHeader"] {{
    background-color: {bg} !important;
    border-bottom: 1px solid {border} !important;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div {{
    background-color: {sidebar} !important;
}}

/* ── Text ── */
.stApp, .stApp p, .stApp label,
.stApp .stMarkdown, .stApp span,
.stApp h1, .stApp h2, .stApp h3 {{
    color: {text} !important;
}}

/* ── Inputs ── */
.stApp input, .stApp textarea {{
    background-color: {input_bg} !important;
    color: {text} !important;
    border-color: {border} !important;
}}

/* ── File uploader dropzone ── */
[data-testid="stFileUploader"],
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploader"] > div,
[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] section > div {{
    background-color: {surface} !important;
    border-color: {border} !important;
    color: {text} !important;
}}
[data-testid="stFileUploaderDropzoneInput"] + div {{
    color: {text_muted} !important;
}}

/* ── Expanders ── */
[data-testid="stExpander"] {{
    background-color: {surface} !important;
    border-color: {border} !important;
}}

/* ── Info / alert boxes ── */
[data-testid="stNotification"],
[data-testid="stInfo"] > div {{
    background-color: {surface} !important;
    border-color: {border} !important;
    color: {text} !important;
}}

/* ── Captions / muted text ── */
.stApp .stCaption, .stApp small {{
    color: {text_muted} !important;
}}

/* ── Browse files button ── */
[data-testid="stFileUploaderDropzone"] button {{
    background-color: {"#dbeafe" if not st.session_state.dark_mode else "#1e3a5f"} !important;
    color: {"#1d4ed8" if not st.session_state.dark_mode else "#93c5fd"} !important;
    border: 1px solid {"#93c5fd" if not st.session_state.dark_mode else "#2563eb"} !important;
    font-weight: 600 !important;
}}
[data-testid="stFileUploaderDropzone"] button:hover {{
    background-color: {"#bfdbfe" if not st.session_state.dark_mode else "#1e40af"} !important;
}}

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {{
    background-color: {"#dbeafe" if not st.session_state.dark_mode else "#1e3a5f"} !important;
    color: {"#1d4ed8" if not st.session_state.dark_mode else "#93c5fd"} !important;
    border: 1px solid {"#93c5fd" if not st.session_state.dark_mode else "#2563eb"} !important;
    font-weight: 600 !important;
}}
[data-testid="stDownloadButton"] > button:hover {{
    background-color: {"#bfdbfe" if not st.session_state.dark_mode else "#1e40af"} !important;
}}

/* ── Theme toggle button ── */
[data-testid="stSidebar"] [data-testid="stButton"]:first-child > button {{
    background-color: {"#dbeafe" if not st.session_state.dark_mode else "#1e3a5f"} !important;
    color: {"#1d4ed8" if not st.session_state.dark_mode else "#93c5fd"} !important;
    border: 1px solid {"#93c5fd" if not st.session_state.dark_mode else "#2563eb"} !important;
    font-weight: 600 !important;
}}
[data-testid="stSidebar"] [data-testid="stButton"]:first-child > button:hover {{
    background-color: {"#bfdbfe" if not st.session_state.dark_mode else "#1e40af"} !important;
}}

/* ── Hide Streamlit menu ── */
#MainMenu, footer {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)

st.title("Khmer OCR")
st.caption("CNN + BiLSTM model trained on Khmer script (CTC loss). Val CER: 1.93%")

# ── Sidebar: settings ──────────────────────────────────────────────────────────

with st.sidebar:
    label = "🌙 Dark mode" if not st.session_state.dark_mode else "☀️ Light mode"
    if st.button(label, use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

    st.divider()
    st.header("Settings")

    checkpoint_path = st.text_input(
        "Checkpoint path",
        value="outputs/best_model.pth",
        help="Path to the .pth model checkpoint file",
    )

    mode = st.radio(
        "Prediction mode",
        ["Single line image", "Full document"],
        help=(
            "Single line: image contains one text line.\n"
            "Full document: image is a full page — segments into lines automatically."
        ),
    )

    st.subheader("Document settings")
    threshold = st.slider("Binary threshold", 50, 200, 127, help="For line segmentation")
    min_gap = st.slider("Min line gap (px)", 1, 20, 3)
    min_height = st.slider("Min line height (px)", 1, 30, 5)
    padding_tb = st.slider("Vertical padding (px)", 0, 30, 8, help="Extra pixels above/below each line (important for Khmer diacritics)")
    padding_lr = st.slider("Horizontal padding (px)", 0, 20, 2)
    use_deskew = st.checkbox("Deskew lines", value=True)
    use_improved = st.checkbox("Improved mode (confidence scores)", value=True)
    conf_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.6, step=0.05,
                                help="Lines below this confidence will be flagged")

    st.divider()
    device_label = "cuda" if torch.cuda.is_available() else "cpu"
    st.caption(f"Device: **{device_label}**")

# ── Load model (cached) ────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model…")
def get_model(ckpt_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    abs_path = Path(ckpt_path).resolve()
    if not abs_path.exists():
        return None, None, None, str(abs_path)
    model, img_height = load_model(str(abs_path), device)
    return model, img_height, device, None


model, img_height, device, load_err = get_model(checkpoint_path)

if load_err:
    st.error(f"Checkpoint not found: `{load_err}`\nUpdate the path in the sidebar.")
    st.stop()

# ── Image upload ───────────────────────────────────────────────────────────────

st.subheader("Upload image")
uploaded = st.file_uploader(
    "Choose an image file",
    type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
    help="Upload a Khmer text image. For 'Full document' mode, upload a full page scan.",
)

if uploaded is None:
    st.info("Upload an image to begin.")
    st.stop()

# Show upload metadata
img_bytes = uploaded.read()
pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

col_img, col_meta = st.columns([2, 1])
with col_img:
    st.image(pil_img, caption=f"{uploaded.name}", use_container_width=True)
with col_meta:
    st.markdown("**File info**")
    st.write(f"**Name:** {uploaded.name}")
    st.write(f"**Size:** {len(img_bytes) / 1024:.1f} KB")
    st.write(f"**Dimensions:** {pil_img.width} × {pil_img.height} px")
    st.write(f"**Mode:** {pil_img.mode}")
    # Save to a temp file so we can show the real resolved path
    tmp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=Path(uploaded.name).suffix or ".png",
        prefix="khmer_ocr_",
    )
    tmp.write(img_bytes)
    tmp.flush()
    tmp_path = Path(tmp.name).resolve()
    tmp.close()
    st.write(f"**Temp path:** `{tmp_path}`")

st.divider()

# ── Run OCR ────────────────────────────────────────────────────────────────────

run_btn = st.button("Run OCR", type="primary", use_container_width=True)

if run_btn:
    if mode == "Single line image":
        with st.spinner("Running OCR…"):
            result = predict_image(model, pil_img, img_height, device)

        st.subheader("Result")
        st.text_area("Predicted text", value=result, height=120, label_visibility="collapsed")
        st.download_button(
            "Download result (.txt)",
            data=result.encode("utf-8"),
            file_name=Path(uploaded.name).stem + "_ocr.txt",
            mime="text/plain",
        )

    else:
        # Full document mode
        if use_improved:
            with st.spinner("Segmenting and recognising lines (improved mode)…"):
                full_text, line_texts, diagnostics = predict_document_improved(
                    model,
                    str(tmp_path),
                    img_height,
                    device,
                    threshold=threshold,
                    min_gap=min_gap,
                    min_height=min_height,
                    padding_top_bottom=padding_tb,
                    padding_left_right=padding_lr,
                    deskew=use_deskew,
                    confidence_threshold=conf_threshold,
                    verbose=False,
                    diagnostics=True,
                )

            st.subheader("Full document text")
            st.text_area("", value=full_text, height=300, label_visibility="collapsed")
            st.download_button(
                "Download result (.txt)",
                data=full_text.encode("utf-8"),
                file_name=Path(uploaded.name).stem + "_ocr.txt",
                mime="text/plain",
            )

            st.subheader(f"Line details — {len(line_texts)} lines detected")
            # Segment again to get crops for display
            with st.spinner("Generating line previews…"):
                line_imgs, line_bounds, _ = segment_document_improved(
                    pil_img,
                    threshold=threshold,
                    min_gap=min_gap,
                    min_height=min_height,
                    padding_top_bottom=padding_tb,
                    padding_left_right=padding_lr,
                    deskew=use_deskew,
                    return_metadata=True,
                )

            for i, (diag, line_text) in enumerate(zip(diagnostics, line_texts)):
                conf = diag.get("confidence", 0.0)
                status = diag.get("status", "ok")
                conf_color = "🟢" if conf >= conf_threshold else "🔴"

                with st.expander(
                    f"Line {i+1} — conf: {conf:.3f} {conf_color}  |  {line_text[:60]}{'…' if len(line_text) > 60 else ''}",
                    expanded=False,
                ):
                    cols = st.columns([1, 2])
                    with cols[0]:
                        if i < len(line_imgs):
                            st.image(line_imgs[i], use_container_width=True)
                            if i < len(line_bounds):
                                y0, y1 = line_bounds[i]
                                st.caption(f"Rows {y0}–{y1} (padded ±{padding_tb}px)")
                    with cols[1]:
                        st.markdown(f"**Text:** {line_text}")
                        st.markdown(f"**Confidence:** {conf:.4f}")
                        st.markdown(f"**Status:** {status}")

        else:
            # Basic document prediction
            with st.spinner("Segmenting and recognising lines…"):
                full_text = predict_document(
                    model,
                    pil_img,
                    img_height,
                    device,
                    threshold=threshold,
                    min_gap=min_gap,
                    min_height=min_height,
                    expand_margin=padding_tb,
                )

            st.subheader("Full document text")
            st.text_area("", value=full_text, height=300, label_visibility="collapsed")
            st.download_button(
                "Download result (.txt)",
                data=full_text.encode("utf-8"),
                file_name=Path(uploaded.name).stem + "_ocr.txt",
                mime="text/plain",
            )

            lines = [l for l in full_text.split("\n") if l.strip()]
            st.caption(f"{len(lines)} lines recognised.")

    # Clean up temp file
    try:
        os.unlink(tmp_path)
    except Exception:
        pass
