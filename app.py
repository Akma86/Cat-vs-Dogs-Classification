# app.py
import io
import os
import sys
import time
from typing import Tuple, List, Optional

import numpy as np
from PIL import Image, ImageOps
import streamlit as st

# Optional backends: we'll import lazily
_TORCH_AVAILABLE = False
_TF_AVAILABLE = False
try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:
    pass
try:
    import tensorflow as tf
    _TF_AVAILABLE = True
except Exception:
    pass

# ---------------------------- UI THEME / STYLES ----------------------------
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐾",
    layout="centered",
)

# Minimal tasteful CSS for cards and headers
st.markdown(
    """
    <style>
      .hero {
        background: radial-gradient(1000px 400px at 10% 0%, rgba(59,130,246,0.10), transparent),
                    radial-gradient(900px 400px at 90% 0%, rgba(16,185,129,0.10), transparent);
        padding: 26px 22px; border-radius: 20px; border: 1px solid rgba(255,255,255,0.08);
      }
      .pill {
        display:inline-block; padding: 4px 10px; border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.15); font-size: 0.85rem; opacity: 0.8;
      }
      .card {
        border-radius: 18px; padding: 16px; border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.02);
      }
      .prob-bar {height:10px; background:#eee; border-radius:999px; overflow:hidden;}
      .prob-fill {height:100%; border-radius:999px;}
      .small-muted {font-size: 0.85rem; opacity: 0.8;}
      footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

with st.container():
    st.markdown('<div class="hero">', unsafe_allow_html=True)
    c1, c2 = st.columns([0.85, 0.15])
    with c1:
        st.markdown("### 🐾 Cat vs Dog — Real-time & Upload")
        st.markdown(
            '<span class="pill">PyTorch / TensorFlow</span> '
            '<span class="pill">Camera Capture</span> '
            '<span class="pill">Image Upload</span> '
            '<span class="pill">Pretty UI</span>',
            unsafe_allow_html=True,
        )
    with c2:
        st.write("")
        st.image(
            "https://raw.githubusercontent.com/serengil/deepface/master/icon/cat-dog.png",
            use_column_width=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------- Sidebar Controls ----------------------------
st.sidebar.header("⚙️ Settings")

model_path = 'cnn_traditional.h5'
use_gpu = st.sidebar.checkbox("Use GPU if available (PyTorch)", value=True)

img_size = st.sidebar.number_input("Input size (square)", min_value=32, max_value=1024, value=224, step=8)
scale_0_1 = st.sidebar.checkbox("Scale to [0,1]", value=True)
normalize = st.sidebar.checkbox("Normalize (mean/std)", value=True)
mean_str = st.sidebar.text_input("Mean (comma-separated)", value="0.485,0.456,0.406")
std_str  = st.sidebar.text_input("Std (comma-separated)",  value="0.229,0.224,0.225")

st.sidebar.markdown("---")
st.sidebar.subheader("Class Labels")
class0 = st.sidebar.text_input("Class 0 label", value="Cat")
class1 = st.sidebar.text_input("Class 1 label", value="Dog")
labels = [class0, class1]

# ---------------------------- Model Loader ----------------------------
@st.cache_resource(show_spinner=True)
def load_any_model(path: str, prefer_gpu: bool = True):
    """
    Load PyTorch (.pt/.pth) or TF/Keras (.h5/.keras) model based on file extension.
    Returns (backend, model, device_or_none).
    backend: "torch" or "tf"
    """
    path = path.strip()
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext in [".pt", ".pth"]:
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available in this environment.")
        device = torch.device("cuda" if (prefer_gpu and torch.cuda.is_available()) else "cpu")
        # map_location handles CPU fallback
        model = torch.load(path, map_location=device)
        # If it's a state_dict, you should replace with your model class; we attempt to handle nn.Module directly
        if isinstance(model, dict) and "state_dict" in model:
            raise RuntimeError(
                "Loaded a state_dict. Replace with your specific model class and load_state_dict in this script."
            )
        model.to(device)
        model.eval()
        return "torch", model, device

    elif ext in [".h5", ".keras"]:
        if not _TF_AVAILABLE:
            raise RuntimeError("TensorFlow/Keras not available in this environment.")
        model = tf.keras.models.load_model(path, compile=False)
        return "tf", model, None

    else:
        raise ValueError(f"Unsupported model extension: {ext}. Use .pt/.pth or .h5/.keras")


def _parse_stats(s: str) -> List[float]:
    try:
        return [float(x.strip()) for x in s.split(",") if x.strip() != ""]
    except Exception:
        return [0.0, 0.0, 0.0]


def preprocess_pil(img: Image.Image, backend: str):
    img = ImageOps.exif_transpose(img).convert("RGB")
    img = img.resize((128, 128))  # sesuai input model
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr[None, ...]  # (1,128,128,3)
    return arr



def postprocess_logits(logits, backend: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (probabilities, top_idx)
    Handles binary (1 logit) and 2+ class softmax outputs.
    """
    if backend == "torch":
        with torch.no_grad():
            if logits.ndim == 1:
                logits = logits[None, ...]
            if logits.shape[1] == 1:
                # Binary logit -> sigmoid -> [p0, p1]
                p1 = torch.sigmoid(logits[:, 0])
                p0 = 1 - p1
                probs = torch.stack([p0, p1], dim=1).cpu().numpy()
            else:
                probs = torch.softmax(logits, dim=1).cpu().numpy()
    else:
        if logits.ndim == 1:
            logits = logits[None, ...]
        if logits.shape[1] == 1:
            p1 = 1 / (1 + np.exp(-logits[:, 0]))
            p0 = 1 - p1
            probs = np.stack([p0, p1], axis=1)
        else:
            # numerically stable softmax
            x = logits - np.max(logits, axis=1, keepdims=True)
            probs = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    top_idx = np.argmax(probs, axis=1)
    return probs, top_idx


# ---------------------------- Load Model (if exists) ----------------------------
backend = None
model = None
device = None
load_error = None
if model_path:
    try:
        with st.spinner("Loading model..."):
            backend, model, device = load_any_model(model_path, prefer_gpu=use_gpu)
        st.success(f"Model loaded with backend: **{backend.upper()}**")
        if backend == "torch" and device:
            st.caption(f"Device: {device}")
    except Exception as e:
        load_error = str(e)
        st.error(f"Failed to load model: {e}")

# ---------------------------- Input Mode Tabs ----------------------------
tab1, tab2 = st.tabs(["📸 Live Camera (Take Portrait)", "🖼️ Upload Image"])

def infer_image(pil_img: Image.Image, show: bool = True):
    mean = _parse_stats(mean_str)
    std = _parse_stats(std_str)
    if len(mean) not in [1, 3]: mean = [0.485, 0.456, 0.406]
    if len(std) not in [1, 3]:  std = [0.229, 0.224, 0.225]
    if len(mean) == 1: mean = mean * 3
    if len(std) == 1:  std  = std  * 3

    if backend is None or model is None:
        st.warning("Model belum ter-load. Pastikan path benar di sidebar.")
        return

    x = preprocess_pil(pil_img, img_size)


    # Inference
    t0 = time.time()
    if backend == "torch":
        with torch.no_grad():
            x = x.to(device)
            out = model(x)
            if isinstance(out, (list, tuple)):
                out = out[0]
            logits = out.detach().cpu().numpy()
    else:
        out = model(x, training=False)
        logits = out.numpy() if hasattr(out, "numpy") else np.array(out)
    dt = (time.time() - t0) * 1000

    probs, top_idx = postprocess_logits(logits, backend)
    prob0, prob1 = float(probs[0, 0]), float(probs[0, 1])
    pred_idx = int(top_idx[0])
    pred_label = labels[pred_idx]
    pred_prob = float(probs[0, pred_idx])

    # UI render
    col_img, col_pred = st.columns([0.55, 0.45])
    with col_img:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(pil_img, caption="Input", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_pred:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Prediction")
        st.markdown(f"**{pred_label}**  ·  {pred_prob*100:.2f}%")
        st.caption(f"Latency: {dt:.1f} ms")

        # probability bars
        st.markdown("##### Probabilities")
        for i, p in enumerate([prob0, prob1]):
            lbl = labels[i] if i < len(labels) else f"Class {i}"
            st.write(f"{lbl}: {p*100:.2f}%")
            st.markdown(
                f'''
                <div class="prob-bar">
                  <div class="prob-fill" style="width:{p*100:.2f}%; background: linear-gradient(90deg, rgba(59,130,246,0.9), rgba(16,185,129,0.9));"></div>
                </div>
                ''',
                unsafe_allow_html=True
            )
            st.write("")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        f'<p class="small-muted">Tips: Pastikan ukuran & normalisasi sesuai training. Jika hasil aneh, cek mean/std dan urutan label.</p>',
        unsafe_allow_html=True,
    )


with tab1:
    st.subheader("📸 Ambil Foto dari Kamera")
    cam_img = st.camera_input("Arahkan ke kucing/anjing lalu klik 'Take Photo'.")
    if cam_img is not None:
        pil = Image.open(cam_img)
        infer_image(pil)

with tab2:
    st.subheader("🖼️ Upload Gambar")
    file = st.file_uploader("Pilih file gambar (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if file is not None:
        pil = Image.open(file)
        infer_image(pil)

# ---------------------------- Footer / Diagnostics ----------------------------
with st.expander("🔍 Diagnostics"):
    st.write("Backend availability:", {"torch": _TORCH_AVAILABLE, "tensorflow": _TF_AVAILABLE})
    st.write("Model path:", model_path)
    if load_error:
        st.error(load_error)
    st.caption("Binary output (1 logit) diinterpretasikan sebagai [Cat, Dog] via sigmoid. "
               "Multi-class akan di-softmax lalu diambil top-1.")

