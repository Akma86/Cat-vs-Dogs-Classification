# =================== app.py ===================
import io, os, sys, time
from typing import Tuple, List
import numpy as np
from PIL import Image, ImageOps
import streamlit as st

# Optional backends
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

st.markdown("""
<style>
.hero {
    background: radial-gradient(1000px 400px at 10% 0%, rgba(59,130,246,0.10), transparent),
                radial-gradient(900px 400px at 90% 0%, rgba(16,185,129,0.10), transparent);
    padding: 26px 22px; border-radius: 20px; border: 1px solid rgba(255,255,255,0.08);
    text-align:center;
}
.pill {
    display:inline-block; padding: 4px 10px; border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.15); font-size: 0.85rem; opacity: 0.8;
}
.card {
    border-radius: 18px; padding: 16px; border: 1px solid rgba(255,255,255,0.08);
    background: rgba(255,255,255,0.02); margin-bottom:10px;
}
.prob-bar {height:10px; background:#eee; border-radius:999px; overflow:hidden;}
.prob-fill {height:100%; border-radius:999px;}
.small-muted {font-size: 0.85rem; opacity: 0.8;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------------------- Hero Section ----------------------------
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
        st.image("https://raw.githubusercontent.com/serengil/deepface/master/icon/cat-dog.png",
                 use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------- Sidebar Controls ----------------------------
st.sidebar.header("⚙️ Settings")
model_path = 'cnn_traditional.h5'
use_gpu = st.sidebar.checkbox("Use GPU if available (PyTorch)", value=True)
img_size = st.sidebar.number_input("Input size (square)", min_value=32, max_value=1024, value=224, step=8)
scale_0_1 = st.sidebar.checkbox("Scale to [0,1]", value=True)
normalize = st.sidebar.checkbox("Normalize (mean/std)", value=False)
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
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext in [".pt", ".pth"]:
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        device = torch.device("cuda" if (prefer_gpu and torch.cuda.is_available()) else "cpu")
        model = torch.load(path, map_location=device)
        if isinstance(model, dict) and "state_dict" in model:
            raise RuntimeError("Replace with your model class for state_dict.")
        model.to(device).eval()
        return "torch", model, device
    elif ext in [".h5", ".keras"]:
        if not _TF_AVAILABLE:
            raise RuntimeError("TensorFlow not available")
        model = tf.keras.models.load_model(path, compile=False)
        return "tf", model, None
    else:
        raise ValueError(f"Unsupported model extension: {ext}")

# ---------------------------- Helper Functions ----------------------------
def _parse_stats(s: str) -> List[float]:
    try: return [float(x.strip()) for x in s.split(",") if x.strip() != ""]
    except: return [0.0, 0.0, 0.0]

def preprocess_pil(img: Image.Image, size: int):
    img = ImageOps.exif_transpose(img).convert("RGB").resize((size, size))
    arr = np.array(img).astype(np.float32)/255.0
    arr = arr[None, ...]  # (1,H,W,3)
    return arr

def postprocess_logits(logits, backend: str):
    if backend == "torch":
        import torch
        with torch.no_grad():
            if logits.ndim==1: logits=logits[None,...]
            if logits.shape[1]==1:
                p1=torch.sigmoid(logits[:,0]); p0=1-p1
                probs = torch.stack([p0,p1],dim=1).cpu().numpy()
            else:
                probs = torch.softmax(logits,dim=1).cpu().numpy()
    else:
        if logits.ndim==1: logits=logits[None,...]
        if logits.shape[1]==1:
            p1 = 1/(1+np.exp(-logits[:,0]))
            p0 = 1-p1
            probs=np.stack([p0,p1],axis=1)
        else:
            x = logits - np.max(logits, axis=1, keepdims=True)
            probs = np.exp(x)/np.sum(np.exp(x), axis=1, keepdims=True)
    top_idx = np.argmax(probs, axis=1)
    return probs, top_idx

# ---------------------------- Load Model ----------------------------
backend = None; model = None; device = None; load_error=None
if model_path:
    try:
        with st.spinner("Loading model..."):
            backend, model, device = load_any_model(model_path, prefer_gpu=use_gpu)
        st.success(f"Model loaded with backend: **{backend.upper()}**")
        if backend=="torch" and device: st.caption(f"Device: {device}")
    except Exception as e:
        load_error=str(e)
        st.error(f"Failed to load model: {e}")

# ---------------------------- Input Mode Tabs ----------------------------
tab1, tab2 = st.tabs(["📸 Camera", "🖼️ Upload Image"])

def infer_image(pil_img: Image.Image):
    if backend is None or model is None: return st.warning("Model belum ter-load")
    x = preprocess_pil(pil_img, img_size)
    t0 = time.time()
    if backend=="torch":
        with torch.no_grad(): out=model(x.to(device)) if hasattr(x,'to') else model(x)
        logits = out.detach().cpu().numpy() if hasattr(out,'detach') else np.array(out)
    else:
        out=model(x, training=False)
        logits = out.numpy() if hasattr(out,"numpy") else np.array(out)
    dt=(time.time()-t0)*1000
    probs, top_idx = postprocess_logits(logits, backend)
    pred_label = labels[int(top_idx[0])]
    pred_prob = float(probs[0,int(top_idx[0])])

    # --- UI Render ---
    col_img, col_pred = st.columns([0.55,0.45])
    with col_img:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(pil_img, caption="Input", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col_pred:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Prediction")
        st.markdown(f"**{pred_label}**  ·  {pred_prob*100:.2f}%")
        st.caption(f"Latency: {dt:.1f} ms")
        st.markdown("##### Probabilities")
        for i,p in enumerate([float(probs[0,0]), float(probs[0,1])]):
            st.write(f"{labels[i]}: {p*100:.2f}%")
            st.markdown(f'''
                <div class="prob-bar">
                  <div class="prob-fill" style="width:{p*100:.2f}%;
                       background: linear-gradient(90deg, rgba(59,130,246,0.9), rgba(16,185,129,0.9));"></div>
                </div>
            ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<p class="small-muted">Tips: Pastikan ukuran & normalisasi sesuai training.</p>', unsafe_allow_html=True)

with tab1:
    st.subheader("📸 Ambil Foto dari Kamera")
    cam_img = st.camera_input("Arahkan ke kucing/anjing lalu klik 'Take Photo'.")
    if cam_img is not None: infer_image(Image.open(cam_img))

with tab2:
    st.subheader("🖼️ Upload Gambar")
    file = st.file_uploader("Pilih file gambar (JPG/PNG)", type=["jpg","jpeg","png"])
    if file is not None: infer_image(Image.open(file))

# ---------------------------- Dataset Info ----------------------------
with st.expander("📚 About the Dataset (Yapping)"):
    st.markdown("""
    Dataset ini berisi ribuan gambar kucing 🐱 dan anjing 🐶 yang diambil dari internet.
    Ada berbagai breed, lighting, background, dan pose. Tujuannya agar model bisa
    mengenali perbedaan kucing dan anjing dengan akurat.  

    Semua gambar di-resize ke 224x224 dan dinormalisasi [0,1]. Augmentasi sederhana
    digunakan seperti flipping, rotation, dan brightness.  

    Basically, ini model belajar "ini kucing" atau "ini anjing" saja. Silakan coba
    upload foto peliharaanmu sendiri untuk tes.
    """)
