import os, time
from typing import Tuple, List
import numpy as np
from PIL import Image, ImageOps
import streamlit as st

# Optional backends
_TORCH_AVAILABLE = False
_TF_AVAILABLE = False
try:
    import torch; _TORCH_AVAILABLE = True
except: pass
try:
    import tensorflow as tf; _TF_AVAILABLE = True
except: pass

# ---------------------------- CONFIG / STYLE ----------------------------
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐾",
    layout="centered",
)

st.markdown("""
<style>
.hero {
    background: linear-gradient(135deg, #3b82f6 0%, #16a34a 100%);
    padding: 28px 20px; border-radius: 16px; color: white;
    text-align: center; margin-bottom: 20px;
}
.hero h1 {margin: 0; font-size: 2rem;}
.hero p {margin-top: 4px; opacity: 0.9;}
.card {
    border-radius: 16px; padding: 18px;
    background: rgba(255,255,255,0.05); margin-bottom: 16px;
    border: 1px solid rgba(255,255,255,0.1);
}
.prob-bar {height:10px; background:#e5e7eb; border-radius:999px; overflow:hidden;}
.prob-fill {height:100%; border-radius:999px;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------------------- HERO SECTION ----------------------------
st.markdown("""
<div class="hero">
    <h1>🐱🐶 Cat vs Dog Classifier</h1>
    <p>Real-time webcam & image upload powered by Deep Learning</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------- Sidebar ----------------------------
st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/616/616408.png", width=100
)
st.sidebar.header("⚙️ Settings")
model_path = "cnn_traditional.h5"
use_gpu = st.sidebar.checkbox("Use GPU if available (PyTorch)", value=True)
img_size = st.sidebar.slider("Input size", 64, 512, 224, step=16)
st.sidebar.markdown("---")
class0 = st.sidebar.text_input("Class 0 label", value="Cat")
class1 = st.sidebar.text_input("Class 1 label", value="Dog")
labels = [class0, class1]

# ---------------------------- Model Loader ----------------------------
@st.cache_resource
def load_any_model(path: str, prefer_gpu: bool = True):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext in [".pt", ".pth"]:
        if not _TORCH_AVAILABLE: raise RuntimeError("PyTorch not available")
        device = torch.device("cuda" if (prefer_gpu and torch.cuda.is_available()) else "cpu")
        model = torch.load(path, map_location=device)
        model.to(device).eval()
        return "torch", model, device
    elif ext in [".h5", ".keras"]:
        if not _TF_AVAILABLE: raise RuntimeError("TensorFlow not available")
        model = tf.keras.models.load_model(path, compile=False)
        return "tf", model, None
    else:
        raise ValueError("Unsupported model extension")

def preprocess_pil(img: Image.Image, size: int):
    img = ImageOps.exif_transpose(img).convert("RGB").resize((size, size))
    arr = np.array(img).astype(np.float32)/255.0
    return arr[None, ...]

def postprocess_logits(logits, backend: str):
    if backend == "torch":
        import torch
        with torch.no_grad():
            if logits.ndim==1: logits=logits[None,...]
            if logits.shape[1]==1:
                p1=torch.sigmoid(logits[:,0]); p0=1-p1
                probs=torch.stack([p0,p1],dim=1).cpu().numpy()
            else: probs=torch.softmax(logits,dim=1).cpu().numpy()
    else:
        if logits.ndim==1: logits=logits[None,...]
        if logits.shape[1]==1:
            p1=1/(1+np.exp(-logits[:,0])); p0=1-p1
            probs=np.stack([p0,p1],axis=1)
        else:
            x=logits-np.max(logits,axis=1,keepdims=True)
            probs=np.exp(x)/np.sum(np.exp(x),axis=1,keepdims=True)
    return probs, np.argmax(probs, axis=1)

# ---------------------------- Load Model ----------------------------
backend, model, device = None, None, None
try:
    with st.spinner("Loading model..."):
        backend, model, device = load_any_model(model_path, prefer_gpu=use_gpu)
    st.success(f"Model loaded ({backend.upper()})")
except Exception as e:
    st.error(f"Failed to load model: {e}")

# ---------------------------- Inference ----------------------------
def infer_image(pil_img: Image.Image):
    if backend is None or model is None: return
    x = preprocess_pil(pil_img, img_size)
    t0=time.time()
    if backend=="torch":
        out=model(torch.tensor(x).permute(0,3,1,2).to(device).float())
        logits=out.detach().cpu().numpy()
    else:
        out=model(x, training=False)
        logits = out.numpy() if hasattr(out,"numpy") else np.array(out)
    dt=(time.time()-t0)*1000
    probs, idx = postprocess_logits(logits, backend)
    pred_label, pred_prob = labels[int(idx[0])], float(probs[0,int(idx[0])])

    col1,col2 = st.columns([0.5,0.5])
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(pil_img, caption="Input Image", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"### 🎯 Prediction\n**{pred_label}** · {pred_prob*100:.2f}%")
        st.caption(f"Latency: {dt:.1f} ms")
        for i,p in enumerate(probs[0]):
            st.write(f"{labels[i]}: {p*100:.2f}%")
            st.markdown(f"""
                <div class="prob-bar">
                    <div class="prob-fill" style="width:{p*100:.2f}%;
                        background: linear-gradient(90deg,#3b82f6,#16a34a);"></div>
                </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------- Tabs ----------------------------
tab1, tab2, tab3 = st.tabs(["📸 Camera", "🖼️ Upload", "📚 About Dataset"])

with tab1:
    st.subheader("📸 Capture from Webcam")
    cam = st.camera_input("Take a photo")
    if cam: infer_image(Image.open(cam))

with tab2:
    st.subheader("🖼️ Upload an Image")
    file = st.file_uploader("Choose JPG/PNG", type=["jpg","jpeg","png"])
    if file: infer_image(Image.open(file))

with tab3:
    st.subheader("📚 About the Dataset")
    st.image("https://cdn-icons-png.flaticon.com/512/616/616408.png", width=80)
    st.markdown("""
    Dataset ini berisi ribuan gambar kucing 🐱 dan anjing 🐶 dari berbagai sumber online.  
    Ada variasi besar: breed berbeda, background rumah hingga outdoor, lighting terang dan gelap, 
    bahkan beberapa foto buram.  

    - **Jumlah data:** puluhan ribu gambar (balanced antara kucing & anjing)  
    - **Ukuran standar:** 224×224 piksel  
    - **Augmentasi:** flipping, rotation, brightness shift untuk membuat model lebih robust  
    - **Tujuan:** membuat model bisa membedakan *Cat* vs *Dog* dengan baik  

    > Fun fact: dataset ini sering dipakai untuk benchmarking model CNN klasik sampai arsitektur modern seperti ResNet & ViT.
    """)
