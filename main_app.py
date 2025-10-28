"""
Heritage Restoration AI - FINAL Polished Version (cv2 fix, thin crack masking)
No Streamlit warnings, perfect step status, safe restoration.
"""
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

from src.image_enhancement import ImageEnhancer
from src.crack_segmentation import CrackSegmentationModel
from src.texture_classification import TextureClassifier
from src.inpainting import InpaintingGAN

@st.cache_resource
def load_models():
    return (
        ImageEnhancer(),
        CrackSegmentationModel(),
        TextureClassifier(),
        InpaintingGAN()
    )

st.set_page_config(
    page_title="Heritage Restoration AI",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    * {font-family: 'Inter', sans-serif;}
    .stApp {background: linear-gradient(135deg, #1a0033 0%, #2d0052 25%, #1a1a4d 50%, #0d0d2d 75%, #1a0033 100%);}
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    [data-testid="stToolbar"] {display: none;}
    .stApp::before {
        content: '';
        position: fixed; top: 0; left: 0; right: 0; bottom: 0;
        background:
            radial-gradient(600px at 20% 30%, rgba(255, 105, 180, 0.15), transparent),
            radial-gradient(800px at 80% 60%, rgba(138, 43, 226, 0.15), transparent);
        pointer-events: none; z-index: -1;
    }
    [data-testid="stSidebar"] {background: linear-gradient(135deg, rgba(255, 105, 180, 0.13), rgba(138, 43, 226, 0.12)) !important; backdrop-filter: blur(20px) !important;}
    .status-item {background: linear-gradient(135deg, rgba(255, 105, 180, 0.04), rgba(138, 43, 226, 0.06)); border: 2px solid rgba(255, 105, 180, 0.17); border-radius: 12px; padding: 1rem 1.2rem; margin: 1rem 0; transition: all 0.3s;}
    .status-item.active {border-color: rgba(251, 191, 36, 0.8); background: linear-gradient(120deg, rgba(251, 191, 36, 0.13), rgba(251, 191, 36, 0.06)); box-shadow: 0 0 20px rgba(251, 191, 36, 0.22); animation: pulse-border 1.4s infinite;}
    .status-item.complete {border-color: rgba(34, 197, 94, 0.55); background: linear-gradient(120deg, rgba(34, 197, 94, 0.09), rgba(34, 197, 94, 0.05));}
    @keyframes pulse-border {0%,100%{box-shadow:0 0 20px rgba(251,191,36,0.22);}50%{box-shadow:0 0 32px rgba(251,191,36,0.29);}}
    .status-label {font-size: 1.12rem;font-weight: 700;color: white;}
    .status-desc {font-size: .88rem;color: rgba(255,255,255,0.60);}
    .status-icon {font-size: 1.8rem; float: right; margin-top: -2.3rem;}
    .header-section {background: linear-gradient(135deg, rgba(255, 105, 180, 0.07), rgba(138, 43, 226, 0.07));backdrop-filter: blur(20px); border: 2px solid rgba(255, 105, 180, 0.13); border-radius: 24px; padding: 2.1rem 1rem 2.7rem 1rem;margin-bottom:1.7rem;text-align:center;}
    .main-title {font-size:3rem;font-weight:800;background:linear-gradient(135deg,#ff69b4,#ff1493,#da70d6,#ff69b4);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0}
    .subtitle{font-size:1.1rem;color:rgba(255,255,255,0.8);margin-top:0.5rem}
    .glass-card{background:linear-gradient(135deg,rgba(255,105,180,0.08),rgba(138,43,226,0.09));backdrop-filter:blur(18px);border:1.5px solid rgba(255,105,180,0.20);border-radius:20px;padding:2rem;margin-bottom:1.1rem;box-shadow:0 7px 27px rgba(255,105,180,0.15);}
    .section-title{font-size: 1.52rem;font-weight:700;background:linear-gradient(135deg,#ff69b4,#da70d6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;padding-bottom:.2rem;}
    .step-card{background:linear-gradient(135deg,rgba(255,105,180,0.05),rgba(138,43,226,0.05));border:2px solid rgba(255,105,180,0.13);border-radius:14px;padding:1.3rem;margin:1rem 0;}
    .step-card.complete{border-color:rgba(34,197,94,0.5);background:linear-gradient(135deg,rgba(34,197,94,0.10),rgba(34,197,94,0.06));}
    .step-title{font-size:1.14rem;font-weight:600;color:#fff;}
    .step-description{color:rgba(255,255,255,0.79);font-size:.95rem;margin-top:.22rem;}
    .badge{display:inline-block;padding:.38rem 1rem;border-radius:50px;font-size:.9rem;font-weight:600;background:linear-gradient(135deg,rgba(255,105,180,0.2),rgba(138,43,226,0.15));border:1px solid rgba(255,105,180,0.34);color:#ff69b4;}
    .badge.complete{background:linear-gradient(135deg,rgba(34,197,94,0.15),rgba(34,197,94,0.16));border-color:rgba(34,197,94,0.35);color:#86efac;}
    .stButton>button{background:linear-gradient(135deg,#ff69b4,#da70d6)!important;color:white!important;border:none!important;border-radius:10px!important;padding:1rem 1.6rem!important;font-weight:600!important;font-size:1.01rem!important;box-shadow:0 6px 18px rgba(255,105,180,.16)!important;text-transform:uppercase;letter-spacing:.5px;}
    .stButton>button:hover{background:linear-gradient(135deg,#ff1493,#da70d6)!important;box-shadow:0 11px 31px rgba(255,105,180,.23)!important;transform:translateY(-2px);}
    .image-frame{border:2px solid rgba(255,105,180,0.22);border-radius:17px;overflow:hidden;background:rgba(0,0,0,0.25);box-shadow:0 6px 23px rgba(255,105,180,0.14);}
</style>
""", unsafe_allow_html=True)

# -- Status session state
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'status' not in st.session_state:
    st.session_state.status = {'enhancement': 'pending', 'detection': 'pending', 'texture': 'pending', 'restoration': 'pending'}
if 'run_all' not in st.session_state:
    st.session_state.run_all = False

enhancer, crack_detector, texture_classifier, inpainting_gan = load_models()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 🚦 Processing Status")
    steps = [
        ("enhancement", "Enhancement", "Adjust overall quality"),
        ("detection", "Damage Detection", "Locate cracks & flaws"),
        ("texture", "Texture Analysis", "Classify surface"),
        ("restoration", "Restoration", "Repair image")
    ]
    for key, title, desc in steps:
        status = st.session_state.status[key]
        classnm = "active" if status == 'active' else "complete" if status == 'complete' else ""
        icon = "✓" if status == 'complete' else "○"
        st.markdown(f"""
        <div class="status-item {classnm}">
            <div class="status-icon">{icon}</div>
            <div class="status-label">{title}</div>
            <div class="status-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    if st.button("🔁 Reset All"):
        st.session_state.results = {}
        st.session_state.status = {k:'pending' for k in st.session_state.status}
        st.session_state.run_all = False
        st.rerun()

# -- HEADER --
st.markdown("""
<div class="header-section">
    <h1 class="main-title">Heritage Restoration</h1>
    <p class="subtitle">AI-Powered Cultural Heritage Restoration Pipeline</p>
</div>
""", unsafe_allow_html=True)

# --- UPLOAD ---
col_upload, col_info = st.columns([2, 1])
with col_upload:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Upload Image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload your artifact image", type=['jpg','jpeg','png'],
        label_visibility="collapsed"
    )
    if uploaded_file:
        img = Image.open(uploaded_file)
        img_arr = np.array(img)
        st.session_state.original_image = img_arr
        st.markdown('<div class="image-frame">', unsafe_allow_html=True)
        st.image(img, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_info:
    if uploaded_file:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title" style="font-size:1.2rem;">Info</div>', unsafe_allow_html=True)
        h, w = img_arr.shape[:2]
        st.markdown(f"**Resolution:** {w}×{h}")
        st.markdown('</div>', unsafe_allow_html=True)

# Pipeline logic
def run_full_pipeline():
    st.session_state.status = {k: 'pending' for k in st.session_state.status}

    st.session_state.status['enhancement'] = 'active'
    enhanced = enhancer.enhance_pipeline(st.session_state.original_image)
    st.session_state.results['enhanced'] = enhanced
    st.session_state.status['enhancement'] = 'complete'

    st.session_state.status['detection'] = 'active'
    cracks = crack_detector.predict(enhanced)
    st.session_state.results['cracks'] = cracks
    st.session_state.status['detection'] = 'complete'

    st.session_state.status['texture'] = 'active'
    st.session_state.results['texture'] = texture_classifier.predict(enhanced)
    st.session_state.status['texture'] = 'complete'

    st.session_state.status['restoration'] = 'active'
    base = enhanced
    mask = cracks
    mask_bin = cv2.threshold(mask, 80, 255, cv2.THRESH_BINARY)[1]
    mask_thin = cv2.dilate(mask_bin, np.ones((2,2), np.uint8), iterations=1)
    mask_thin = cv2.erode(mask_thin, np.ones((3,3), np.uint8), iterations=2)
    if np.sum(mask_thin) > 0:
        try:
            import cv2.ximgproc
            mask_thin = cv2.ximgproc.thinning(mask_thin)
        except Exception:
            pass
        restored = inpainting_gan.inpaint(base, mask_thin)
        out = base.copy()
        for c in range(3):
            out[:,:,c] = np.where(mask_thin==0, base[:,:,c], restored[:,:,c])
        st.session_state.results['restored'] = out
    else:
        st.session_state.results['restored'] = base
    st.session_state.status['restoration'] = 'complete'

    st.session_state.run_all = False
    st.rerun()

if uploaded_file:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Pipeline</div>', unsafe_allow_html=True)

    runall = st.button("🚀 Run Full Restoration Pipeline", type="primary")
    if runall:
        st.session_state.run_all = True
        run_full_pipeline()

    # Step 1: Enhancement
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"""
        <div class="step-card {'complete' if 'enhanced' in st.session_state.results else ''}">
        <div class="step-title">Enhancement</div>
        <div class="step-description">Brightness, contrast, sharpness</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        if st.button("Run", key="enh", use_container_width=True):
            st.session_state.status['enhancement'] = 'active'
            st.session_state.results['enhanced'] = enhancer.enhance_pipeline(st.session_state.original_image)
            st.session_state.status['enhancement'] = 'complete'
            st.rerun()
    if 'enhanced' in st.session_state.results:
        c1, c2 = st.columns(2)
        with c1: st.image(st.session_state.original_image, width="stretch")
        with c2: st.image(st.session_state.results['enhanced'], width="stretch")
    st.markdown("")

    # Step 2: Crack Detection
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"""
        <div class="step-card {'complete' if 'cracks' in st.session_state.results else ''}">
        <div class="step-title">Damage Detection</div>
        <div class="step-description">Find cracks and flaws</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        if st.button("Run", key="crack", use_container_width=True):
            st.session_state.status['detection'] = 'active'
            inp = st.session_state.results.get('enhanced', st.session_state.original_image)
            st.session_state.results['cracks'] = crack_detector.predict(inp)
            st.session_state.status['detection'] = 'complete'
            st.rerun()
    if 'cracks' in st.session_state.results:
        vis = cv2.applyColorMap(st.session_state.results['cracks'], cv2.COLORMAP_INFERNO)
        vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        st.image(vis, width="stretch")
    st.markdown("")

    # Step 3: Texture
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"""
        <div class="step-card {'complete' if 'texture' in st.session_state.results else ''}">
        <div class="step-title">Texture Analysis</div>
        <div class="step-description">Material classification</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        if st.button("Run", key="tex", use_container_width=True):
            st.session_state.status['texture'] = 'active'
            inp = st.session_state.results.get('enhanced', st.session_state.original_image)
            st.session_state.results['texture'] = texture_classifier.predict(inp)
            st.session_state.status['texture'] = 'complete'
            st.rerun()
    if 'texture' in st.session_state.results:
        c1, c2 = st.columns(2)
        with c1: st.metric("Type", st.session_state.results['texture']['class'].title())
        with c2: st.metric("Confidence", f"{st.session_state.results['texture']['confidence']:.0%}")
    st.markdown("")

    # Step 4: Restoration (only restore crack pixels)
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"""
        <div class="step-card {'complete' if 'restored' in st.session_state.results else ''}">
        <div class="step-title">Restoration</div>
        <div class="step-description">Repair cracked regions</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        if st.button("Run", key="rest", use_container_width=True):
            st.session_state.status['restoration'] = 'active'
            base = st.session_state.results.get('enhanced', st.session_state.original_image)
            mask = st.session_state.results.get('cracks', np.zeros(base.shape[:2], dtype=np.uint8))
            mask_bin = cv2.threshold(mask, 80, 255, cv2.THRESH_BINARY)[1]
            mask_thin = cv2.dilate(mask_bin, np.ones((2,2), np.uint8), iterations=1)
            mask_thin = cv2.erode(mask_thin, np.ones((3,3), np.uint8), iterations=2)
            if np.sum(mask_thin) > 0:
                try:
                    import cv2.ximgproc
                    mask_thin = cv2.ximgproc.thinning(mask_thin)
                except Exception:
                    pass
                restored = inpainting_gan.inpaint(base, mask_thin)
                out = base.copy()
                for c in range(3):
                    out[:,:,c] = np.where(mask_thin==0, base[:,:,c], restored[:,:,c])
                st.session_state.results['restored'] = out
            else:
                st.session_state.results['restored'] = base
            st.session_state.status['restoration'] = 'complete'
            st.rerun()
    if 'restored' in st.session_state.results:
        st.image(st.session_state.results['restored'], width="stretch")
        pil = Image.fromarray(st.session_state.results['restored'])
        buf = BytesIO()
        pil.save(buf, format="PNG")
        st.download_button("Download", buf.getvalue(), "restored.png", "image/png")
    st.markdown('</div>', unsafe_allow_html=True)

    # Comparison
    if 'restored' in st.session_state.results:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Comparison</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: st.image(st.session_state.original_image, width="stretch"); st.caption("Original")
        with c2: st.image(st.session_state.results['enhanced'], width="stretch"); st.caption("Enhanced")
        with c3: st.image(st.session_state.results['restored'], width="stretch"); st.caption("Restored")
        st.markdown('</div>', unsafe_allow_html=True)
