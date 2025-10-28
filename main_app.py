import streamlit as st
import numpy as np
import cv2
from PIL import Image
import time
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

from src.image_enhancement import ImageEnhancer
from src.crack_segmentation import CrackSegmentationModel
from src.texture_classification import TextureClassifier
from src.inpainting import InpaintingGAN

st.set_page_config(
    page_title="Heritage Restoration AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * {font-family: 'Inter', sans-serif;}
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #7e22ce 100%);
    }
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    [data-testid="stToolbar"] {display: none;}
    .main-header {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-title {font-size: 2.8rem; font-weight: 700; color: #fff;}
    .main-subtitle {font-size: 1.1rem; color: rgba(255,255,255,0.7);}
    .section-card {
        background: rgba(255,255,255,0.08);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 12px;
        padding: 1.8rem;
        margin-bottom: 1.5rem;
    }
    .section-title {font-size: 1.3rem; font-weight: 600; color: #fff; margin-bottom: 1rem;}
    .step-card {
        background: linear-gradient(135deg,rgba(255,255,255,0.10),rgba(255,255,255,0.03));
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #3b82f6;
    }
    .step-card.complete {border-left: 4px solid #10b981;}
    .step-header {display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;}
    .step-title {font-size: 1.15rem; font-weight: 600; color: #fff;}
    .step-description {color: rgba(255,255,255,0.75); font-size: 0.95rem;}
    .step-status {display: inline-block; padding: 0.35rem 0.9rem; border-radius: 20px; font-size: 0.85rem; font-weight: 600;}
    .status-ready {background: rgba(59,130,246,0.2); color: #93c5fd; border: 1px solid rgba(59,130,246,0.3);}
    .status-complete {background: rgba(16,185,129,0.2); color: #fff; border: 1px solid rgba(16,185,129,0.3);}
    .metric-box {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 8px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-value {font-size: 2rem; font-weight: 700; color: #60a5fa; margin-bottom: 0.3rem;}
    .metric-label {font-size: 0.9rem; color: rgba(255,255,255,0.6);}
    .stButton > button {
        width:100%;
        background: linear-gradient(135deg,#3b82f6,#8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.7rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    .image-frame {
        border: 2px solid rgba(255,255,255,0.15);
        border-radius: 10px;
        overflow: hidden;
        background: rgba(0,0,0,0.2);
    }
    .stDownloadButton > button {
        background: linear-gradient(135deg,#10b981,#059669) !important;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

if 'results' not in st.session_state:
    st.session_state.results = {}
    st.session_state.processing = {}
    st.session_state.models_loaded = False

# Header
st.markdown("""
<div class="main-header">
    <h1 class="main-title">Heritage Restoration AI</h1>
    <p class="main-subtitle">Independent Step Execution • Any Order • With VISIBLE Results</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### System Controls")
    if st.button("Clear All Results"):
        st.session_state.results = {}
        st.session_state.processing = {}
        st.rerun()
    st.markdown("---")
    st.markdown("### Execution Status")
    status_items = [
        ("Enhancement", 'enhanced'),
        ("Crack Detection", 'cracks'),
        ("Texture Analysis", 'texture'),
        ("Restoration", 'restored')
    ]
    for label, key in status_items:
        if key in st.session_state.results:
            st.markdown(f"✅ {label}")
        else:
            st.markdown(f"⚪ {label}")
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    - Execute any processing step independently
    - No execution order required
    - Run steps multiple times
    - Mix and match as needed
    """)

col_upload, col_info = st.columns([2, 1])
with col_upload:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Image Upload</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Select heritage artifact image", type=['jpg', 'jpeg', 'png'],
        help="Upload a high-resolution image", label_visibility="collapsed"
    )
    if uploaded_file:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        st.session_state.original_image = img_array
        st.markdown('<div class="image-frame">', unsafe_allow_html=True)
        st.image(image, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_info:
    if uploaded_file:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Image Properties</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{img_array.shape[1]} × {img_array.shape[0]}</div>
            <div class="metric-label">Resolution</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        size_mb = uploaded_file.size / (1024 * 1024)
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{size_mb:.2f} MB</div>
            <div class="metric-label">File Size</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Processing Pipeline
if uploaded_file:
    if not st.session_state.models_loaded:
        with st.spinner("Loading AI models..."):
            st.session_state.enhancer = ImageEnhancer()
            st.session_state.crack_detector = CrackSegmentationModel()
            st.session_state.texture_classifier = TextureClassifier()
            st.session_state.inpainting_gan = InpaintingGAN()
            st.session_state.models_loaded = True

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Processing Pipeline - Independent Execution</div>', unsafe_allow_html=True)
    
    # Enhancement
    status_class = 'complete' if 'enhanced' in st.session_state.results else ''
    st.markdown(f"""
    <div class="step-card {status_class}">
        <div class="step-header">
            <div class="step-title">Step 1: Image Enhancement</div>
            <span class="step-status status-{'complete' if 'enhanced' in st.session_state.results else 'ready'}">
                {'Complete' if 'enhanced' in st.session_state.results else 'Ready'}
            </span>
        </div>
        <div class="step-description">
            Brightness boost, contrast enhancement, color saturation, and sharpening.
        </div>
    </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Execute", key="btn_enhance"):
            with st.spinner("Enhancing..."):
                st.session_state.results['enhanced'] = st.session_state.enhancer.enhance_pipeline(
                    st.session_state.original_image
                )
                time.sleep(0.3)
                st.rerun()
    if 'enhanced' in st.session_state.results:
        st.markdown("**Enhanced Result:**")
        st.image(st.session_state.results['enhanced'], use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Crack Detection
    status_class = 'complete' if 'cracks' in st.session_state.results else ''
    st.markdown(f"""
    <div class="step-card {status_class}">
        <div class="step-header">
            <div class="step-title">Step 2: Damage Detection</div>
            <span class="step-status status-{'complete' if 'cracks' in st.session_state.results else 'ready'}">
                {'Complete' if 'cracks' in st.session_state.results else 'Ready'}
            </span>
        </div>
        <div class="step-description">
            Multi-method edge detection to identify cracks, fractures, and damage.
        </div>
    </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    with col1:
        use_enhanced = st.checkbox("Use enhanced image if available", value=True, key="crack_use_enh")
    with col2:
        if st.button("Execute", key="btn_crack"):
            with st.spinner("Detecting damage..."):
                if use_enhanced and 'enhanced' in st.session_state.results:
                    input_img = st.session_state.results['enhanced']
                else:
                    input_img = st.session_state.original_image
                st.session_state.results['cracks'] = st.session_state.crack_detector.predict(input_img)
                time.sleep(0.3)
                st.rerun()
    if 'cracks' in st.session_state.results:
        st.markdown("**Damage Heatmap:**")
        crack_vis = cv2.applyColorMap(st.session_state.results['cracks'], cv2.COLORMAP_INFERNO)
        crack_vis = cv2.cvtColor(crack_vis, cv2.COLOR_BGR2RGB)
        st.image(crack_vis, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Texture
    status_class = 'complete' if 'texture' in st.session_state.results else ''
    st.markdown(f"""
    <div class="step-card {status_class}">
        <div class="step-header">
            <div class="step-title">Step 3: Material Analysis</div>
            <span class="step-status status-{'complete' if 'texture' in st.session_state.results else 'ready'}">
                {'Complete' if 'texture' in st.session_state.results else 'Ready'}
            </span>
        </div>
        <div class="step-description">
            Real-time texture classification based on image features.
        </div>
    </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    with col1:
        use_enhanced_tex = st.checkbox("Use enhanced image if available", value=True, key="tex_use_enh")
    with col2:
        if st.button("Execute", key="btn_texture"):
            with st.spinner("Analyzing..."):
                if use_enhanced_tex and 'enhanced' in st.session_state.results:
                    input_img = st.session_state.results['enhanced']
                else:
                    input_img = st.session_state.original_image
                st.session_state.results['texture'] = st.session_state.texture_classifier.predict(input_img)
                time.sleep(0.3)
                st.rerun()
    if 'texture' in st.session_state.results:
        result = st.session_state.results['texture']
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Material Type", result['class'].title())
        with col2:
            st.metric("Confidence", f"{result['confidence']:.1%}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Restoration
    status_class = 'complete' if 'restored' in st.session_state.results else ''
    st.markdown(f"""
    <div class="step-card {status_class}">
        <div class="step-header">
            <div class="step-title">Step 4: Intelligent Restoration</div>
            <span class="step-status status-{'complete' if 'restored' in st.session_state.results else 'ready'}">
                {'Complete' if 'restored' in st.session_state.results else 'Ready'}
            </span>
        </div>
        <div class="step-description">
            Inpainting and restoration of damaged regions with enhanced brightness.
        </div>
    </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    with col1:
        restore_use_enh = st.checkbox("Use enhanced for base", value=True, key="rest_use_enh")
        restore_use_crack = st.checkbox("Use crack mask if available", value=True, key="rest_use_crack")
    with col2:
        if st.button("Execute", key="btn_restore"):
            with st.spinner("Restoring..."):
                if restore_use_enh and 'enhanced' in st.session_state.results:
                    base_img = st.session_state.results['enhanced']
                else:
                    base_img = st.session_state.original_image

                if restore_use_crack and 'cracks' in st.session_state.results:
                    mask = st.session_state.results['cracks']
                else:
                    mask = np.zeros(base_img.shape[:2], dtype=np.uint8)

                st.session_state.results['restored'] = st.session_state.inpainting_gan.inpaint(base_img, mask)
                time.sleep(0.3)
                st.rerun()
    if 'restored' in st.session_state.results:
        st.markdown("**Restored Result:**")
        st.image(st.session_state.results['restored'], use_container_width=True)
        restored_pil = Image.fromarray(st.session_state.results['restored'])
        buf = BytesIO()
        restored_pil.save(buf, format="PNG")
        st.download_button(
            label="Download Restored Image",
            data=buf.getvalue(),
            file_name="restored_heritage.png",
            mime="image/png"
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # Full Comparison
    if 'enhanced' in st.session_state.results or 'restored' in st.session_state.results:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Full Comparison View</div>', unsafe_allow_html=True)
        if 'enhanced' in st.session_state.results and 'restored' in st.session_state.results:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Original**")
                st.image(st.session_state.original_image, use_container_width=True)
            with col2:
                st.markdown("**Enhanced**")
                st.image(st.session_state.results['enhanced'], use_container_width=True)
            with col3:
                st.markdown("**Restored**")
                st.image(st.session_state.results['restored'], use_container_width=True)
        elif 'enhanced' in st.session_state.results:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original**")
                st.image(st.session_state.original_image, use_container_width=True)
            with col2:
                st.markdown("**Enhanced**")
                st.image(st.session_state.results['enhanced'], use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- NEW FINAL PROCESSING STAGE CARD ---
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Step 5: Processing Stage (Final Output & Download)</div>', unsafe_allow_html=True)
    if 'restored' in st.session_state.results:
        st.markdown("**Final Restored Image (Download or Review):**")
        st.image(st.session_state.results['restored'], use_container_width=True)
        pil_final = Image.fromarray(st.session_state.results['restored'])
        buf_final = BytesIO()
        pil_final.save(buf_final, format="PNG")
        st.download_button(
            label="Download Final Restored Image",
            data=buf_final.getvalue(),
            file_name="final_restored_heritage.png",
            mime="image/png"
        )
    else:
        st.info("Process the restoration to get your final result ready for download.")
    st.markdown('</div>', unsafe_allow_html=True)
