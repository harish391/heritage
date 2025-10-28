"""
Heritage Restoration AI - Professional Web Interface
Advanced Cultural Heritage Preservation System
"""
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

# Page Configuration
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
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #7e22ce 100%);
    }
    
    /* Hide all Streamlit elements */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    [data-testid="stToolbar"] {display: none;}
    
    /* Hide deprecation warnings */
    .stAlert {display: none;}
    
    /* Header */
    .main-header {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.5px;
    }
    
    .main-subtitle {
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.7);
        font-weight: 400;
    }
    
    /* Section Cards */
    .section-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 12px;
        padding: 1.8rem;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .section-card:hover {
        background: rgba(255, 255, 255, 0.12);
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.15);
    }
    
    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 1rem;
        padding-bottom: 0.8rem;
        border-bottom: 2px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Step Cards */
    .step-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .step-card.enabled {
        border-left: 4px solid #3b82f6;
    }
    
    .step-card.processing {
        border-left: 4px solid #fbbf24;
        background: linear-gradient(135deg, rgba(251,191,36,0.15), rgba(251,191,36,0.05));
    }
    
    .step-card.complete {
        border-left: 4px solid #10b981;
        background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(16,185,129,0.05));
    }
    
    .step-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.8rem;
    }
    
    .step-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #ffffff;
    }
    
    .step-description {
        color: rgba(255, 255, 255, 0.75);
        font-size: 0.95rem;
        line-height: 1.6;
        margin-bottom: 1rem;
    }
    
    .step-status {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .status-ready {
        background: rgba(59, 130, 246, 0.2);
        color: #93c5fd;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    
    .status-processing {
        background: rgba(251, 191, 36, 0.2);
        color: #fde68a;
        border: 1px solid rgba(251, 191, 36, 0.3);
    }
    
    .status-complete {
        background: rgba(16, 185, 129, 0.2);
        color: #6ee7b7;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    /* Metrics */
    .metric-box {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 1.2rem;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #60a5fa;
        margin-bottom: 0.3rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.6);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Buttons */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.7rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4) !important;
    }
    
    .stButton > button:disabled {
        background: rgba(255, 255, 255, 0.1) !important;
        color: rgba(255, 255, 255, 0.3) !important;
        cursor: not-allowed !important;
    }
    
    /* Image Container */
    .image-frame {
        border: 2px solid rgba(255, 255, 255, 0.15);
        border-radius: 10px;
        overflow: hidden;
        background: rgba(0, 0, 0, 0.2);
    }
    
    /* Download Button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981, #059669) !important;
        width: 100%;
    }
    
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #059669, #047857) !important;
    }
    
    /* Hide file uploader details */
    .uploadedFile {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if 'step_status' not in st.session_state:
    st.session_state.step_status = {
        'enhancement': 'ready',
        'crack_detection': 'ready',
        'texture_analysis': 'ready',
        'restoration': 'ready'
    }
    st.session_state.results = {}
    st.session_state.models_loaded = False

# Header
st.markdown("""
<div class="main-header">
    <h1 class="main-title">Heritage Restoration AI</h1>
    <p class="main-subtitle">Professional Cultural Heritage Preservation System</p>
</div>
""", unsafe_allow_html=True)

# Sidebar - Controls
with st.sidebar:
    st.markdown("### System Controls")
    
    if st.button("Reset All Steps"):
        st.session_state.step_status = {
            'enhancement': 'ready',
            'crack_detection': 'ready',
            'texture_analysis': 'ready',
            'restoration': 'ready'
        }
        st.session_state.results = {}
        st.rerun()
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This system uses advanced AI algorithms for:
    - Image quality enhancement
    - Damage detection
    - Material analysis
    - Intelligent restoration
    """)
    
    st.markdown("---")
    st.markdown("**Version:** 2.0  \n**Status:** Active")

# Main Content
col_upload, col_info = st.columns([2, 1])

with col_upload:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Image Upload</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Select heritage artifact image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a high-resolution image for best results",
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        st.markdown('<div class="image-frame">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
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
    
    # Load models
    if not st.session_state.models_loaded:
        with st.spinner("Loading AI models..."):
            st.session_state.enhancer = ImageEnhancer()
            st.session_state.crack_detector = CrackSegmentationModel()
            st.session_state.texture_classifier = TextureClassifier()
            st.session_state.inpainting_gan = InpaintingGAN()
            st.session_state.models_loaded = True
    
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Processing Pipeline</div>', unsafe_allow_html=True)
    
    # Step 1: Enhancement
    st.markdown(f"""
    <div class="step-card {st.session_state.step_status['enhancement']}">
        <div class="step-header">
            <div class="step-title">Step 1: Image Enhancement</div>
            <span class="step-status status-{st.session_state.step_status['enhancement']}">
                {st.session_state.step_status['enhancement'].title()}
            </span>
        </div>
        <div class="step-description">
            Applies CLAHE contrast enhancement, bilateral noise filtering, and unsharp masking 
            to improve overall image quality and reveal hidden details.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Run Enhancement", key="btn_enhance"):
            st.session_state.step_status['enhancement'] = 'processing'
            st.rerun()
    
    if st.session_state.step_status['enhancement'] == 'processing':
        with st.spinner("Processing enhancement..."):
            st.session_state.results['enhanced'] = st.session_state.enhancer.enhance_pipeline(img_array)
            st.session_state.step_status['enhancement'] = 'complete'
            time.sleep(0.5)
            st.rerun()
    
    if 'enhanced' in st.session_state.results:
        st.markdown("**Enhanced Result:**")
        st.image(st.session_state.results['enhanced'], use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Step 2: Crack Detection
    st.markdown(f"""
    <div class="step-card {st.session_state.step_status['crack_detection']}">
        <div class="step-header">
            <div class="step-title">Step 2: Damage Detection</div>
            <span class="step-status status-{st.session_state.step_status['crack_detection']}">
                {st.session_state.step_status['crack_detection'].title()}
            </span>
        </div>
        <div class="step-description">
            Employs U-Net architecture for precise segmentation of cracks, fractures, and 
            structural damage across the artifact surface.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col2:
        can_run_crack = 'enhanced' in st.session_state.results
        if st.button("Run Detection", key="btn_crack", disabled=not can_run_crack):
            st.session_state.step_status['crack_detection'] = 'processing'
            st.rerun()
    
    if st.session_state.step_status['crack_detection'] == 'processing':
        with st.spinner("Detecting damage..."):
            enhanced_img = st.session_state.results['enhanced']
            st.session_state.results['cracks'] = st.session_state.crack_detector.predict(enhanced_img)
            st.session_state.step_status['crack_detection'] = 'complete'
            time.sleep(0.5)
            st.rerun()
    
    if 'cracks' in st.session_state.results:
        crack_vis = cv2.applyColorMap(st.session_state.results['cracks'], cv2.COLORMAP_INFERNO)
        crack_vis = cv2.cvtColor(crack_vis, cv2.COLOR_BGR2RGB)
        st.markdown("**Damage Heatmap:**")
        st.image(crack_vis, use_container_width=True)
        num_cracks = np.count_nonzero(st.session_state.results['cracks'] > 50)
        st.info(f"Detected {num_cracks:,} damaged pixels")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Step 3: Texture Analysis
    st.markdown(f"""
    <div class="step-card {st.session_state.step_status['texture_analysis']}">
        <div class="step-header">
            <div class="step-title">Step 3: Material Analysis</div>
            <span class="step-status status-{st.session_state.step_status['texture_analysis']}">
                {st.session_state.step_status['texture_analysis'].title()}
            </span>
        </div>
        <div class="step-description">
            Uses ResNet50 transfer learning to classify surface textures and material composition 
            for informed restoration decisions.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col2:
        can_run_texture = 'enhanced' in st.session_state.results
        if st.button("Run Analysis", key="btn_texture", disabled=not can_run_texture):
            st.session_state.step_status['texture_analysis'] = 'processing'
            st.rerun()
    
    if st.session_state.step_status['texture_analysis'] == 'processing':
        with st.spinner("Analyzing materials..."):
            enhanced_img = st.session_state.results['enhanced']
            st.session_state.results['texture'] = st.session_state.texture_classifier.predict(enhanced_img)
            st.session_state.step_status['texture_analysis'] = 'complete'
            time.sleep(0.5)
            st.rerun()
    
    if 'texture' in st.session_state.results:
        result = st.session_state.results['texture']
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Material Type", result['class'].title())
        with col2:
            st.metric("Confidence", f"{result['confidence']:.1%}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Step 4: Restoration
    st.markdown(f"""
    <div class="step-card {st.session_state.step_status['restoration']}">
        <div class="step-header">
            <div class="step-title">Step 4: Intelligent Restoration</div>
            <span class="step-status status-{st.session_state.step_status['restoration']}">
                {st.session_state.step_status['restoration'].title()}
            </span>
        </div>
        <div class="step-description">
            Applies advanced inpainting algorithms to reconstruct damaged areas while 
            preserving authentic textures and historical integrity.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col2:
        can_run_restore = 'enhanced' in st.session_state.results and 'cracks' in st.session_state.results
        if st.button("Run Restoration", key="btn_restore", disabled=not can_run_restore):
            st.session_state.step_status['restoration'] = 'processing'
            st.rerun()
    
    if st.session_state.step_status['restoration'] == 'processing':
        with st.spinner("Restoring artifact..."):
            st.session_state.results['restored'] = st.session_state.inpainting_gan.inpaint(
                st.session_state.results['enhanced'],
                st.session_state.results['cracks']
            )
            st.session_state.step_status['restoration'] = 'complete'
            time.sleep(0.5)
            st.rerun()
    
    if 'restored' in st.session_state.results:
        st.markdown("**Restored Result:**")
        st.image(st.session_state.results['restored'], use_container_width=True)
        
        # Download
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
    
    # Final Comparison
    if 'restored' in st.session_state.results:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Comparison View</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original**")
            st.image(img_array, use_container_width=True)
        with col2:
            st.markdown("**Restored**")
            st.image(st.session_state.results['restored'], use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
