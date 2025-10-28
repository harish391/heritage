"""
🏛️ Heritage Restoration AI - Ultra Modern Web UI
Glassmorphic design with smooth animations and step-by-step visualization
"""
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import time
import base64
from io import BytesIO

from src.image_enhancement import ImageEnhancer
from src.crack_segmentation import CrackSegmentationModel
from src.texture_classification import TextureClassifier
from src.inpainting import InpaintingGAN

# Page config
st.set_page_config(
    page_title="🏛️ Heritage AI",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern CSS with glassmorphism
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Poppins', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom Header */
    .hero-section {
        text-align: center;
        padding: 3rem 1rem;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 30px;
        margin: 2rem auto;
        max-width: 900px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(45deg, #fff, #00f2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        animation: fadeInDown 0.8s ease-out;
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        color: rgba(255, 255, 255, 0.9);
        font-weight: 300;
        animation: fadeInUp 0.8s ease-out;
    }
    
    /* Glass Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 25px;
        padding: 2rem;
        margin: 1.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        animation: fadeIn 0.6s ease-out;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Upload Zone */
    .upload-zone {
        background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
        backdrop-filter: blur(10px);
        border: 3px dashed rgba(255, 255, 255, 0.3);
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .upload-zone:hover {
        border-color: #00f2fe;
        background: rgba(0, 242, 254, 0.1);
        transform: scale(1.02);
    }
    
    /* Step Cards */
    .step-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.15), rgba(255,255,255,0.05));
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .step-card.active {
        border-left-color: #00f2fe;
        background: linear-gradient(135deg, rgba(0,242,254,0.2), rgba(0,242,254,0.05));
        animation: pulse 2s infinite;
    }
    
    .step-card.complete {
        border-left-color: #4ade80;
        background: linear-gradient(135deg, rgba(74,222,128,0.2), rgba(74,222,128,0.05));
    }
    
    .step-card.pending {
        border-left-color: rgba(255,255,255,0.3);
        opacity: 0.6;
    }
    
    /* Step Title */
    .step-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: white;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }
    
    .step-icon {
        font-size: 1.8rem;
        filter: drop-shadow(0 0 10px rgba(255,255,255,0.5));
    }
    
    .step-description {
        color: rgba(255, 255, 255, 0.8);
        font-size: 1rem;
        line-height: 1.6;
        margin-bottom: 1rem;
    }
    
    /* Progress Bar */
    .progress-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 50px;
        height: 10px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #00f2fe, #4ade80);
        border-radius: 50px;
        transition: width 0.5s ease;
        box-shadow: 0 0 20px rgba(0,242,254,0.5);
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 50px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0.5rem 0;
    }
    
    .status-processing {
        background: linear-gradient(135deg, #00f2fe, #0080ff);
        color: white;
        animation: pulse 1.5s infinite;
    }
    
    .status-complete {
        background: linear-gradient(135deg, #4ade80, #22c55e);
        color: white;
    }
    
    .status-pending {
        background: rgba(255, 255, 255, 0.2);
        color: rgba(255, 255, 255, 0.7);
    }
    
    /* Metrics */
    .metric-card {
        background: linear-gradient(135deg, rgba(0,242,254,0.15), rgba(0,242,254,0.05));
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(0,242,254,0.3);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(0,242,254,0.3);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #00f2fe;
        text-shadow: 0 0 20px rgba(0,242,254,0.5);
    }
    
    .metric-label {
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.8);
        margin-top: 0.5rem;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, #00f2fe, #4ade80) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 0.8rem 2.5rem !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(0,242,254,0.4) !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 30px rgba(0,242,254,0.6) !important;
    }
    
    /* Image containers */
    .image-container {
        border-radius: 20px;
        overflow: hidden;
        border: 2px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    .image-container:hover {
        transform: scale(1.02);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #ff006e, #ff4d8d) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 0.8rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
</style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero-section">
    <div class="hero-title">🏛️ Heritage AI Restoration</div>
    <div class="hero-subtitle">
        Advanced AI-Powered Cultural Heritage Preservation<br>
        ✨ Modern • Interactive • Beautiful Visualizations
    </div>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
    st.session_state.step = 0
    st.session_state.results = {}
    st.session_state.models_loaded = False

# Upload Section
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("### 📤 Upload Your Heritage Artifact")

uploaded_file = st.file_uploader(
    "Drag and drop or click to browse",
    type=['jpg', 'jpeg', 'png'],
    help="Upload a high-quality image of your heritage artifact"
)

if uploaded_file:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(image, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 📊 Image Information")
        
        # Metrics
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        
        with metrics_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{img_array.shape[1]}</div>
                <div class="metric-label">Width (px)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metrics_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{img_array.shape[0]}</div>
                <div class="metric-label">Height (px)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metrics_col3:
            size_mb = uploaded_file.size / (1024 * 1024)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{size_mb:.2f}</div>
                <div class="metric-label">Size (MB)</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("⚡ START RESTORATION PIPELINE"):
            st.session_state.processed = True
            st.session_state.step = 0
            st.session_state.results = {}
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Processing Pipeline
if uploaded_file and st.session_state.processed:
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🔄 AI Processing Pipeline")
    
    # Initialize models once
    if not st.session_state.models_loaded:
        with st.spinner("🤖 Loading AI models..."):
            st.session_state.enhancer = ImageEnhancer()
            st.session_state.crack_detector = CrackSegmentationModel()
            st.session_state.texture_classifier = TextureClassifier()
            st.session_state.inpainting_gan = InpaintingGAN()
            st.session_state.models_loaded = True
            time.sleep(1)
    
    # Steps definition
    steps = [
        {
            "icon": "🎨",
            "title": "Image Enhancement",
            "description": "Applying CLAHE contrast enhancement, bilateral filtering for noise reduction, and unsharp masking for edge sharpening to improve overall image quality."
        },
        {
            "icon": "🔍",
            "title": "Crack Detection",
            "description": "Using advanced U-Net based edge detection algorithm to identify and segment damaged regions, structural cracks, and areas requiring restoration."
        },
        {
            "icon": "🧱",
            "title": "Texture Analysis",
            "description": "Classifying surface texture patterns and material types using deep learning with ResNet50 architecture for accurate restoration planning."
        },
        {
            "icon": "✨",
            "title": "Restoration & Inpainting",
            "description": "Intelligently reconstructing damaged regions using OpenCV Telea inpainting algorithm for seamless and authentic restoration results."
        }
    ]
    
    # Display all steps with status
    for idx, step_info in enumerate(steps):
        status = "complete" if idx < st.session_state.step else ("active" if idx == st.session_state.step else "pending")
        
        st.markdown(f"""
        <div class="step-card {status}">
            <div class="step-title">
                <span class="step-icon">{step_info['icon']}</span>
                <span>Step {idx + 1}: {step_info['title']}</span>
            </div>
            <div class="step-description">{step_info['description']}</div>
            <span class="status-badge status-{status}">
                {'✓ Complete' if status == 'complete' else ('⚡ Processing...' if status == 'active' else '○ Pending')}
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    # Process current step
    if st.session_state.step < len(steps):
        idx = st.session_state.step
        step_info = steps[idx]
        
        with st.spinner(f"⚡ Processing {step_info['title']}..."):
            time.sleep(0.8)  # Visual effect
            
            if idx == 0:  # Enhancement
                st.session_state.results['enhanced'] = st.session_state.enhancer.enhance_pipeline(img_array)
                st.markdown("#### ✨ Enhanced Result")
                st.image(st.session_state.results['enhanced'], use_column_width=True)
            
            elif idx == 1:  # Crack Detection
                if 'enhanced' in st.session_state.results:
                    st.session_state.results['cracks'] = st.session_state.crack_detector.predict(st.session_state.results['enhanced'])
                    crack_vis = cv2.applyColorMap(st.session_state.results['cracks'], cv2.COLORMAP_INFERNO)
                    crack_vis = cv2.cvtColor(crack_vis, cv2.COLOR_BGR2RGB)
                    st.markdown("#### 🔥 Crack Detection Heatmap")
                    st.image(crack_vis, use_column_width=True)
                    num_cracks = np.count_nonzero(st.session_state.results['cracks'] > 50)
                    st.metric("Detected Damage Pixels", f"{num_cracks:,}")
            
            elif idx == 2:  # Texture
                if 'enhanced' in st.session_state.results:
                    st.session_state.results['texture'] = st.session_state.texture_classifier.predict(st.session_state.results['enhanced'])
                    st.markdown("#### 🧱 Texture Analysis Results")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Texture Type", st.session_state.results['texture']['class'].title())
                    with col2:
                        st.metric("Confidence Score", f"{st.session_state.results['texture']['confidence']:.1%}")
            
            elif idx == 3:  # Restoration
                if 'enhanced' in st.session_state.results and 'cracks' in st.session_state.results:
                    st.session_state.results['restored'] = st.session_state.inpainting_gan.inpaint(
                        st.session_state.results['enhanced'], 
                        st.session_state.results['cracks']
                    )
                    st.markdown("#### ✨ Final Restored Result")
                    st.image(st.session_state.results['restored'], use_column_width=True)
                    st.balloons()
            
            st.session_state.step += 1
            time.sleep(0.5)
            st.rerun()
    
    # Progress bar
    progress = (st.session_state.step / len(steps)) * 100
    st.markdown(f"""
    <div class="progress-container">
        <div class="progress-bar" style="width: {progress}%"></div>
    </div>
    <p style="text-align: center; color: rgba(255,255,255,0.9); font-size: 1.2rem; font-weight: 600; margin-top: 1rem;">
        {progress:.0f}% Complete • {st.session_state.step}/{len(steps)} Steps Finished
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Final comparison
    if st.session_state.step >= len(steps) and 'restored' in st.session_state.results:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🎉 Restoration Complete!")
        
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("#### 📷 Original Image")
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(img_array, use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### ✨ Restored Image")
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(st.session_state.results['restored'], use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Action buttons
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            # Create download button for restored image
            restored_pil = Image.fromarray(st.session_state.results['restored'])
            buf = BytesIO()
            restored_pil.save(buf, format="PNG")
            
            st.download_button(
                label="💾 Download Restored Image",
                data=buf.getvalue(),
                file_name="restored_heritage.png",
                mime="image/png"
            )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Reset button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🔄 Process Another Image"):
                st.session_state.processed = False
                st.session_state.step = 0
                st.session_state.results = {}
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="upload-zone">
        <h2 style="color: white; margin-bottom: 1rem;">
            📤 Drop Your Heritage Image Here
        </h2>
        <p style="color: rgba(255,255,255,0.8); font-size: 1.2rem; line-height: 1.8;">
            Upload a heritage artifact image to begin the AI-powered restoration process<br>
            <span style="font-size: 0.9rem; color: rgba(255,255,255,0.6);">
                Supported formats: JPG, JPEG, PNG
            </span>
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; color: rgba(255,255,255,0.5); padding: 2rem; font-size: 0.9rem;">
    Made with ❤️ using AI • Heritage Restoration System v2.0
</div>
""", unsafe_allow_html=True)
