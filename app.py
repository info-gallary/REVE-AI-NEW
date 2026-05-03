import streamlit as st
import requests
from PIL import Image
import re
import base64
from predict_d import predict_d
from predict_c import predict_c
from io import BytesIO
import subprocess
import os

# Page Configuration
st.set_page_config(
    page_title="REVE AI",
    page_icon="🧬",
    layout="wide"
)
def get_image_base64(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

image_base64 = get_image_base64("C:/Users/DELL/Downloads/reve.png")
# Custom CSS with Premium Animated Design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 25%, #16213e 50%, #0f3460 75%, #533483 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        font-family: 'Inter', sans-serif;
        min-height: 100vh;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Sidebar Styling */
    .css-1d391kg, .css-1cypcdb {
        background: rgba(10, 10, 10, 0.8);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(102, 126, 234, 0.3);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
    }
    
    /* Logo Animation */
    .logo-container {
        display: flex;
        justify-content: center;
        margin: 1rem 0;
        animation: logoFloat 3s ease-in-out infinite;
    }
    
    @keyframes logoFloat {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-10px) rotate(5deg); }
    }
    
    .custom-logo {
        width: 180px;
        height: 180px;
        background: linear-gradient(135deg, #667eea, #0f3460);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 3rem;
        color: white;
        box-shadow: 0 20px 40px rgba(15, 52, 96, 0.5);
        border: 3px solid rgba(102, 126, 234, 0.5);
        position: relative;
        overflow: hidden;
    }
    
    .custom-logo::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    /* Main Content Area */
    .main {
        background: rgba(10, 10, 10, 0.7);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(102, 126, 234, 0.3);
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.6);
    }
    
    /* Header Styling */
    .premium-header {
        text-align: center;
        margin-bottom: 3rem;
        padding: 2rem;
        background: linear-gradient(135deg, rgba(10,10,10,0.8), rgba(15,52,96,0.3));
        border-radius: 20px;
        border: 1px solid rgba(102,126,234,0.4);
        backdrop-filter: blur(15px);
        position: relative;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0,0,0,0.7);
    }
    
    .premium-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb, #f5576c, #4facfe);
        background-size: 300% 100%;
        animation: gradientMove 3s ease infinite;
    }
    
    @keyframes gradientMove {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea, #4facfe, #00d4ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 30px rgba(102, 126, 234, 0.8);
        animation: titleGlow 2s ease-in-out infinite alternate;
    }
    
    @keyframes titleGlow {
        from { filter: brightness(1); }
        to { filter: brightness(1.2); }
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.9);
        font-weight: 300;
        letter-spacing: 2px;
    }
    
    /* Upload Section */
    .upload-container {
        background: linear-gradient(135deg, rgba(10,10,10,0.9), rgba(15,52,96,0.4));
        border: 2px dashed rgba(102, 126, 234, 0.6);
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        margin: 2rem 0;
        backdrop-filter: blur(15px);
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
        box-shadow: 0 15px 35px rgba(0,0,0,0.5);
    }
    
    .upload-container:hover {
        border-color: rgba(102, 126, 234, 0.9);
        background: linear-gradient(135deg, rgba(10,10,10,0.95), rgba(15,52,96,0.6));
        transform: translateY(-5px);
        box-shadow: 0 25px 50px rgba(102, 126, 234, 0.3);
    }
    
    /* Professional Image Display Container */
    .image-display-container {
        background: linear-gradient(145deg, rgba(10,10,10,0.95), rgba(15,52,96,0.3));
        border: 1px solid rgba(102, 126, 234, 0.4);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        text-align: center;
        backdrop-filter: blur(20px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.6);
        position: relative;
        overflow: hidden;
    }
    
    .image-display-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #667eea, #4facfe, #00d4ff);
        background-size: 200% 100%;
        animation: gradientMove 3s ease infinite;
    }
    
    .image-frame {
        background: linear-gradient(145deg, rgba(0,0,0,0.8), rgba(15,52,96,0.2));
        border: 2px solid rgba(102, 126, 234, 0.5);
        border-radius: 15px;
        padding: 1rem;
        display: inline-block;
        box-shadow: 0 15px 30px rgba(0,0,0,0.7);
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    
    .image-frame:hover {
        transform: scale(1.02);
        border-color: rgba(102, 126, 234, 0.8);
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.2);
    }
    
    .image-frame::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 30%, rgba(102,126,234,0.1) 50%, transparent 70%);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .image-frame:hover::after {
        opacity: 1;
        animation: shimmer 1.5s ease-in-out;
    }
    
    .image-caption {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        font-weight: 500;
        margin-top: 1rem;
        text-shadow: 0 2px 10px rgba(0,0,0,0.7);
    }
    
    .image-meta {
        color: rgba(102, 126, 234, 0.8);
        font-size: 0.9rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    /* File Upload Styling */
    .stFileUploader {
        background: rgba(10,10,10,0.6) !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 15px !important;
        padding: 1rem !important;
    }
    
    .stFileUploader > div > div {
        background: transparent !important;
        border: none !important;
    }
    
    .stFileUploader label {
        color: rgba(255,255,255,0.9) !important;
        font-weight: 500 !important;
    }
    
    /* Camera Input Styling */
    .stCameraInput {
        background: rgba(10,10,10,0.6) !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 15px !important;
        padding: 1rem !important;
    }
    
    .upload-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.1) 50%, transparent 70%);
        animation: scanLine 3s ease-in-out infinite;
    }
    
    @keyframes scanLine {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .upload-title {
        font-size: 1.8rem;
        font-weight: 600;
        color: white;
        margin-bottom: 1rem;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    
    /* Analysis Cards */
    .analysis-card {
        background: linear-gradient(135deg, rgba(10,10,10,0.8), rgba(15,52,96,0.3));
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(102,126,234,0.4);
        box-shadow: 0 15px 35px rgba(0,0,0,0.5);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .analysis-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 25px 50px rgba(102,126,234,0.2);
        border-color: rgba(102,126,234,0.6);
    }
    
    .analysis-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        animation: cardGlow 2s ease-in-out infinite alternate;
    }
    
    @keyframes cardGlow {
        from { opacity: 0.5; }
        to { opacity: 1; }
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #0f3460) !important;
        color: white !important;
        border: none !important;
        border-radius: 15px !important;
        padding: 1rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        letter-spacing: 1px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 10px 30px rgba(15, 52, 96, 0.5) !important;
        position: relative !important;
        overflow: hidden !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4) !important;
        background: linear-gradient(135deg, #4facfe, #00d4ff) !important;
        border-color: rgba(102, 126, 234, 0.6) !important;
    }
    
    .stButton > button::before {
        content: '' !important;
        position: absolute !important;
        top: 0 !important;
        left: -100% !important;
        width: 100% !important;
        height: 100% !important;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent) !important;
        transition: left 0.5s !important;
    }
    
    .stButton > button:hover::before {
        left: 100% !important;
    }
    
    /* Severity Indicators */
    .severity-high {
        color: #ff6b6b !important;
        font-weight: bold !important;
        text-shadow: 0 2px 10px rgba(255, 107, 107, 0.3) !important;
        animation: pulse 2s ease-in-out infinite !important;
    }
    
    .severity-medium {
        color: #feca57 !important;
        font-weight: bold !important;
        text-shadow: 0 2px 10px rgba(254, 202, 87, 0.3) !important;
    }
    
    .severity-low {
        color: #48dbfb !important;
        font-weight: bold !important;
        text-shadow: 0 2px 10px rgba(72, 219, 251, 0.3) !important;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    /* Loading Animation */
    .stSpinner > div {
        border-color: rgba(102, 126, 234, 0.3) !important;
        border-top-color: #667eea !important;
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.2), rgba(0, 212, 255, 0.1)) !important;
        border: 1px solid rgba(0, 212, 255, 0.4) !important;
        color: white !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.2), rgba(255, 107, 107, 0.1)) !important;
        border: 1px solid rgba(255, 107, 107, 0.4) !important;
        color: white !important;
        backdrop-filter: blur(10px) !important;
    }
    
    /* Footer */
    .premium-footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        background: linear-gradient(135deg, rgba(10,10,10,0.9), rgba(15,52,96,0.3));
        border-radius: 15px;
        border: 1px solid rgba(102,126,234,0.3);
        backdrop-filter: blur(15px);
        color: rgba(255, 255, 255, 0.9);
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(10,10,10,0.8), rgba(15,52,96,0.3)) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(102,126,234,0.4) !important;
        color: white !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .streamlit-expanderContent {
        background: linear-gradient(135deg, rgba(10,10,10,0.7), rgba(15,52,96,0.2)) !important;
        border: 1px solid rgba(102,126,234,0.3) !important;
        border-radius: 0 0 10px 10px !important;
        backdrop-filter: blur(15px) !important;
    }
    
    /* Radio Button Styling */
    .stRadio > div > div > div > div {
        background: rgba(10,10,10,0.6) !important;
        border: 1px solid rgba(102,126,234,0.4) !important;
        color: white !important;
        backdrop-filter: blur(10px) !important;
    }
    
    /* Text Color */
    .stMarkdown, .stText, p, span, div {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Sidebar Text */
    .css-1d391kg .stMarkdown, .css-1cypcdb .stMarkdown {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Floating Elements Animation */
    .floating-element {
        animation: float 6s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
    }
    
    /* Particles Background Effect */
    .particles {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: -1;
    }
    
    .particle {
        position: absolute;
        background: rgba(255, 255, 255, 0.3);
        border-radius: 50%;
        animation: particleFloat 20s infinite linear;
    }
    
    @keyframes particleFloat {
        0% {
            transform: translateY(100vh) rotate(0deg);
            opacity: 0;
        }
        10% {
            opacity: 1;
        }
        90% {
            opacity: 1;
        }
        100% {
            transform: translateY(-100vh) rotate(360deg);
            opacity: 0;
        }
    }
</style>
""", unsafe_allow_html=True)

# Add floating particles effect
st.markdown("""
<div class="particles">
    <div class="particle" style="left: 10%; width: 4px; height: 4px; animation-delay: 0s;"></div>
    <div class="particle" style="left: 20%; width: 6px; height: 6px; animation-delay: 2s;"></div>
    <div class="particle" style="left: 30%; width: 3px; height: 3px; animation-delay: 4s;"></div>
    <div class="particle" style="left: 40%; width: 5px; height: 5px; animation-delay: 6s;"></div>
    <div class="particle" style="left: 50%; width: 4px; height: 4px; animation-delay: 8s;"></div>
    <div class="particle" style="left: 60%; width: 7px; height: 7px; animation-delay: 10s;"></div>
    <div class="particle" style="left: 70%; width: 3px; height: 3px; animation-delay: 12s;"></div>
    <div class="particle" style="left: 80%; width: 5px; height: 5px; animation-delay: 14s;"></div>
    <div class="particle" style="left: 90%; width: 4px; height: 4px; animation-delay: 16s;"></div>
</div>
""", unsafe_allow_html=True)
image = Image.open("C:/Users/DELL/Downloads/reve.png")

    # Sidebar with Premium Logo
with st.sidebar:
            # 🧬
        st.markdown(f"""
    <div class="logo-container">
        <div class="custom-logo">
            <img src="data:image/png;base64,{image_base64}" alt="REVE AI Logo" />
        </div>
    </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center; margin: 1rem 0;">
            <h2 style="color: white; font-weight: 800; font-size: 1.8rem; margin: 0;">REVE AI</h2>
            <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem; margin: 0.5rem 0;">Advanced Dermatology Intelligence</p>
        </div>
        """, unsafe_allow_html=True)
        input_method = st.radio(
    "**Choose Analysis Method:**", 
    ["📤 Upload from Device", "📸 Capture with Webcam", "🔌 USB Microscope"],
    horizontal=False
)
    



# Clean duplicate report function
def clean_report_content(report_text):
    parts = report_text.split("# Skin Disease Diagnosis Report 🏥")
    if len(parts) > 2:
        return "# Skin Disease Diagnosis Report 🏥" + parts[2]
    return report_text
def extract_float_from_string(s):
    match = re.search(r"[-+]?\d*\.\d+|\d+", s)
    return float(match.group()) if match else 10

# Premium Header
st.markdown("""
<div class="premium-header">
    <h1 class="main-title">REVE AI</h1>
    <p class="subtitle">Agentic AI & Deep Image Analysis based</p>
    <p style="color: rgba(255,255,255,0.7); font-size: 1rem; margin-top: 1rem;">
        Advanced Skin Disease & Cancer Prediction Platform for Dermatologists
    </p>
</div>
""", unsafe_allow_html=True)

# Upload Section with Premium Design
st.markdown("""
<div class="upload-container floating-element">
    <h3 class="upload-title">🎯 Advanced AI Skin Diagnosis</h3>
    <p style="color: rgba(255,255,255,0.8); font-size: 1.1rem; margin-bottom: 1.5rem;">
        Upload or capture high-resolution skin images for instant AI-powered diagnosis
    </p>
</div>
""", unsafe_allow_html=True)



image = None
uploaded_file = None

col1, col2 = st.columns(2)

with col1:
    if input_method == "📤 Upload from Device":
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        image = Image.open(uploaded_file) if uploaded_file is not None else None
    
    elif input_method == "📸 Capture with Webcam":
        captured_file = st.camera_input("📸 Capture high-quality skin image")
        print("Captured file:", captured_file)
        if captured_file is not None:
            uploaded_file = captured_file
            image = Image.open(captured_file)

    elif input_method == "🔌 USB Microscope":
        st.write("📷 Press SPACE in the popup window to capture an image.")
        if st.button("🔴 Launch USB Microscope Capture"):
            subprocess.run(["python", "cap.py"])  
            uploaded_file_path = r"C:\Users\DELL\OneDrive\CodeDB\Project\PY_ML_DL_GenAI\Skin_Cancer\usb_microscope_capture.jpg"
            import io
            with open(uploaded_file_path, "rb") as f:
                file_bytes = f
                u_file = io.BytesIO(file_bytes)
                image = Image.open(u_file) if u_file is not None else None
                desired_height = 300
                aspect_ratio = image.width / image.height
                new_width = int(desired_height * aspect_ratio)
                resized_image = image.resize((new_width, desired_height))
                st.image(resized_image, caption="📋 Uploaded Skin Image for Analysis")
        st.markdown("📂 Please upload the image from: `Skin_Cancer/usb_microscope_capture.jpg`")
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")        
        image = Image.open(uploaded_file) if uploaded_file is not None else None    # uploaded_file = io.BytesIO(file_bytes)

with col2:
    if uploaded_file is not None and image is not None:
        desired_height = 300
        aspect_ratio = image.width / image.height
        new_width = int(desired_height * aspect_ratio)
        resized_image = image.resize((new_width, desired_height))
        st.image(resized_image, caption="📋 Uploaded Skin Image for Analysis")


print("Image uploaded successfully:", uploaded_file is not None, uploaded_file)
# Premium Analyze Button
if uploaded_file is not None and st.button("🔬 ANALYZE WITH AI", type="primary"):
    with st.spinner("🧠 Processing with Advanced AI Models... Please wait 30-60 seconds for comprehensive analysis..."):
        try:
            files = {"file": uploaded_file.getvalue()}
            from PIL import Image
            from io import BytesIO

            def analyze_image_predictions(file):
                    contents = file.read()
                    image = Image.open(BytesIO(contents)).convert("RGB")

                    result_c = predict_c(image)
                    result_d = predict_d(image)

                    if result_c["confidence"] > result_d["confidence"]:
                        result_pred = result_c
                        minor_result = result_d
                    elif result_d["confidence"] > result_c["confidence"]:
                        result_pred = result_d
                        minor_result = result_c
                    else:
                        result_pred = result_c
                        minor_result = result_d

                    results = {
                        "primary": result_pred,
                        "secondary": minor_result
                    }
                    primary = results["primary"]
                    secondary = results["secondary"]
                    with st.sidebar:
                        st.markdown("## 🧠 Raw Prediction Results")
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("### ✅ Primary Prediction")
                            st.success(f"**{primary['class']}**")
                            st.progress(int(primary["confidence"] * 100))
                            st.markdown(f"**Confidence:** {primary['confidence']:.2%}")

                        with col2:
                            st.markdown("### ⚠️ Secondary Possibility")
                            st.info(f"**{secondary['class']}**")
                            st.progress(int(secondary["confidence"] * 100))
                            st.markdown(f"**Confidence:** {secondary['confidence']:.2%}")
                    
                
                    
            try:
                    analyze_image_predictions(uploaded_file.getvalue())
            except Exception as e:
                    st.error(f"⚠️ print failed: {str(e)}")
            import time        
            start_time = time.time()
            response = requests.post("http://127.0.0.1:6701/predict", files=files)
            end_time = time.time()
            total_time = end_time - start_time
            st.write(f"🕒 Total processing time: {total_time:.2f} seconds")
            if response.status_code == 200:
                results = response.json()
                cleaned_report = clean_report_content(results["report"])
                st.success("✅ AI Analysis Complete! Results generated successfully.")

                col1, col2 = st.columns(2)
                with col1:
                    with st.expander("🔍 Initial AI Assessment", expanded=True):
                        st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
                        verify_parts = results['verify'].strip('`').split(',')
                        st.write(f"**🎯 Classification:** {verify_parts[0]}")
                        confidence = verify_parts[1].replace('%', '')
                        x=extract_float_from_string(confidence)
                        conf_class = "severity-high" if x > 80 else "severity-medium" if x > 50 else "severity-low"
                        st.write(f"**📊 Confidence:** <span class='{conf_class}'>{verify_parts[1]}</span>", unsafe_allow_html=True)
                        st.write(f"**🧬 Skin Type:** {verify_parts[2]}")
                        st.write(f"**💡 Remarks:** {verify_parts[3]}")
                        st.markdown('</div>', unsafe_allow_html=True)

                with col2:
                    with st.expander("🧠 Deep Learning Prediction", expanded=True):
                        st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
                        pred_parts = results['prediction'].split(',')
                        st.write(f"**🏥 Condition:** {pred_parts[0]}")
                        confidence = pred_parts[1].replace('%', '')
                        y=extract_float_from_string(confidence)
                        conf_class = "severity-high" if y > 80 else "severity-medium" if y > 50 else "severity-low"
                        st.write(f"**📈 Confidence:** <span class='{conf_class}'>{pred_parts[1]}</span>", unsafe_allow_html=True)
                        if len(pred_parts) > 2:
                            st.write(f"**📝 Analysis:** {pred_parts[2]}")
                        st.markdown('</div>', unsafe_allow_html=True)

                with st.expander("📋 Comprehensive Diagnosis Report", expanded=True):
                    st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
                    st.markdown(results["report"])
                    # st.markdown(cleaned_report)
                    st.markdown('</div>', unsafe_allow_html=True)

                # with st.expander("💊 Clinical Recommendations & Treatment", expanded=False):
                #     st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
                #     st.markdown(results['jarvis'])
                #     st.markdown('</div>', unsafe_allow_html=True)
            else:
                from PIL import Image
                from io import BytesIO

                def analyze_image_predictions(file):
                    contents = file.read()
                    image = Image.open(BytesIO(contents)).convert("RGB")

                    result_c = predict_c(image)
                    result_d = predict_d(image)

                    if result_c["confidence"] > result_d["confidence"]:
                        result_pred = result_c
                        minor_result = result_d
                    elif result_d["confidence"] > result_c["confidence"]:
                        result_pred = result_d
                        minor_result = result_c
                    else:
                        result_pred = result_c
                        minor_result = result_d

                    result = {
                        "primary": result_pred,
                        "secondary": minor_result
                    }
                    return result
                
                def display_results(results):
                    primary = results["primary"]
                    secondary = results["secondary"]
                    with st.sidebar:
                        st.markdown("## 🧠 Prediction Results")
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("### ✅ Primary Prediction")
                            st.success(f"**{primary['class']}**")
                            st.progress(int(primary["confidence"] * 100))
                            st.markdown(f"**Confidence:** {primary['confidence']:.2%}")

                        with col2:
                            st.markdown("### ⚠️ Secondary Possibility")
                            st.info(f"**{secondary['class']}**")
                            st.progress(int(secondary["confidence"] * 100))
                            st.markdown(f"**Confidence:** {secondary['confidence']:.2%}")
                    
                try:
                    results = analyze_image_predictions(uploaded_file.getvalue())
                    display_results(results)
                    st.error(f"❌ Error analyzing image: {response.text}")
                except Exception as e:
                    st.error(f"⚠️ print failed: {str(e)}")

        except Exception as e:
            from PIL import Image
            from io import BytesIO

            def analyze_image_predictions(file):
                contents = file
                image = Image.open(BytesIO(contents)).convert("RGB")

                result_c = predict_c(image)
                result_d = predict_d(image)

                if result_c["confidence"] > result_d["confidence"]:
                    result_pred = result_c
                    minor_result = result_d
                elif result_d["confidence"] > result_c["confidence"]:
                    result_pred = result_d
                    minor_result = result_c 
                else:
                    result_pred = result_c
                    minor_result = result_d

                result = {
                    "primary": result_pred,
                    "secondary": minor_result
                }
                return result
            
            def display_results(results):
                primary = results["primary"]
                secondary = results["secondary"]

                st.markdown("## 🧠 Prediction Results")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### ✅ Primary Prediction")
                    st.success(f"**{primary['class']}**")
                    st.progress(int(primary["confidence"] * 100))
                    st.markdown(f"**Confidence:** {primary['confidence']:.2%}")

                with col2:
                    st.markdown("### ⚠️ Secondary Possibility")
                    st.info(f"**{secondary['class']}**")
                    st.progress(int(secondary["confidence"] * 100))
                    st.markdown(f"**Confidence:** {secondary['confidence']:.2%}")
                
            try:
                results = analyze_image_predictions(uploaded_file.getvalue())
                display_results(results)
                st.error(f"⚠️ Analysis failed: {str(e)}")
            except Exception as e:
                st.error(f"⚠️ print failed: {str(e)}")
else:
    print("No file uploaded or capture not taken.")
with st.sidebar:
        st.markdown("---")
        st.markdown("🔬 **AI-Powered Skin Analysis**")
        st.markdown("🎯 **Deep AI Analysis**")
        st.markdown("📊 **Professional Diagnostics**")
        st.markdown("⚡ **Real-time Processing**")
        st.markdown("---")
        st.info("🩺 For professional medical use only", icon="⚠️")
        st.markdown("---")
# Premium Footer
st.markdown("""
<div class="premium-footer">
    <h3 style="color: white; margin-bottom: 1rem;">🧬 REVE AI</h3>
    <p><strong>© 2025 REVE AI - Professional Dermatology Intelligence Platform</strong></p>
    <p>🏥 Exclusively designed for certified dermatologists and medical professionals</p>
    <p>⚠️ Emergency cases require immediate consultation with healthcare specialists</p>
    <div style="margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.2);">
        <p style="font-size: 0.9rem; color: rgba(255,255,255,0.7);">
            Powered by Advanced Deep Learning • Agentic AI Technology • Professional Grade Analysis
        </p>
    </div>
</div>
""", unsafe_allow_html=True)