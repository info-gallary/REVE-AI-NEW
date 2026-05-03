import streamlit as st
import requests
from PIL import Image
import re



# Page Configuration
st.set_page_config(
    page_title="REVE AI",
    page_icon="🏥",
    layout="wide"
)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2965/2965567.png", width=100)
    st.title("🏥 REVE AI")
    st.markdown("**Advanced Skin Disease Diagnosis**")
    st.markdown("---")
    st.markdown("🔍 Upload or capture a skin image to get instant AI-powered analysis.")
    st.markdown("💡 Developed for research & medical professionals.")
    st.markdown("---")
    st.info("For emergency, always consult a certified doctor.", icon="⚠️")

# Custom CSS
st.markdown("""
<style>
    .main {
        background: linear-gradient(to bottom right, #f0f4f8, #e3f2fd);
        padding: 2rem;
        border-radius: 10px;
    }
    .upload-box {
        border: 2px dashed #0077b6;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(to right, #ade8f4, #caf0f8);
    }
    .report-section {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-top: 1.5rem;
    }
    .header {
        color: #023e8a;
        padding-bottom: 1rem;
        border-bottom: 2px solid #0077b6;
    }
    .severity-high {
        color: #d90429;
        font-weight: bold;
    }
    .severity-medium {
        color: #ffb703;
        font-weight: bold;
    }
    .severity-low {
        color: #40916c;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #0077b6;
        color: white;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# Clean duplicate report
def clean_report_content(report_text):
    parts = report_text.split("# Skin Disease Diagnosis Report 🏥")
    if len(parts) > 2:
        return "# Skin Disease Diagnosis Report 🏥" + parts[2]
    return report_text

def extract_float_from_string(s):
    match = re.search(r"[-+]?\d*\.\d+|\d+", s)
    return float(match.group()) if match else 10

# Header
st.markdown("<div class='header'><h1>🏥 REVE AI</h1><h3>Advanced Skin Disease Diagnosis</h3></div>", unsafe_allow_html=True)

# Upload or Capture Section
st.markdown("<div class='upload-box'><h3>📤 Upload or Capture Skin Image for AI Diagnosis</h3></div>", unsafe_allow_html=True)
input_method = st.radio("Choose input method:", ["Upload from Device", "Capture with Webcam"], horizontal=True)

image = None
uploaded_file = None

if input_method == "Upload from Device":
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Skin Image", width=300)

elif input_method == "Capture with Webcam":
    captured_file = st.camera_input("Capture your skin image")
    if captured_file is not None:
        image = Image.open(captured_file)
        uploaded_file = captured_file
        st.image(image, caption="Captured Skin Image", width=300)

# Analyze Button
if uploaded_file is not None and st.button("Analyze Image", type="primary"):
    with st.spinner("🔍 Analyzing image with our AI models. This may take 30-60 seconds..."):
        try:
            files = {"file": uploaded_file.getvalue()}
            response = requests.post("http://127.0.0.1:6701/predict", files=files)

            if response.status_code == 200:
                results = response.json()
                cleaned_report = clean_report_content(results["report"])
                st.success("✅ Analysis Complete!")

                col1, col2 = st.columns(2)
                with col1:
                    with st.expander("🧾 Initial Assessment", expanded=True):
                        verify_parts = results['verify'].strip('`').split(',')
                        st.write(f"**Classification:** {verify_parts[0]}")
                        confidence = verify_parts[1].replace('%', '')
                        conf_class = "severity-high" if float(confidence) > 80 else "severity-medium" if float(confidence) > 50 else "severity-low"
                        st.write(f"**Confidence:** <span class='{conf_class}'>{verify_parts[1]}</span>", unsafe_allow_html=True)
                        st.write(f"**Skin Type:** {verify_parts[2]}")
                        st.write(f"**Remarks:** {verify_parts[3]}")

                with col2:
                    with st.expander("🧬 Disease Prediction", expanded=True):
                        pred_parts = results['prediction'].split(',')
                        st.write(f"**Condition:** {pred_parts[0]}")
                        confidence = pred_parts[1].replace('%', '')
                        x=extract_float_from_string(confidence)
                        conf_class = "severity-high" if x > 80 else "severity-medium" if  x > 50 else "severity-low"
                        st.write(f"**Confidence:** <span class='{conf_class}'>{pred_parts[1]}</span>", unsafe_allow_html=True)
                        if len(pred_parts) > 2:
                            st.write(f"**Remarks:** {pred_parts[2]}")

                with st.expander("📋 Full Diagnosis Report", expanded=True):
                    st.markdown("<div class='report-section'>", unsafe_allow_html=True)
                    st.markdown(cleaned_report)
                    st.markdown("</div>", unsafe_allow_html=True)

                with st.expander("💊 Clinical Recommendations", expanded=False):
                    st.markdown("<div class='report-section'>", unsafe_allow_html=True)
                    st.markdown(results['jarvis'])
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.error(f"❌ Error analyzing image: {response.text}")

        except Exception as e:
            st.error(f"⚠️ An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 1rem;">
    <p>© 2025 REVE AI | For professional medical use only</p>
    <p>Not for emergency use | Always consult a healthcare professional</p>
</div>
""", unsafe_allow_html=True)
