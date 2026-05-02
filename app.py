import streamlit as st 
import numpy as np
import pandas as pd
from PIL import Image
from utils.preprocessing import predict_xray

# ------------------ INPUT VALIDATION ------------------
def is_likely_xray(image):
    """
    Lightweight heuristic to check if an uploaded image
    looks like a grayscale X-ray. Compares the mean intensity
    of the R, G, B channels — in a true grayscale image they
    will be nearly identical.
    Returns True if the image appears to be an X-ray.
    """
    img_array = np.array(image.convert("RGB"))
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    # Mean absolute difference between channels
    rg_diff = np.mean(np.abs(r.astype(float) - g.astype(float)))
    rb_diff = np.mean(np.abs(r.astype(float) - b.astype(float)))
    gb_diff = np.mean(np.abs(g.astype(float) - b.astype(float)))
    avg_diff = (rg_diff + rb_diff + gb_diff) / 3.0
    # Threshold: if channels differ by more than 15 on average,
    # the image is likely a colour photograph, not an X-ray.
    return avg_diff < 15.0

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Chest X-ray Classifier",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background-color: #0077b5;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #005582;
        border-color: #005582;
        color: white;
    }
    .header-container {
        text-align: center;
        padding: 2rem 1rem 1.5rem;
    }
    .header-icon {
        font-size: 2.8rem;
        display: inline-block;
        background: linear-gradient(135deg, #0077b5, #00b4d8);
        padding: 0.6rem 1.2rem;
        border-radius: 16px;
        margin-bottom: 0.6rem;
    }
    .main-title {
        text-align: center;
        font-weight: 800;
        font-size: 2.6rem;
        background: linear-gradient(90deg, #0077b5, #00b4d8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.3rem 0 0;
        letter-spacing: -0.5px;
    }
    .sub-title {
        text-align: center;
        color: #7f8c8d;
        font-size: 1.15rem;
        margin-top: 0.4rem;
        margin-bottom: 0;
        font-weight: 400;
        letter-spacing: 0.3px;
    }
    .desc-card {
        max-width: 700px;
        margin: 1.2rem auto 0;
        background: rgba(0,119,181,0.06);
        border-left: 4px solid #0077b5;
        border-radius: 0 10px 10px 0;
        padding: 1rem 1.4rem;
        font-size: 1rem;
        color: #4a4a4a;
        line-height: 1.6;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.header("ℹ️ Information")
    
    st.markdown("### How It Works")
    st.markdown("""
    1. **Upload** a patient's chest X-ray image (JPG/PNG).
    2. Click **Run Prediction**.
    3. The AI model analyzes the radiograph.
    4. Review the diagnosis and confidence scores.
    """)
    st.divider()
    
    st.markdown("### About the Model")
    st.markdown("""
    This application utilizes a custom-trained **Convolutional Neural Network (CNN)** 
    optimized for medical imaging to classify radiographs into three distinct categories: 
    **Normal**, **Pneumonia**, and **Tuberculosis (TB)**.
    """)
    st.divider()
    
    st.markdown("### Common Symptoms")
    st.markdown("""
    **Pneumonia:**
    - Cough with phlegm or pus
    - Fever, chills, and difficulty breathing
    
    **Tuberculosis:**
    - Persistent cough (lasting >3 weeks)
    - Chest pain, coughing up blood
    - Night sweats and weight loss
    """)
    st.divider()
    
    st.markdown("### ⚠️ Disclaimer")
    st.warning("""
    This application is for **educational and demonstration purposes only**. 
    It is **not** a substitute for professional medical advice, diagnosis, or treatment. 
    Always consult a qualified healthcare provider with any medical questions.
    """)

# ------------------ HEADER ------------------
st.markdown("""
<style>
.header-container {
    text-align: center;
    padding: 30px 20px;
    border-radius: 16px;
    background: linear-gradient(145deg, #0f172a, #1e293b);
    box-shadow: 0 8px 30px rgba(0,0,0,0.4);
}

.header-icon {
    font-size: 50px;
    margin-bottom: 10px;
}

.header-title {
    font-size: 42px;
    font-weight: 700;
    background: linear-gradient(90deg, #38bdf8, #22c55e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
}

.header-subtitle {
    font-size: 18px;
    color: #94a3b8;
    margin-bottom: 20px;
}

.header-desc {
    font-size: 15px;
    color: #cbd5f5;
    max-width: 700px;
    margin: auto;
    line-height: 1.6;
}
</style>

<div class="header-container">
    <div class="header-icon">🩺</div>
    <div class="header-title">Chest X-ray Classifier</div>
    <div class="header-subtitle">
        AI-Powered Diagnostic Assistance
    </div>
    <div class="header-desc">
        Upload a chest X-ray image and let our deep learning model detect 
        signs of <b>Normal</b>, <b>Pneumonia</b>, or <b>Tuberculosis</b> 
        with confidence scores and visual insights.
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ------------------ FILE UPLOAD ------------------
with st.container():
    st.subheader("📁 Upload Patient Image")
    st.markdown("Please upload a clear, high-quality chest X-ray image (JPG or PNG) for analysis.")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

# ------------------ MAIN PREDICTION ------------------
if uploaded_file is not None:
    st.divider()
    
    # Use st.columns to split layout
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("📷 Image Preview")
        st.image(uploaded_file, caption="Uploaded X-ray Image", use_column_width=True)
            
    with col2:
        st.subheader("🔬 Analysis & Results")
        
        # Predict Button Centered
        st.write("") # Spacing
        btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
        with btn_col2:
            predict_button = st.button("🔍 Run Prediction")
            
        if predict_button:
            # --- Step 1: Validate input image ---
            pil_image = Image.open(uploaded_file)
            if not is_likely_xray(pil_image):
                st.error("🚫 This does not appear to be a chest X-ray image.")
                st.info("Please upload a valid grayscale chest radiograph (JPG/PNG).")
                st.stop()
            
            # Reset file pointer so predict_xray reads from the start
            uploaded_file.seek(0)
            
            # Loading UX
            with st.spinner("Analyzing X-ray using deep learning model..."):
                # DO NOT change model loading logic
                label, probs = predict_xray(uploaded_file)
            
            # --- Step 2: Low-confidence gate ---
            max_confidence = float(np.max(probs))
            if max_confidence < 0.65:
                st.warning("⚠️ Low confidence prediction. The model is not confident enough to provide a reliable diagnosis.")
                st.info("Please upload a clearer, properly oriented chest X-ray image.")
                st.stop()
            
            st.write("---")
            
            # Results Section
            if label.strip().lower() == "normal":
                st.success(f"### Diagnosis: **{label}** ✅")
                st.info("No significant signs of Pneumonia or Tuberculosis detected.")
            else:
                st.error(f"### Diagnosis: **{label}** ⚠️")
                st.warning("Abnormalities detected. Please consult a radiologist for clinical confirmation.")
                
            st.markdown("#### Confidence Scores")
            
            for name, p in zip(["Normal", "Pneumonia", "TB"], probs):
                st.markdown(f"- **{name}:** `{p*100:.2f}%`")
                
            st.markdown("#### Probability Distribution")
            labels = ["Normal", "Pneumonia", "TB"]
            df = pd.DataFrame({"Confidence %": probs * 100}, index=labels)
            st.bar_chart(df)

# ------------------ FOOTER ------------------
st.divider()
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p style='font-size: 16px; margin-bottom: 5px;'>Created by <b>Diwanshu</b> & team.</p>
        <p>
            <a href='https://www.linkedin.com/in/diwanshu-gangwar/' target='_blank' 
            style='text-decoration: none; font-size: 14px; color: #0077b5; font-weight: bold; padding: 8px 12px; border: 1px solid #0077b5; border-radius: 5px; transition: all 0.3s;'>
                🤝 Connect on LinkedIn
            </a>
        </p>
    </div>
""", unsafe_allow_html=True)
