import streamlit as st
import numpy as np
import pandas as pd
import json
from io import BytesIO
from utils.preprocessing import predict_xray  # existing function: returns (label, probs)
# If you later create a function to produce Grad-CAM heatmaps, import it here:
# from utils.explainability import generate_gradcam

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Chest X-ray Classifier",
    page_icon="🩺",
    layout="wide"  # wider layout looks nicer for presentations
)

# ------------------ SMALL CSS POLISH ------------------
st.markdown(
    """
    <style>
    .reportview-container .main .block-container{padding:1.2rem 2rem;}
    h1{font-size:30px;}
    .footer {text-align:center; color: #6c6c6c; font-size:14px; margin-top: 12px;}
    .link {text-decoration:none; color:#0077b5; font-weight:600;}
    .pred-badge {font-size:20px; font-weight:700; color:#111;}
    .small-muted {color:#6c6c6c; font-size:13px;}
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------ SIDEBAR (OPTIONS + INFO) ------------------
with st.sidebar:
    st.header("Options & Info")
    st.write("📌 Upload a chest X-ray (JPG/PNG). The model classifies into **Normal**, **Pneumonia**, or **TB**.")
    st.markdown("---")

    # Example images (if you want to ship a few sample X-rays with your repo)
    st.subheader("Try sample X-rays")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Sample: Normal"):
            st.session_state['sample'] = "samples/normal_example.jpg"
    with col2:
        if st.button("Sample: Pneumonia"):
            st.session_state['sample'] = "samples/pneumonia_example.jpg"
    if 'sample' in st.session_state:
        try:
            st.image(st.session_state['sample'], use_column_width=True)
        except Exception:
            st.info("Sample file not found in repo. Place sample images under `samples/`.")

    st.markdown("---")
    # Advanced options (non-blocking)
    st.subheader("Advanced")
    threshold = st.slider("Decision threshold for top class (%)", 50, 95, 50)
    show_explain = st.checkbox("Show explainability (Grad-CAM)", value=False)
    st.markdown("---")
    st.write("Author")
    st.markdown("**Diwanshu & team**")
    st.markdown("[LinkedIn](https://www.linkedin.com/in/diwanshu-gangwar/)")

# ------------------ HEADER ------------------
st.title("🩺 Chest X-ray Classifier")
st.markdown("Upload a chest X-ray to classify it as **Normal**, **Pneumonia**, or **TB**. "
            "Use the sidebar to try sample images or adjust options.")

# ------------------ FILE UPLOAD ------------------
uploaded_file = st.file_uploader("Upload an X-ray image (JPG / PNG)", type=["jpg", "jpeg", "png"])

# If user clicked a sample button, set uploaded_file to that file (optional)
if 'sample' in st.session_state and uploaded_file is None:
    try:
        with open(st.session_state['sample'], "rb") as f:
            uploaded_bytes = f.read()
        uploaded_file = BytesIO(uploaded_bytes)
    except Exception:
        pass

# ------------------ MAIN PREDICTION AREA ------------------
if uploaded_file is not None:
    # show columns: left = image, right = results
    col_img, col_res = st.columns([1, 1])

    with col_img:
        st.image(uploaded_file, caption="Uploaded X-ray", use_column_width=True)
        st.write("")  # spacer

    with col_res:
        # Predict button separated from upload so user can preview image first
        if st.button("🔍 Predict"):
            # spinner while processing
            with st.spinner("Running model..."):
                # Call your existing prediction function (keeps compatibility)
                # Expect predict_xray to return: (label_str, probs_numpy_array_of_len_3)
                label, probs = predict_xray(uploaded_file)

            # Normalise probs -> ensure numpy array
            probs = np.array(probs).astype(float)
            labels = ["Normal", "Pneumonia", "TB"]
            top_idx = int(np.argmax(probs))
            top_label = labels[top_idx]
            top_score = float(probs[top_idx]) * 100.0

            # Decide pass/fail based on threshold slider
            status = "Confident" if top_score >= threshold else "Low confidence"

            # show top prediction as metric
            st.markdown(f"<div class='pred-badge'>Prediction: {top_label}  —  <span class='small-muted'>{status}</span></div>",
                        unsafe_allow_html=True)
            st.write("")  # spacer
            st.markdown("**Confidence Scores**")
            # Build dataframe for chart
            df = pd.DataFrame({"label": labels, "confidence": (probs * 100)})
            df = df.set_index("label")
            # Bar chart (streamlit native)
            st.bar_chart(df)

            # Pretty list with percentages and tiny progress bars (HTML)
            for name, p in zip(labels, probs):
                percent = p * 100
                color = "#2ecc71" if name == top_label else "#6c6c6c"
                st.markdown(f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
                            f"<div style='width:45%;'>{name}</div>"
                            f"<div style='width:50%;'>"
                            f"<progress value='{percent:.2f}' max='100' style='width:100%; height:12px;'></progress>"
                            f"</div>"
                            f"<div style='width:10%; text-align:right; padding-left:8px;'>{percent:.2f}%</div>"
                            f"</div>",
                            unsafe_allow_html=True)

            # Save last result to session state for download
            st.session_state['last_prediction'] = {
                "prediction": top_label,
                "confidence": float(top_score),
                "confidence_vector": (probs * 100).tolist()
            }

            # Optional explainability (placeholder)
            if show_explain:
                st.markdown("**Explainability (Grad-CAM)**")
                try:
                    # If you implement generate_gradcam, it should return an image/bytes you can display
                    # heatmap = generate_gradcam(uploaded_file)
                    # st.image(heatmap, caption="Grad-CAM heatmap", use_column_width=True)
                    st.info("Grad-CAM is not implemented yet. Add `utils/explainability.py` and call it here.")
                except Exception as e:
                    st.error(f"Explainability failed: {e}")

            # Download button: JSON
            result_json = json.dumps(st.session_state['last_prediction'], indent=2)
            st.download_button(
                label="⬇️ Download result (JSON)",
                data=result_json,
                file_name="xray_prediction.json",
                mime="application/json"
            )

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown(
    """
    <div class="footer">
      <div>Created by <b>Diwanshu</b> & team • For academic demo / final-year project</div>
      <div><a class="link" href="https://www.linkedin.com/in/diwanshu-gangwar/" target="_blank">Connect on LinkedIn</a></div>
    </div>
    """,
    unsafe_allow_html=True
)
