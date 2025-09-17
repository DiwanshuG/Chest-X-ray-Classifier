import streamlit as st
import numpy as np
import pandas as pd
from utils.preprocessing import predict_xray

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Chest X-ray Classifier",
    page_icon="🩺",
    layout="centered"
)

# ------------------ HEADER ------------------
st.title("🩺 Chest X-ray Classifier")
st.markdown("Upload a chest X-ray to classify it as **Normal**, **Pneumonia**, or **TB**.")

# ------------------ FILE UPLOAD ------------------
uploaded_file = st.file_uploader(
    "Upload an X-ray image (JPG/PNG)", type=["jpg", "jpeg", "png"]
)

# ------------------ MAIN PREDICTION ------------------
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded X-ray", use_column_width=True)
    st.write("---")

    if st.button("🔍 Predict"):
        # Call your prediction function
        label, probs = predict_xray(uploaded_file)

        st.subheader(f"**Prediction:** {label}")
        st.write("### Confidence Scores")

        for name, p in zip(["Normal", "Pneumonia", "TB"], probs):
            st.write(f"- {name}: **{p*100:.2f}%**")

        # ✅ Confidence bar chart
        labels = ["Normal", "Pneumonia", "TB"]
        df = pd.DataFrame({"Confidence %": probs * 100}, index=labels)
        st.bar_chart(df)

# ------------------ FOOTER ------------------
st.write("---")
st.markdown(
    """
    <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #f9f9f9;
            color: #333333;
            text-align: center;
            padding: 12px;
            border-top: 1px solid #e0e0e0;
            font-size: 16px;
        }
        .footer a {
            text-decoration: none;
            font-weight: bold;
            color: #1f77b4;
        }
    </style>

    <div class="footer">
        Created by <b>Diwanshu</b> with ❤️ | 
        <a href="https://www.linkedin.com/in/diwanshu-gangwar/" target="_blank">Connect with me</a>
    </div>
    """,
    unsafe_allow_html=True
)


# ------------------ END ------------------
