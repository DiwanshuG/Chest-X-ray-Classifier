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

    if st.button("🔍Analyse X-ray "):
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
st.markdown("""
    <hr>
    <p style='text-align: center; font-size:16px;'>Created by <b>Diwanshu</b> with ❤️</p>
    <p style='text-align: center;'>
        <a href='https://www.linkedin.com/in/diwanshu-gangwar/' target='_blank' 
        style='text-decoration: none; font-size:16px; color: #0077b5; font-weight: bold;'>
            Connect with me 
        </a>
    </p>
""", unsafe_allow_html=True)


# ------------------ END ------------------
