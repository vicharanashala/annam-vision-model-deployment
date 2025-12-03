# Working..without enabling google login 

import streamlit as st
import datetime
import io
import requests
import sys
import os

# ----------------------------------------------------------
# IMPORT GOOGLE AUTH UTIL ONLY WHEN REAL LOGIN IS ENABLED
# ----------------------------------------------------------
BACKEND_PATH = "/home/aic_u3/aic_u3/ComputerVision/App_Leaf_disease_detector/api"
sys.path.append(BACKEND_PATH)

# from oauth_utils import get_user_info_from_google   # ENABLE LATER

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------
# API_URL = "http://10.10.0.12:8001/predict" # public url
API_URL = "http://127.0.0.1:8001/predict"
ALLOWED_DOMAIN = None     # example: "iitropar.ac.in"
deadline = datetime.date(2025, 11, 30)

# ----------------------------------------------------------
# PAGE SETTINGS
# ----------------------------------------------------------
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="centered"
)

# ----------------------------------------------------------
# HEADER
# ----------------------------------------------------------
st.markdown(
    """
    <h1 style='text-align: center; color: #2E7D32;'>🌿 Plant Disease Detection App</h1>
    """,
    unsafe_allow_html=True
)
st.markdown("---")

# ----------------------------------------------------------
# TEMPORARY GOOGLE LOGIN DISABLED (TEST MODE)
# ----------------------------------------------------------
st.info("🔓 **Developer Testing Mode — Google Login Disabled**")
email = "testuser@example.com"
st.success(f"Logged in as: **{email}**")

# ----------------------------------------------------------
# REGISTRATION DEADLINE CHECK
# ----------------------------------------------------------
if datetime.date.today() > deadline:
    st.error("❌ Registration is closed.")
    st.stop()

# ----------------------------------------------------------
# IMAGE UPLOAD SECTION
# ----------------------------------------------------------
st.markdown("### 📤 Upload a Plant Leaf Image")

col1, col2 = st.columns(2)

# Upload from device
with col1:
    uploaded_file = st.file_uploader(
        "Upload from device",
        type=["jpg", "jpeg", "png", "JPG", "PNG"]
    )

# Camera activation
with col2:
    st.write("")
    st.write("")
    open_cam = st.button("📷 Enable Camera")

camera_img = None
if open_cam:
    camera_img = st.camera_input("Take a picture")

# Final file selection
file_obj = uploaded_file if uploaded_file else camera_img

if file_obj is None:
    st.warning("⚠️ Please upload an image or use the camera.")
    st.stop()

st.markdown("---")

# ----------------------------------------------------------
# PREDICT BUTTON
# ----------------------------------------------------------
if st.button("🔍 Predict Disease", use_container_width=True):
    bytes_data = file_obj.getvalue()
    files = {"file": ("upload.jpg", io.BytesIO(bytes_data), "image/jpeg")}

    with st.spinner("⏳ Analyzing image..."):
        try:
            resp = requests.post(API_URL, files=files, timeout=30)
            resp.raise_for_status()
            result = resp.json()

            st.markdown("## 🌱 Prediction Results")

            st.markdown(
                """
                <div style="
                    background-color: #F1F8E9;
                    padding: 20px;
                    border-radius: 12px;
                    border: 1px solid #C5E1A5;">
                """,
                unsafe_allow_html=True
            )

            st.write("**🌿 Plant:**", result.get("plant"))
            st.write("**🦠 Disease:**", result.get("disease_name"))
            st.write("**📊 Confidence:**", f"{result.get('confidence'):.2f}")
            st.write("**📝 Symptoms:**", result.get("symptom"))
            st.write("**💊 Treatment:**", result.get("treatment"))

            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("#### 📸 Uploaded Image")
            st.image(bytes_data, use_container_width=True)

        except Exception as e:
            st.error(f"❌ API Error: {e}")
