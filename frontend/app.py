# enabled ngrok

import streamlit as st
import datetime
import io
import requests
import sys
import os
import json
from urllib.parse import urlencode
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------
BACKEND_PATH = "/home/aic_u3/aic_u3/ComputerVision/App_Leaf_disease_detector/api"
sys.path.append(BACKEND_PATH)

# API_URL = "http://127.0.0.1:8080/api/predict" # for local machine
API_URL = "https://adolph-unmeteorologic-sheepishly.ngrok-free.dev/api/predict" # for public url

DEADLINE = datetime.date(2025, 12, 31)

CLIENT_SECRET_PATH = "/home/aic_u3/aic_u3/ComputerVision/App_Leaf_disease_detector/frontend/client_secret.json"

# Load Google OAuth credentials
with open(CLIENT_SECRET_PATH) as f:
    data = json.load(f)

CLIENT_ID = data["web"]["client_id"]
CLIENT_SECRET = data["web"]["client_secret"]
REDIRECT_URI = data["web"]["redirect_uris"][0]

# st.write("DEBUG REDIRECT URI:", REDIRECT_URI)

GOOGLE_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile"
]

# ----------------------------------------------------------
# STREAMLIT PAGE
# ----------------------------------------------------------
st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿")

# ----------------------------------------------------------
# Helper: Build Google OAuth Login URL
# ----------------------------------------------------------
def build_google_login_url():
    params = {
        "client_id": CLIENT_ID,
        "response_type": "code",
        "scope": " ".join(GOOGLE_SCOPES),
        "redirect_uri": REDIRECT_URI,
        "access_type": "offline",
        "prompt": "consent"
    }
    return "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params)

# ----------------------------------------------------------
# Helper: Exchange Code → Token
# ----------------------------------------------------------
def exchange_code_for_token(code):
    token_url = "https://oauth2.googleapis.com/token"
    data = {
        "code": code,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "redirect_uri": REDIRECT_URI,
        "grant_type": "authorization_code"
    }
    try:
        r = requests.post(token_url, data=data)
        st.write("DEBUG TOKEN RESPONSE STATUS:", r.status_code)
        st.write("DEBUG TOKEN RESPONSE:", r.text)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Token exchange failed: {e}")
        return None

# ----------------------------------------------------------
# SESSION STATE INIT
# ----------------------------------------------------------
if "user" not in st.session_state:
    st.session_state["user"] = None
if "auth_code_exchanged" not in st.session_state:
    st.session_state.auth_code_exchanged = False

# ----------------------------------------------------------
# Google Callback Handler
# ----------------------------------------------------------
query_params = st.query_params
if "code" in query_params and not st.session_state.auth_code_exchanged:
    code = query_params["code"]
    token_data = exchange_code_for_token(code)
    st.session_state.auth_code_exchanged = True
    if token_data and "id_token" in token_data:
        try:
            id_info = id_token.verify_oauth2_token(token_data["id_token"], google_requests.Request(), CLIENT_ID)
            st.session_state["user"] = {
                "email": id_info.get("email"),
                "name": id_info.get("name"),
                "id_token": token_data["id_token"]
            }
            st.success(f"Google Login Successful! 🎉 Logged in as {id_info.get('email')}")
            st.rerun()
        except Exception as e:
            st.error(f"ID token verification failed: {e}")

# ----------------------------------------------------------
# HEADER
# ----------------------------------------------------------
st.markdown("<h1 style='text-align: center; color: #2E7D32;'>🌿 Plant Disease Detection App</h1>", unsafe_allow_html=True)
st.markdown("---")

# ----------------------------------------------------------
# LOGIN SECTION
# ----------------------------------------------------------
st.markdown("### 🔐 Sign in")
col1, col2 = st.columns([1, 3])

with col1:
    if st.button("Sign in with Google"):
        st.markdown(f"[Click here to continue →]({build_google_login_url()})")

with col2:
    if st.session_state["user"]:
        st.success(f"Logged in as {st.session_state['user']['email']}")
        if st.button("Logout"):
            st.session_state["user"] = None
            st.session_state.auth_code_exchanged = False
            st.rerun()
    else:
        st.info("Not logged in.")

if st.session_state["user"] is None:
    st.warning("Please sign in with Google to continue.")
    st.stop()

# ----------------------------------------------------------
# DEADLINE CHECK
# ----------------------------------------------------------
if datetime.date.today() > DEADLINE:
    st.error("❌ Registration closed.")
    st.stop()

# ----------------------------------------------------------
# IMAGE UPLOAD + CAMERA
# ----------------------------------------------------------
st.markdown("### 📤 Upload a Plant Leaf Image")

if "camera_mode" not in st.session_state:
    st.session_state.camera_mode = False

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

with col2:
    st.write("")
    st.write("")
    if not st.session_state.camera_mode:
        if st.button("📷 Enable Camera"):
            st.session_state.camera_mode = True
    else:
        if st.button("🚫 Disable Camera"):
            st.session_state.camera_mode = False

camera_img = None
if st.session_state.camera_mode:
    camera_img = st.camera_input("Take a picture")

file_obj = camera_img if camera_img is not None else uploaded_file
if not file_obj:
    st.warning("Please upload an image or use camera.")
    st.stop()

st.markdown("---")

# ----------------------------------------------------------
# PREDICT BUTTON
# ----------------------------------------------------------
if st.button("🔍 Predict Disease", use_container_width=True):
    st.session_state.camera_mode = False
    bytes_data = file_obj.getvalue()

    files = {"file": ("upload.jpg", io.BytesIO(bytes_data), "image/jpeg")}
    headers = {"Authorization": f"Bearer {st.session_state['user']['id_token']}"}

    with st.spinner("Analyzing..."):
        try:
            resp = requests.post(API_URL, files=files, headers=headers)
            resp.raise_for_status()
            result = resp.json()

            st.success("Analysis Complete")
            st.write("🌿 Plant:", result.get("plant"))
            st.write("🦠 Disease:", result.get("disease_name"))
            st.write("📊 Confidence:", f"{result.get('confidence'):.2f}")
            st.write("📝 Symptoms:", result.get("symptom"))
            st.write("💊 Treatment:", result.get("treatment"))
            st.image(bytes_data)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
