# Plant Disease Detector (DINOv3-large)


## Overview
This repo contains a FastAPI backend and a Streamlit frontend. The backend loads your fine-tuned DINOv3-large weights and exposes a `/predict` endpoint. The frontend provides Google login (demo token flow), camera/file upload, and displays predictions.


## Local run (backend)
1. Place your weights in `model_weights/best_Potato_Tomato_L.pt`
2. Create a virtualenv and install dependencies:


```bash
cd api
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt