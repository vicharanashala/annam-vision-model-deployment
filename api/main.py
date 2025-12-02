import os
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from PIL import Image
import uvicorn
from model_loader import load_dino_model, preprocess_pil_image, load_kg

# Paths
WEIGHTS_PATH = "/home/aic_u3/aic_u3/ComputerVision/App_Leaf_disease_detector/model_weights/best_Potato_Tomato_L.pt"
MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"
KG_PATH = "/home/aic_u3/aic_u3/ComputerVision/App_Leaf_disease_detector/api/knowledge_graph.json"

app = FastAPI(title="Plant Disease Detector API")

# Redirect root → /docs
@app.get("/")
async def root():
    return RedirectResponse(url="/docs")

# CORS
origins = [
    os.getenv("FRONTEND_ORIGIN", "*")
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model + KG
print("Loading model... this may take a while")
model, class_names = load_dino_model(WEIGHTS_PATH, MODEL_NAME)
kg = load_kg(KG_PATH)
print("Model loaded successfully!")

# Health check
@app.get("/health")
async def health():
    return {"status": "API is running"}

# Prediction API
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Only JPG/PNG images are supported")

    try:
        pil_img = Image.open(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    x = preprocess_pil_image(pil_img)
    THRESHOLD = 0.7

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(probs.argmax())
        confidence = float(probs[pred_idx])

        # Low-confidence → return Unknown
        if confidence < THRESHOLD:
            return {
                "class_key": "Unknown",
                "plant": "Unknown",
                "disease_name": "Unknown or Not a Plant Disease",
                "symptom": "Unable to determine disease from image",
                "treatment": "Please upload a clear leaf image of Tomato or Potato plant",
                "confidence": confidence
            }

    # High-confidence → return real prediction
    cls_key = class_names[pred_idx]
    kg_info = kg.get(cls_key, {})

    return {
        "class_key": cls_key,
        "plant": kg_info.get("plant", "N/A"),
        "disease_name": kg_info.get("disease_name", cls_key),
        "symptom": kg_info.get("symptom", "N/A"),
        "treatment": kg_info.get("treatment", "N/A"),
        "confidence": confidence,
    }

# Run App
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
