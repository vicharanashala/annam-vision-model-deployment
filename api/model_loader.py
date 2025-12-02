# ###########################################################
import os
import json
from PIL import Image
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from torchvision import transforms

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------
# DINO Classifier wrapper
# -------------------------------
class DinoClassifier(nn.Module):
    def __init__(self, base_model, hidden_size, num_classes):
        super().__init__()
        self.base = base_model
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        outputs = self.base(x)
        pooled = outputs.last_hidden_state.mean(dim=1)
        return self.classifier(pooled)


# -------------------------------
# Image transforms used at inference
# -------------------------------
val_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -------------------------------
# Load knowledge graph
# -------------------------------
def load_kg(path: str):
    with open(path, "r") as f:
        return json.load(f)


# -------------------------------
# Load model + checkpoint
# -------------------------------
def load_dino_model(weight_path: str, model_name: str):
    """
    Loads finetuned DINOv3 model and class labels.
    Returns model, class_names (list)
    """
    # Load base DINO
    config = AutoConfig.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name, config=config)

    # Load checkpoint
    checkpoint = torch.load(weight_path, map_location=device)

    # Extract classes
    class_names = checkpoint.get("classes")
    if class_names is None:
        raise KeyError(
            "Checkpoint missing key 'classes'. Ensure your training script saved it."
        )

    # Build full classifier
    model = DinoClassifier(base_model, config.hidden_size, len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, class_names


# -------------------------------
# Preprocess PIL image
# -------------------------------
def preprocess_pil_image(pil_img: Image.Image):
    img = pil_img.convert("RGB")
    x = val_tfms(img).unsqueeze(0).to(device)
    return x
