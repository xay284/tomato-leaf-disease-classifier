import json
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

st.set_page_config(page_title="Tomato leaf disease classifier", layout="centered")

classes = json.loads(Path("models/classes.json").read_text())
labels = [c.replace("Tomato___", "").replace("_", " ") for c in classes]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@st.cache_resource
def load_model():
    model = models.mobilenet_v3_large(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
    state = torch.load("models/mobilenetv3_finetuned.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


model = load_model()

st.title("Tomato leaf disease classifier")
st.caption(
    "MobileNetV3 fine-tuned on 6 tomato classes of the PlantVillage dataset. "
    "Upload a photo of a single tomato leaf."
)

file = st.file_uploader("Upload a tomato leaf photo", type=["jpg", "jpeg", "png"])
if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded image")
    with torch.no_grad():
        probs = torch.softmax(model(transform(img).unsqueeze(0)), dim=1)[0]
    top = probs.argmax().item()
    st.subheader(f"Prediction: {labels[top]} ({probs[top]:.1%})")
    if probs[top] < 0.7:
        st.warning("Low confidence: this image may be unlike the training data.")
    st.bar_chart(pd.DataFrame({"probability": probs.numpy()}, index=labels))

st.info(
    "Limits: trained only on lab-style PlantVillage photos (single leaf, plain background) "
    "and only 6 tomato classes. The model always answers with one of those 6 classes, even "
    "for an image that is not a tomato leaf. This is a portfolio demo, not a diagnostic tool."
)