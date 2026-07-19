import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    decode_predictions,
    preprocess_input,
)

# 1. Page Configuration
st.set_page_config(page_title="OmniVision AI", layout="centered", page_icon="📸")

IMG_SIZE = (224, 224)
TOP_K = 5

# 2. Safe Caching for MobileNetV2 (Prevents OOM Crashes)
@st.cache_resource
def load_model():
    return MobileNetV2(weights="imagenet")

model = load_model()

# 3. Your Core Image Processing Functions (Kept Intact)
def preprocess_webcam_image(img_pil: Image.Image) -> np.ndarray:
    """Center-crop to square, resize, and apply MobileNetV2 preprocessing."""
    w, h = img_pil.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img_pil = img_pil.crop((left, top, left + side, top + side))
    img_pil = img_pil.resize(IMG_SIZE, Image.LANCZOS)

    arr = np.array(img_pil, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return preprocess_input(arr)

def run_inference(img_arr: np.ndarray) -> list[dict]:
    preds = model.predict(img_arr, verbose=0)
    decoded = decode_predictions(preds, top=TOP_K)[0]
    return [
        {
            "class": label.replace("_", " ").title(),
            "confidence": round(float(score) * 100, 1),
        }
        for _, label, score in decoded
    ]

# 4. Streamlit User Interface
st.title("📸 OmniVision Deep Learning Hub")
st.success("MobileNetV2 Engine is online and ready!")
st.write("Capture a frame from your webcam below to run real-time classification.")

# Native browser webcam widget (Replaces index.html and Flask POST routines)
cam_image = st.camera_input("Capture an object to analyze")

if cam_image is not None:
    # Open the incoming camera byte stream as a clean PIL image
    img_pil = Image.open(cam_image).convert("RGB")
    
    # Run pipeline with a clean visual loading state
    with st.spinner("Analyzing frame features..."):
        img_arr = preprocess_webcam_image(img_pil)
        top_predictions = run_inference(img_arr)
        
    # Render results smoothly onto the page
    best_match = top_predictions[0]
    st.markdown("---")
    st.subheader(f"🎯 Target Identified: **{best_match['class']}**")
    st.progress(int(best_match['confidence']))
    
    # Render the statistical top-5 breakdown list
    st.write(f"**Confidence Level:** {best_match['confidence']}%")
    
    with st.expander("📊 View Top 5 Alternative Predictions"):
        for idx, pred in enumerate(top_predictions):
            st.write(f"**{idx+1}. {pred['class']}** — {pred['confidence']}% match")