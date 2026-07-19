# ============================================================
# OmniVision — Flask Backend
# Real-time object classification from webcam frames using
# MobileNetV2 pretrained on ImageNet (1000 everyday object classes).
# ============================================================
import os
import base64
import io
import streamlit as st
import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    decode_predictions,
    preprocess_input,
)

app = Flask(__name__)

IMG_SIZE = (224, 224)
TOP_K = 5

print("Loading MobileNetV2 (ImageNet weights)...")
model = MobileNetV2(weights="imagenet")
print("Model ready — 1000 ImageNet classes.")


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


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": "MobileNetV2-ImageNet", "classes": 1000})


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        if not data or "image" not in data:
            return jsonify({"error": "No image field in request body"}), 400

        img_b64 = data["image"]
        if "," in img_b64:
            img_b64 = img_b64.split(",", 1)[1]

        img_bytes = base64.b64decode(img_b64)
        img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_arr = preprocess_webcam_image(img_pil)

        top_predictions = run_inference(img_arr)
        best = top_predictions[0]

        return jsonify(
            {
                "prediction": best["class"],
                "confidence": best["confidence"],
                "top3": top_predictions[:3],
                "top5": top_predictions,
            }
        )

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


def find_free_port(preferred: int = 8080, attempts: int = 10) -> int:
    """Return preferred port, or the next free one if it is already in use."""
    import socket

    for offset in range(attempts):
        port = preferred + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found starting at {preferred}")


st.title("OmniVision App Live!")
st.write("If you see this, the interface is working perfectly.")