# ============================================================
# OmniVision — HuggingFace Spaces (Gradio) Version
# Deploy on: https://huggingface.co/spaces
# pip install gradio tensorflow Pillow
# ============================================================

import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
IMG_SIZE    = (32, 32)   # match your training size

print("Loading model...")
model = tf.keras.models.load_model('object_classifier.h5')
model.predict(np.zeros((1, 32, 32, 3)))   # warm-up
print("Model ready.")


def classify(img):
    """Receives a PIL Image from Gradio webcam, returns label dict."""
    if img is None:
        return {}
    img_resized = img.resize(IMG_SIZE, Image.LANCZOS)
    arr  = np.array(img_resized, dtype='float32') / 255.0
    arr  = np.expand_dims(arr, axis=0)
    pred = model.predict(arr, verbose=0)[0]
    return {CLASS_NAMES[i]: float(pred[i]) for i in range(len(CLASS_NAMES))}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(sources=["webcam"], streaming=True, type="pil"),
    outputs=gr.Label(num_top_classes=3, label="Prediction"),
    title="👁️ OmniVision — Live Object Classifier",
    description="Point your webcam at any object to classify it in real time.",
    live=True,
    flagging_mode="never",
)

demo.launch()
