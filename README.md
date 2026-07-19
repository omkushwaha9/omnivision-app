# 📸 OmniVision Deep Learning Hub

[![Streamlit App](https://static.streamlit.io/badge_svg.svg)](https://omnivision-app.streamlit.app/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/omkushwaha9/omnivision-app)
[![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.12-green?logo=python)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![Engine](https://img.shields.io/badge/Engine-TensorFlow%20%7C%20Keras-FF6F00?logo=tensorflow)](https://tensorflow.org/)

OmniVision Deep Learning Hub is a production-grade, highly optimized computer vision web application designed to run real-time object classification models natively inside standard web browsers. By leveraging a deep convolutional neural network (**MobileNetV2**) pre-trained on the comprehensive 1,000-class ImageNet dataset, the application delivers near-instantaneous target identification and granular statistical breakdowns directly from a user's live webcam stream.

The system is custom-engineered to bypass common cloud environment bottlenecks. It implements microsecond-optimized image transformation pipelines, thread-safe memory boundary caching (`@st.cache_resource`), and custom headless operating system layer bindings (`packages.txt`) to achieve maximum stability within strict cloud resource limits (such as Streamlit's 1 GB RAM tier).

---

## 🚀 Live Access & Developer Channels

* **Production Live URL:** [omnivision-app.streamlit.app](https://omnivision-app.streamlit.app/)
* **Source Code Repository:** [github.com/omkushwaha9/omnivision-app](https://github.com/omkushwaha9/omnivision-app)
* **Official Developer Portfolio:** [omkushwaha.in](https://omkushwaha.in/)

---

## ✨ Core Core Capabilities & Architecture

* **Zero-Plugin Web Camera Handshake:** Utilizes modern browser WebRTC components through native Streamlit hardware hooks (`st.camera_input`). This ensures immediate image capturing without requiring the client to install desktop wrappers or external libraries.
* **Deterministic Input Normalization:** Automatically pipes frames into a multi-stage preprocessing sequence:
  * *Symmetric Center-Cropping:* Extracts a balanced square frame from any incoming rectangular aspect ratio to mitigate spatial distortion.
  * *Lanczos Interpolation:* Resizes images down to a precise `224x224` pixel boundary to align with the neural network's strict input shape.
  * *Channel Normalization:* Scales and maps pixel matrices between standard `[-1, 1]` constraints via MobileNetV2 mathematical utilities.
* **Thread-Safe Resource Boundaries:** The heavy pre-trained weights file (`~14.5 MB`) and its underlying graph are isolated inside an explicit operational cache. This stops the server from re-downloading or re-parsing the architecture on subsequent run cycles, preventing common memory leaks.
* **Granular Visual Telemetry:** Displays classification updates instantly alongside a linear performance confidence loader bar, primary classifications, and an expanding drop-down window showing a Top-5 alternative probability array.

---

## 🛠️ The Complete Tech Stack Architecture

The OmniVision infrastructure is isolated into three core architectural layers:

### 1. Client UI & Asynchronous Presentation Layer
* **Streamlit Core Engine:** Drives the interactive layout engine. It runs a single-threaded execution loop that repaints layout states dynamically whenever a user snaps a photo.
* **Pillow (PIL):** Powers advanced high-level binary manipulations, maintaining exact data integrity when mapping raw capture arrays between channels and scales.

### 2. Deep Learning Frameworks & Mathematics Engine
* **TensorFlow CPU (`tensorflow-cpu`):** Loaded with the foundational MobileNetV2 architecture. By deploying the CPU-only variant, the application skips large GPU execution frameworks, drastically lowering system deployment times and memory usage.
* **Keras Applications API:** Provides pre-compiled ImageNet classification layers, exposing accurate mathematical inference across 1,000 everyday object classes.
* **NumPy:** Manages raw pixel grid transformations, tensor additions, and multi-dimensional float matrix processing.

### 3. Headless Operating System Adaptation (`packages.txt`)
Because cloud application engines execute inside headless Linux distribution containers (Debian Linux instances), standard desktop graphics display layers are entirely missing. OmniVision passes core system dependencies directly to the underlying Linux engine during boot:
* `libgl1`: Implements key system-level Open Graphics Library (OpenGL) execution wrappers, enabling headless OpenCV and image processing tools to compile without throwing system core errors.

---

## 📊 Application Code Blueprint (`app.py`)

The operational core of the application is clean, streamlined, and free of blocking network loops. Below is the full implementation architecture deployed inside `app.py`:

```python
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    decode_predictions,
    preprocess_input,
)

# 1. Page Configuration & Brand Customization
st.set_page_config(page_title="OmniVision AI", layout="centered", page_icon="📸")

IMG_SIZE = (224, 224)
TOP_K = 5

# 2. Thread-Safe Model Cache Caching Boundary (Prevents OOM Crashes)
@st.cache_resource
def load_model():
    return MobileNetV2(weights="imagenet")

model = load_model()

# 3. Deterministic Image Preprocessing Pipeline
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

# 4. Neural Network Inference Sequence
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

# 5. Presentation Layer Layout
st.title("📸 OmniVision Deep Learning Hub")
st.success("MobileNetV2 Engine is online and ready!")
st.write("Capture a frame from your webcam below to run real-time classification.")

# Native browser webcam hardware integration widget
cam_image = st.camera_input("Capture an object to analyze")

if cam_image is not None:
    # Transpile byte stream directly into a clean PIL image structure
    img_pil = Image.open(cam_image).convert("RGB")
    
    # Run processing with visible loading state telemetry
    with st.spinner("Analyzing frame features..."):
        img_arr = preprocess_webcam_image(img_pil)
        top_predictions = run_inference(img_arr)
        
    best_match = top_predictions[0]
    st.markdown("---")
    st.subheader(f"🎯 Target Identified: **{best_match['class']}**")
    st.progress(int(best_match['confidence']))
    st.write(f"**Confidence Level:** {best_match['confidence']}%")
    
    # Detailed Probability Expanders
    with st.expander("📊 View Top 5 Alternative Predictions"):
        for idx, pred in enumerate(top_predictions):
            st.write(f"**{idx+1}. {pred['class']}** — {pred['confidence']}% match")
```

---

## 🛠️ Step-by-Step Local Environment Installation

Follow this complete roadmap to spin up a fully isolated, mirrored development server on your personal machine:

### 1. System Requirements Check
Make sure you have **Python 3.11** or **Python 3.12** configured on your engine.

### 2. Clone the Remote Repository
Open a terminal workspace on your local desktop and pull down the project assets:
```bash
git clone https://github.com/omkushwaha9/omnivision-app.git
cd omnivision-app
```

### 3. Initialize a Strict Virtual Environment Scope
Keep your system libraries safe by isolating the project dependencies inside a local tracking folder:
```bash
# For macOS / Linux Systems
python3 -m venv venv
source venv/bin/activate

# For Windows Systems
python -m venv venv
venv\Scripts\activate
```
*(Verify activation: You will notice an explicit `(venv)` tag prefix at the start of your terminal line.)*

### 4. Upgrade Package Tools & Install Components
Run the package installer to assemble your frameworks directly from the verified requirements manifest:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Boot Up the Native Development Server
Launch the active server runtime environment from the terminal workspace:
```bash
streamlit run app.py
```
The framework will immediately initialize your local thread loops and automatically launch the project layout inside your browser at `http://localhost:8501`.

---

## 🌐 Production Deployment Architecture

When hosting this platform on **Streamlit Community Cloud**, keep these production configuration targets in mind for optimal performance:

1. **Python Environment Selection:** Ensure the app settings are locked into **Python 3.12** or **3.11**. Do not use Python 3.13+ or 3.14+, as they do not yet have stable, pre-compiled wheels for TensorFlow CPU binaries.
2. **System Layer Hooks:** The deployment dashboard reads the presence of the `packages.txt` file automatically. It executes an initial `apt-get install` sequence to spin up the required Linux graphics adapters (`libgl1`) before starting the pip package allocations.
3. **Continuous Deployment Lifecycle:** The main branch is wired directly to production via standard GitHub webhooks. Any future pushes to the `main` branch trigger a zero-downtime, rolling hot-reloading loop on the live servers instantly.

---

## 👨‍💻 Engineering Profile

**Om Kushwaha**
* **Personal Developer Portfolio:** [omkushwaha.in](https://omkushwaha.in/)
* **GitHub Repository Workspace:** [@omkushwaha9](https://github.com/omkushwaha9)

---
*Developed with precision as an exploration into highly streamlined, cloud-native deep learning deployments.*
