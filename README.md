# 👁️ OmniVision Pro

An industry-grade, real-time computer vision pipeline engineered using a thread-safe **Streamlit WebRTC** cloud architecture. OmniVision Pro executes edge-detection, geometric biometric calculations, and neural network graphs across a decentralized background-thread pipeline, enabling clean, low-latency video streaming directly inside cloud sandboxes.

---

## 🛠️ High-Performance Tech Stack

The application is built on a modular engineering stack, separating core artificial intelligence models, real-time data streaming layers, and UI presentation components.

### 🐍 Core Language Tier
* **Python (v3.9 - v3.11):** Serves as the primary runtime environment, coordinating asynchronous background processing threads and state machine parameters.

### 🧠 Machine Learning & AI Core
* **Google MediaPipe Tasks API:** Powers the sub-millisecond facial topology mapping and hand landmark structural tracking.
* **TensorFlow Lite (TFLite Runtime):** Coordinates light, edge-optimized convolutional neural network inferences for spatial target tracking.

### 👁️ Computer Vision & Data Math
* **OpenCV (Headless Edition):** Handles dynamic BGR-to-RGB color matrix transformations, live morphological filter pipelines, and multi-line overlay drawing logic.
* **NumPy:** Drives vector mathematics, scale-invariant spatial distance calculations, and fast multi-dimensional matrix operations for incoming webcam frames.

### 📡 Real-Time WebRTC Streaming Layer
* **Streamlit-WebRTC:** Manages browser-to-server media transmission pipelines via low-latency STUN/ICE connection handshakes.
* **PyAV (av):** Decodes incoming raw browser network packages smoothly into high-performance NumPy arrays for direct algorithmic processing.

### 🎨 UI & Layout Engine
* **Streamlit Core:** Compiles responsive custom CSS layout cards and handles active tracking module session variable configurations.

---

## 🎛️ Dynamic Tracking Modules

| Module Name | Core Tech Architecture | Functional Execution |
| :--- | :--- | :--- |
| **🖐️ Hand Gesture Decoder** | Vector-Angle Dot Product Framework | Calculates joint angle thresholds across vertex nodes to decode 11+ structural hand gestures (Fist, Peace, Thumbs Up, Spiderman, OK, etc.). |
| **👤 Face Shape Analyzer** | Scale-Invariant Biometric Cross-Ratios | Maps cross-sectional spans (forehead, cheekbones, jawline) against vertical facial height lines to classify 6 distinct bone shapes: Oval, Round, Square, Heart, Diamond, and Oblong. |
| **🎭 Face Emotion Engine** | Normalized Geometric Ratios | Gauges subtle mouth stretching, eyebrow displacement, and eyelid opening metrics relative to structural eye baselines to track human expressions instantly. |
| **✍️ Air Canvas** | Local Memory State Path Tracing | Connects tip coordinates of the index finger inside a thread-isolated tracking array (`self.paint_memory`), drawing continuous digital paths without hitting system execution crashes. |
| **🔮 Palm Line Analyzer** | Morphological Segmentation Pipeline | Sequences a Bilateral noise filter, CLAHE contrast mapping, Black-Hat matrix operations, and Canny edge extraction to isolate intricate dermal lines and creases. |
| **🕶️ AR Glasses Overlay** | Geometric Transform Projection | Computes orientation angles and scale boundaries using tracking landmarks to render a floating augmented reality eyewear structure. |
| **🔍 Object Classifier** | Convolutional Deep Neural Net Graph | Feeds localized frames into an *EfficientNet-Lite0* TFLite classification matrix when targets are held within the center alignment reticle. |

---

## 🎨 Dual-Theme Design System

OmniVision Pro includes an ultra-subtle, borderless miniature theme toggle that shifts the interface layout instantly using balanced, non-contrastic custom palettes:

### 🌌 Industrial Stealth Theme (Dark Mode)
* **Application Canvas Background:** `#111111` (Pure Onyx Void)
* **Sidebar System Context Layout:** `#232323` (Matte Carbon Black)
* **Dividers & Panel Borders:** `#343434` (Urban Graphite)
* **Typography Hierarchy:** Primary text rendered in soft matte silver (`#F3F4F6`), secondary details in Dim Grey (`#696969`), and focal terms illuminated via Emerald accents (`#0d6244`).

### 🌿 Organic Meadow Theme (Light Mode)
* **Application Canvas Background:** `#E1F0DA` (Pale Mint Glow)
* **Sidebar System Context Layout:** `#D4E7C5` (Soft Sage Surface)
* **Dividers & Panel Borders:** `#BFD8AF` (Muted Moss Boundaries)
* **Typography Hierarchy:** Uses your base dark slate code (`#27374D`) as the primary text color to guarantee crisp, high-contrast readability under bright lights.

---

## 📦 Core Model Registry

The tracking pipeline automatically secures its verified binary models via cloud storage nodes on startup bypass loops:
* **Hand Landmark Engine:** `hand_landmarker.task` *(MediaPipe Float16 Vector Topology)*
* **Face Mesh Engine:** `face_landmarker.task` *(468-Point Structural Coordinate Mesh)*
* **Image Classification Engine:** `efficientnet_lite0.tflite` *(EfficientNet-Lite0 Convolutions)*

---

## 💻 Localhost Installation

Get the application running locally on your machine with these terminal commands:

```bash
# 1. Clone your project workspace repository
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME

# 2. Spin up your local Python isolation capsule
python3 -m venv venv
source venv/bin/activate

# 3. Inject the production dependency matrix 
pip install -r requirements.txt

# 4. Initialize the Streamlit execution loop
streamlit run app_2.py