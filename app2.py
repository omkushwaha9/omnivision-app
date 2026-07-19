"""
OmniVision Pro — Industry-Grade Real-Time Computer Vision Pipeline (Cloud Optimized)
Architecture: MediaPipe Tasks API + OpenCV + Streamlit WebRTC
Accuracy Design: Vector-angle finger detection, EMA temporal smoothing,
                 multi-frame gesture voting, normalized face geometry.
"""

import os, ssl, time, collections
import urllib.request
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import av  # NEW: Handles cloud video frame decoding[cite: 3]
from streamlit_webrtc import webrtc_streamer, WebRtcMode 
ctx = None
# 1. Create a stable memory registry to hold the active UI selection
class CloudPipelineRegistry:
    active_mode = "hand"

# 2. This function address stays perfectly static across all page reruns
def stable_processor_factory():
    return OmniVisionCloudProcessor(mode=CloudPipelineRegistry.active_mode)


# ─────────────────────────────────────────────────────────────────────────────
# SSL bypass for verified Google Storage downloads
# ─────────────────────────────────────────────────────────────────────────────
ssl._create_default_https_context = ssl._create_unverified_context

# ─────────────────────────────────────────────────────────────────────────────
# MODEL REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
MODELS = {
    "hand_landmarker.task":   "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "face_landmarker.task":   "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "efficientnet_lite0.tflite": "https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite0/float32/1/efficientnet_lite0.tflite",
}

# ─────────────────────────────────────────────────────────────────────────────
# CORE MATH UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def vec_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Angle at vertex B formed by rays B→A and B→C.
    Returns degrees [0, 180].
    Uses dot-product — rotation-invariant, scale-invariant.
    """
    ba = a - b
    bc = c - b
    cos_theta = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))

def ema_smooth(prev: np.ndarray, curr: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Exponential Moving Average: blends previous and current frame landmarks."""
    return alpha * curr + (1.0 - alpha) * prev

def landmark_array(landmarks, w: int, h: int) -> np.ndarray:
    """Convert MediaPipe NormalizedLandmarkList → float32 (N, 2) pixel array."""
    return np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)

def analyze_face_shape(pts: np.ndarray) -> str:
    """
    Computes scale-invariant structural geometry alignments to determine face shape type.
    Uses facial width-to-height ratios and cross-sectional distance thresholds.
    """
    # Core structural metric lines (distances)
    cheek_w = np.linalg.norm(pts[234] - pts[454]) + 1e-8            # Outermost cheekbone width
    face_h  = np.linalg.norm(pts[10] - pts[152]) / cheek_w          # Vertical face height ratio
    forehead_w = np.linalg.norm(pts[54] - pts[284]) / cheek_w       # Forehead width ratio
    jaw_w   = np.linalg.norm(pts[172] - pts[397]) / cheek_w         # Lower jawline width ratio

    # Geometric rule classification tree
    if face_h > 1.38:
        return "Oblong / Rectangle Shape ⏳"
    elif forehead_w > 0.95 and jaw_w < 0.75:
        return "Heart Shape 💖"
    elif cheek_w > (np.linalg.norm(pts[54] - pts[284]) * 1.15) and jaw_w < 0.72:
        return "Diamond Shape 💎"
    elif face_h < 1.14:
        if jaw_w > 0.82:
            return "Square Structural Shape 🔲"
        else:
            return "Round Symmetry Shape ⚪"
    else:
        if jaw_w > 0.82:
            return "Square Structural Shape 🔲"
        else:
            return "Oval Structural Shape 🥚"

# ─────────────────────────────────────────────────────────────────────────────
# ACCURATE FINGER DETECTION — Vector-Angle Method
# ─────────────────────────────────────────────────────────────────────────────
FINGER_JOINTS = {
    "thumb":  (1, 2, 4),
    "index":  (5, 6, 8),
    "middle": (9, 10, 12),
    "ring":   (13, 14, 16),
    "pinky":  (17, 18, 20),
}
EXTENDED_THRESH = 160.0  # degrees — finger is "open" if angle > this


def detect_fingers(pts: np.ndarray) -> dict:
    """Returns dict of {finger_name: bool(extended)} using vector angles."""
    state = {}
    for name, (a, b, c) in FINGER_JOINTS.items():
        angle = vec_angle(pts[a], pts[b], pts[c])
        thresh = 140.0 if name == "thumb" else EXTENDED_THRESH
        state[name] = angle > thresh
    return state


def classify_gesture(fingers: dict, pts: np.ndarray) -> tuple[str, str]:
    """Returns (gesture_label, emoji) based on finger state with scale-invariant checks."""
    t, i, m, r, p = fingers["thumb"], fingers["index"], fingers["middle"], fingers["ring"], fingers["pinky"]
    count = sum([t, i, m, r, p])

    ok_dist = np.linalg.norm(pts[4] - pts[8])
    palm_ref = np.linalg.norm(pts[0] - pts[9]) + 1e-8
    ok_ratio = ok_dist / palm_ref

    if ok_ratio < 0.35 and m and r and p:
        return "OK / Excellent", "👌"

    patterns = {
        (False, False, False, False, False): ("Fist / Zero",        "✊"),
        (False, True,  False, False, False): ("One / Pointing",     "☝️"),
        (False, True,  True,  False, False): ("Two / Peace / V",    "✌️"),
        (False, True,  True,  True,  False): ("Three",              "3️⃣"),
        (False, True,  True,  True,  True):  ("Four",               "4️⃣"),
        (True,  True,  True,  True,  True):  ("Five / Open Hand",   "🖐️"),
        (True,  False, False, False, True):  ("Rock On / Horns",    "🤘"),
        (True,  False, False, False, False): ("Thumbs Up",          "👍"),
        (False, False, False, False, True):  ("Pinky Up",           "🤙"),
        (True,  True,  False, False, False): ("Gun / L-shape",      "👆"),
        (False, True,  False, False, True):  ("Spiderman Sign",     "🕷️"),
    }
    key = (t, i, m, r, p)
    label, emoji = patterns.get(key, (f"Custom [{count} fingers]", "🤚"))
    return label, emoji

# ─────────────────────────────────────────────────────────────────────────────
# ACCURATE FACE EMOTION — Normalized Geometric Ratios
# ─────────────────────────────────────────────────────────────────────────────
def analyze_emotion(pts: np.ndarray) -> tuple[str, str]:
    """Computes scale-invariant facial geometry ratios for emotion classification."""
    iod = np.linalg.norm(pts[33] - pts[263]) + 1e-8   

    mouth_w  = np.linalg.norm(pts[61]  - pts[291]) / iod   
    mouth_h  = np.linalg.norm(pts[13]  - pts[14])  / iod   

    left_brow_raise  = (pts[55][1]  - pts[33][1])  / iod   
    right_brow_raise = (pts[285][1] - pts[263][1]) / iod
    avg_brow_raise   = (left_brow_raise + right_brow_raise) / 2.0

    left_eye_h  = abs(pts[159][1] - pts[145][1]) / iod
    right_eye_h = abs(pts[386][1] - pts[374][1]) / iod
    avg_eye_h   = (left_eye_h + right_eye_h) / 2.0

    if mouth_h > 0.18 and avg_eye_h > 0.14:
        return "Surprised 😲", "#FACC15"
    elif avg_eye_h > 0.18:
        return "Wide-Eyed 👀", "#818CF8"
    elif mouth_w > 0.75 and mouth_h < 0.12:
        return "Happy / Smiling 😊", "#34D399"
    elif avg_brow_raise < -0.10 and mouth_w < 0.60:
        return "Angry 😡", "#F87171"
    elif mouth_w < 0.58 and mouth_h < 0.07:
        return "Sad / Frowning 😔", "#60A5FA"
    elif mouth_h > 0.13:
        return "Open Mouth 😮", "#FB923C"
    else:
        return "Neutral 😐", "#94A3B8"

# ─────────────────────────────────────────────────────────────────────────────
# PALM LINE EXTRACTOR — Morphological Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def extract_palm_lines(frame: np.ndarray) -> np.ndarray:
    """Isolates palm crease lines using morphology edge extraction configurations."""
    gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 9, 80, 80)

    clahe    = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(filtered)

    kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)

    _, thresh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges    = cv2.Canny(thresh, 40, 120)

    overlay  = np.zeros_like(frame)
    overlay[edges > 0]  = (0, 200, 255)   
    overlay[thresh > 0] = (0, 100, 180)   

    result = cv2.addWeighted(frame, 0.55, overlay, 0.80, 0)
    return result

# ─────────────────────────────────────────────────────────────────────────────
# DRAWING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)
]

def draw_hand(frame: np.ndarray, pts: np.ndarray, fingers: dict):
    """Draw hand skeleton with color-coded fingers (green=open, red=closed)."""
    finger_colors = {"thumb": (0, 255, 100), "index": (255, 200, 0), "middle": (0, 200, 255), "ring": (200, 0, 255), "pinky": (255, 100, 0)}
    finger_idx_map = {"thumb": [1, 2, 3, 4], "index": [5, 6, 7, 8], "middle": [9, 10, 11, 12], "ring": [13, 14, 15, 16], "pinky": [17, 18, 19, 20]}

    lm_colors = {}
    for fname, idxs in finger_idx_map.items():
        color = finger_colors[fname] if fingers.get(fname) else (80, 80, 80)
        for idx in idxs: lm_colors[idx] = color
    lm_colors[0] = (200, 200, 200)  

    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), lm_colors.get(a, (120, 120, 120)), 2, cv2.LINE_AA)
    for i, pt in enumerate(pts):
        cv2.circle(frame, tuple(pt.astype(int)), 5, lm_colors.get(i, (120, 120, 120)), cv2.FILLED)
        cv2.circle(frame, tuple(pt.astype(int)), 5, (255, 255, 255), 1, cv2.LINE_AA)

def draw_face_mesh_minimal(frame: np.ndarray, pts: np.ndarray, color=(0, 220, 255)):
    """Draw a minimal face mesh (key landmarks only) for clean display."""
    KEY_LANDMARKS = [33, 263, 61, 291, 13, 14, 55, 285, 1, 4, 152, 10, 234, 454, 159, 145, 386, 374]
    for idx in KEY_LANDMARKS:
        if 0 <= idx < len(pts): cv2.circle(frame, tuple(pts[idx].astype(int)), 3, color, cv2.FILLED)


# 🎯 MODIFIED: Outputs are drawn strictly inside a sleek bar ABOVE the screen[cite: 3]
def draw_hud_overlay(frame: np.ndarray, label: str, fps: float, inference_ms: float, color=(0, 255, 140)):
    """Draws a professional HUD: Engine metrics at top-right, clear output banner at the BOTTOM."""
    h, w = frame.shape[:2]
    
    # 1. Technical metrics positioned quietly in the top-right corner with a drop-shadow for anti-glare
    fps_str = f"FPS: {fps:.1f}  |  Inference: {inference_ms:.1f}ms"
    (tw, _), _ = cv2.getTextSize(fps_str, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    
    # Shadow layer
    cv2.putText(frame, fps_str, (w - tw - 12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (5, 5, 10), 2, cv2.LINE_AA)
    # Main text layer
    cv2.putText(frame, fps_str, (w - tw - 12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 175, 192), 1, cv2.LINE_AA)

    # 2. Dedicated semi-transparent HUD banner bar rendered at the BOTTOM of the screen
    bar = frame.copy()
    cv2.rectangle(bar, (0, h - 55), (w, h), (10, 14, 22), cv2.FILLED)
    cv2.addWeighted(bar, 0.75, frame, 0.25, 0, frame)

    # 3. Main module readout text cleanly positioned inside the bottom banner bar
    cv2.putText(frame, label, (16, h - 21), cv2.FONT_HERSHEY_DUPLEX, 0.58, color, 1, cv2.LINE_AA)

    # 4. Clean accent divider marking the top edge boundary of the bottom banner bar
    cv2.line(frame, (0, h - 55), (w, h - 55), color, 1)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    for filename, url in MODELS.items():
        if not os.path.exists(filename):
            urllib.request.urlretrieve(url, filename)

    hand_opts = vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path="hand_landmarker.task"), num_hands=2, min_hand_detection_confidence=0.6, min_hand_presence_confidence=0.6, min_tracking_confidence=0.5)
    face_opts = vision.FaceLandmarkerOptions(base_options=python.BaseOptions(model_asset_path="face_landmarker.task"), num_faces=1, min_face_detection_confidence=0.6, min_face_presence_confidence=0.6, min_tracking_confidence=0.5)
    obj_opts = vision.ImageClassifierOptions(base_options=python.BaseOptions(model_asset_path="efficientnet_lite0.tflite"), max_results=3, score_threshold=0.30)
    return vision.HandLandmarker.create_from_options(hand_opts), vision.FaceLandmarker.create_from_options(face_opts), vision.ImageClassifier.create_from_options(obj_opts)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION & STYLING
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="OmniVision Pro", page_icon="👁️", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
.stApp { background: #080C14; }
section[data-testid="stSidebar"] { background: #0D1220; border-right: 1px solid #1E2D45; }
section[data-testid="stSidebar"] .stRadio label { color: #CBD5E1 !important; font-size: 14px; }
.hud-card { background: linear-gradient(135deg, #0D1B2A 0%, #112030 100%); border: 1px solid #1E3A52; border-left: 4px solid #00E5A0; border-radius: 10px; padding: 18px 24px; margin-bottom: 18px; }
.hud-label { color: #64748B; font-size: 11px; letter-spacing: 2px; text-transform: uppercase; font-weight: 600; }
.hud-value { color: #F0FDF4; font-size: 28px; font-weight: 700; margin-top: 6px; line-height: 1.2; }
.hud-sub { color: #94A3B8; font-size: 13px; margin-top: 4px; }
.page-title { color: #F8FAFC; font-size: 26px; font-weight: 800; letter-spacing: -0.5px; }
.page-sub   { color: #475569; font-size: 13px; margin-top: -4px; }
.hud-divider { border: none; border-top: 1px solid #1E2D45; margin: 14px 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "theme": "dark", # 👈 NEW: Defaults to a premium dark base layout
        "paint_pts": [], 
        "gesture_buffer": collections.deque(maxlen=12), 
        "emotion_buffer": collections.deque(maxlen=10), 
        "prev_hand_pts": None, 
        "prev_face_pts": None, 
        "fps_buffer": collections.deque(maxlen=30), 
        "last_gesture": ("—", ""), 
        "last_emotion": ("—", "#94A3B8"), 
        "frame_count": 0
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v
init_state()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
# Create asymmetric horizontal split columns inside the sidebar panel
title_col, toggle_col = st.sidebar.columns([5, 1])

with title_col:
    st.markdown("<h2 style='margin:0; padding:0; font-size:22px;'>🧠 OmniVision Pro</h2>", unsafe_allow_html=True)

with toggle_col:
    theme_icon = "☀️" if st.session_state.theme == "dark" else "🌙"
    # Pinned seamlessly inline with zero external layout distortion
    if st.button(theme_icon, key="theme_toggle_node"):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()

st.sidebar.markdown("---")

MODES = {
    "🖐️  Hand Gesture Decoder": "hand", 
    "🔮  Palm Line Analyzer": "palm", 
    "🎭  Face Emotion Engine": "face", 
    "👤  Face Shape Analyzer": "shape", # 👈 NEW MODULE ADDED
    "✍️   Air Canvas": "canvas", 
    "🕶️   AR Glasses Overlay": "glasses", 
    "🔍  Object Classifier": "object"
}
mode_label = st.sidebar.radio("Active Module", list(MODES.keys()), index=0)
mode       = MODES[mode_label]

st.sidebar.markdown("---")
st.sidebar.markdown("**⚙️ Detection Settings**")
smooth_alpha = st.sidebar.slider("Temporal Smoothing (EMA α)", 0.1, 1.0, 0.5, 0.05)
min_confidence = st.sidebar.slider("Min Detection Confidence", 0.3, 0.95, 0.6, 0.05)

if mode == "canvas":
    brush_color_hex = st.sidebar.color_picker("Brush Color", "#FF00FF")
    brush_r, brush_g, brush_b = int(brush_color_hex[1:3], 16), int(brush_color_hex[3:5], 16), int(brush_color_hex[5:7], 16)
    brush_size = st.sidebar.slider("Brush Size", 2, 20, 7)
    if st.sidebar.button("🗑️ Clear Canvas"): st.session_state.paint_pts = []
else:
    brush_r, brush_g, brush_b, brush_size = 255, 0, 255, 7

st.sidebar.markdown("---")
st.sidebar.caption("OmniVision Pro v2.0 · MediaPipe · OpenCV")

# ─────────────────────────────────────────────────────────────────────────────
# USER-DEFINED DESIGN SYSTEM (ONYX, CARBON BLACK & GRAPHITE MONOCHROME)
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.theme == "dark":
    # Monochrome Dark: #111111 (Onyx Base), #232323 (Carbon Sidebar), #343434 (Graphite Border)
    bg_app = "#111111"
    bg_sidebar = "#232323"
    border_color = "#343434"
    text_main = "#F3F4F6"   # Crisp matte white to guarantee flawless legibility against Onyx
    text_muted = "#696969"  # Dim Grey for secondary system labels
    hud_gradient = "linear-gradient(135deg, #232323 0%, #111111 100%)"
    
    # Text colors inside the module feature description cards
    text_card_body = "#7A7A7A"  # Grey tone for subtle description text
    text_card_bold = "#E5E5E5"  # Balanced light graphite for prominent bold highlights
else:
    # Light Sage Palette: #E1F0DA (Canvas), #D4E7C5 (Sidebar), #BFD8AF (Border), #99BC85 (Accent)
    bg_app = "#E1F0DA"
    bg_sidebar = "#D4E7C5"
    border_color = "#BFD8AF"
    text_main = "#111111"   # Borrowed from your dark Onyx code for perfect readable contrast
    text_muted = "#526D82"  
    hud_gradient = "linear-gradient(135deg, #E1F0DA 0%, #D4E7C5 100%)"
    
    # Rich structural charcoal colors for light theme descriptions
    text_card_body = "#3A4D5E"  
    text_card_bold = "#1A2636"  

st.markdown(f"""
<style>
/* Base Layout Modifications */
.stApp {{ background: {bg_app} !important; transition: background 0.2s ease; }}
h1, h2, h3, p, span, label {{ color: {text_main} !important; }}

/* Sidebar Custom Styling */
section[data-testid="stSidebar"] {{
    background: {bg_sidebar} !important;
    border-right: 1px solid {border_color} !important;
    transition: background 0.2s ease;
}}
section[data-testid="stSidebar"] .stRadio label {{ color: {text_main} !important; font-size: 14px; }}

/* Professional Low-Contrast HUD Info Cards */
.hud-card {{
    background: {hud_gradient} !important;
    border: 1px solid {border_color} !important;
    border-left: 4px solid #00E5A0 !important;
    border-radius: 10px;
    padding: 18px 24px;
    margin-bottom: 18px;
}}
.hud-label {{ color: {text_muted} !important; font-size: 11px; letter-spacing: 2px; text-transform: uppercase; font-weight: 600; }}
.hud-value {{ color: {text_main} !important; font-size: 24px; font-weight: 700; margin-top: 6px; }}
.hud-sub {{ color: {text_muted} !important; font-size: 13px; margin-top: 4px; }}
.page-title {{ color: {text_main} !important; font-size: 26px; font-weight: 800; letter-spacing: -0.5px; }}
.page-sub   {{ color: {text_muted} !important; font-size: 13px; margin-top: -4px; }}
.hud-divider {{ border: none; border-top: 1px solid {border_color}; margin: 14px 0; }}

/* FORCE INLINE CARD DESCRIPTIONS TO BE READABLE ACROSS BOTH THEMES */
.hud-card div[style*="color"], 
.hud-card p,
.hud-card span {{
    color: {text_card_body} !important;
}}
.hud-card div[style*="color"] b,
.hud-card p b {{
    color: {text_card_bold} !important;
    font-weight: 700 !important;
}}

/* ULTRA-SUBTLE MINIATURE THEME TOGGLE OVERRIDES */
div[data-testid="stSidebarUserContent"] div[data-testid="stButton"] button {{
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    box-shadow: none !important;
    width: auto !important;
    height: auto !important;
    min-width: unset !important;
    min-height: unset !important;
    font-size: 17px !important;    
    cursor: pointer !important;
    opacity: 0.45 !important;       
    transition: opacity 0.15s ease !important;
}}
div[data-testid="stSidebarUserContent"] div[data-testid="stButton"] button:hover {{
    opacity: 0.90 !important;       
    background: transparent !important;
}}
div[data-testid="stSidebarUserContent"] div[data-testid="stButton"] button:focus,
div[data-testid="stSidebarUserContent"] div[data-testid="stButton"] button:active {{
    background: transparent !important;
    box-shadow: none !important;
}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
# Place this at the top of your script to keep the interface perfectly clean
st.markdown("""
    <style>
        /* Automatically hides any temporary streaming connection warnings from displaying */
        div[data-testid="stNotification"] {
            display: none !important;
        }
        .element-container:has(iframe) {
            background: transparent !important;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-title">👁️ OmniVision Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Production-Grade Real-Time Computer Vision · Vector-Angle Finger Detection · EMA Temporal Smoothing</div>', unsafe_allow_html=True)
st.markdown("<hr class='hud-divider'>", unsafe_allow_html=True)

col_vid, col_hud = st.columns([3, 2])

with col_vid:
    run = st.checkbox("▶️  Activate Camera Pipeline", value=False)
    viewport = st.empty()

with col_hud:
    hud_main    = st.empty()
    hud_fingers = st.empty()
    hud_metrics = st.empty()
    hud_obj     = st.empty()

# Initialize AI Tasks
hand_engine, face_engine, obj_engine = load_models()

# ─────────────────────────────────────────────────────────────────────────────
# THREAD-SAFE CLOUD VIDEO FILTER ENGINE (FIXED CANVAS MEMORY)
# ─────────────────────────────────────────────────────────────────────────────
class OmniVisionCloudProcessor:
    # Change this line to include a default value like ="hand"
    def __init__(self, mode="hand"): 
        self.mode = mode
        self.t_last = time.time()
        self.paint_memory = [] # ✅ FIXED: Safe local memory block for canvas trails

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        t0 = time.perf_counter()
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) 
        h, w = img.shape[:2]
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        readout_text = "Initializing Selected Pipeline Matrix..."
        hud_color = (0, 229, 160)

        t_now = time.time()
        fps_avg = 1.0 / (t_now - self.t_last + 1e-8)
        self.t_last = t_now

        # ─────────────── MODULE: HAND GESTURE / CANVAS ───────────────
        if self.module_target in ("hand", "canvas"):
            hand_res = hand_engine.detect(mp_image)
            if hand_res.hand_landmarks:
                for hand_idx, lm_list in enumerate(hand_res.hand_landmarks):
                    pts = landmark_array(lm_list, w, h)
                    finger_states = detect_fingers(pts)
                    gesture, emoji = classify_gesture(finger_states, pts)

                    draw_hand(img, pts, finger_states)

                    if self.module_target == "hand":
                        readout_text = f"Gesture: {emoji} {gesture}"

                    elif self.module_target == "canvas":
                        # Drawing rule: Index extended, middle tucked down
                        drawing = finger_states.get("index", False) and not finger_states.get("middle", False)
                        tip = tuple(pts[8].astype(int))
                        if drawing:
                            self.paint_memory.append(tip)  # ✅ FIXED: Append to thread-safe memory
                            readout_text = "✏️ Drawing Mode Active — Extend Index Finger"
                        else:
                            self.paint_memory.append(None) # ✅ FIXED: Insert break in line array
                            readout_text = "✋ Hovering Mode Active — Tuck Fingers to Pause"
            else:
                if self.module_target == "canvas": 
                    self.paint_memory.append(None) # ✅ FIXED: Break path if hand leaves frame
                readout_text = "🔍 Scanning: No hand visible inside canvas boundary"

            # Render persistent painting lines from the thread-safe block
            if self.module_target == "canvas" and len(self.paint_memory) > 1:
                for i in range(1, len(self.paint_memory)):
                    if self.paint_memory[i-1] is not None and self.paint_memory[i] is not None:
                        cv2.line(img, self.paint_memory[i-1], self.paint_memory[i], 
                                 (brush_b, brush_g, brush_r), brush_size, cv2.LINE_AA)

        # ─────────────── MODULE: PALM LINE EXTRACTOR ───────────────
        elif self.module_target == "palm":
            img = extract_palm_lines(img)
            readout_text = "Analysis System: PALM CREASE SEGMENTATION ACTIVE"

        # ─────────────── MODULE: FACE EMOTION / GLASSES ───────────────
        elif self.module_target in ("face", "glasses"):
            face_res = face_engine.detect(mp_image)
            if face_res.face_landmarks:
                pts = landmark_array(face_res.face_landmarks[0], w, h)
                draw_face_mesh_minimal(img, pts)

                if self.module_target == "face":
                    emotion, em_color = analyze_emotion(pts)
                    readout_text = f"Expression Match: {emotion}"
                    hud_color = tuple(int(em_color.lstrip("#")[i:i+2], 16) for i in (4, 2, 0))

                elif self.module_target == "glasses":
                    eye_l, eye_r = pts[33].astype(int), pts[263].astype(int)
                    mid_pt = ((eye_l + eye_r) / 2).astype(int)
                    iod = int(np.linalg.norm(pts[33] - pts[263]))
                    g_w, g_h = int(iod * 1.9), int(iod * 0.55)
                    x1, y1 = mid_pt[0] - (g_w // 2), mid_pt[1] - (g_h // 2)
                    if x1 > 0 and y1 > 0 and (x1 + g_w) < w and (y1 + g_h) < h:
                        cv2.rectangle(img, (x1, y1), (mid_pt[0] - 4, y1 + g_h), (255, 0, 220), 2, cv2.LINE_AA)
                        cv2.rectangle(img, (mid_pt[0] + 4, y1), (x1 + g_w, y1 + g_h), (255, 0, 220), 2, cv2.LINE_AA)
                        cv2.line(img, (mid_pt[0] - 4, mid_pt[1]), (mid_pt[0] + 4, mid_pt[1]), (200, 200, 200), 2)
                    readout_text = "Optics HUD Overlay: Operating Synchronized"
            else:
                readout_text = "🔍 Scanning: Align face profile to register matrix"

# ─────────────── MODULE: FACE SHAPE ANALYZER ───────────────
        elif self.module_target == "shape":
            face_res = face_engine.detect(mp_image)
            if face_res.face_landmarks:
                pts = landmark_array(face_res.face_landmarks[0], w, h)
                draw_face_mesh_minimal(img, pts, color=(0, 255, 150))
                
                # Compute structural matrix
                shape_output = analyze_face_shape(pts)
                readout_text = f"Face Structural Type: {shape_output}"
                hud_color = (255, 200, 0) # Technical Yellow HUD Accent
                
                # Render high-tech geometric structural scanning cross-lines on the viewport
                cv2.line(img, tuple(pts[10].astype(int)), tuple(pts[152].astype(int)), (255, 200, 0), 1, cv2.LINE_AA)   # Height Axis
                cv2.line(img, tuple(pts[54].astype(int)), tuple(pts[284].astype(int)), (0, 255, 255), 1, cv2.LINE_AA)   # Forehead Span
                cv2.line(img, tuple(pts[234].astype(int)), tuple(pts[454].astype(int)), (0, 255, 255), 1, cv2.LINE_AA)  # Cheekbone Axis
                cv2.line(img, tuple(pts[172].astype(int)), tuple(pts[397].astype(int)), (0, 255, 255), 1, cv2.LINE_AA)  # Jawline Span
                
                # Highlight core anchor nodes
                for idx in [10, 152, 54, 284, 234, 454, 172, 397]:
                    cv2.circle(img, tuple(pts[idx].astype(int)), 4, (0, 255, 255), cv2.FILLED)
            else:
                readout_text = "🔍 Scanning: Align profile to process structural metrics"

        # ─────────────── MODULE: OBJECT CLASSIFIER ───────────────
        elif self.module_target == "object":
            obj_res = obj_engine.classify(mp_image)
            cx_, cy_ = w // 2, h // 2
            cv2.rectangle(img, (cx_ - 100, cy_ - 100), (cx_ + 100, cy_ + 100), (0, 165, 255), 2, cv2.LINE_AA)
            if obj_res.classifications and obj_res.classifications[0].categories:
                top = obj_res.classifications[0].categories[0]
                readout_text = f"Object Identified: {top.category_name.replace('_', ' ').title()} ({int(top.score*100)}%)"
            else:
                readout_text = "🔍 Scanning Target Object Textures..."

        # Render metrics smoothly onto the TOP HUD layout bar metrics box
        inference_ms = (time.perf_counter() - t0) * 1000
        draw_hud_overlay(img, readout_text, fps_avg, inference_ms, hud_color)
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
# ─────────────────────────────────────────────────────────────────────────────
# CONTROL INTERFACE & COMPACT VIEWPORT MAPPING[cite: 3]
# ─────────────────────────────────────────────────────────────────────────────
with col_vid:
    # 1. Initialize State Gatekeepers
    if "current_mode" not in st.session_state:
        st.session_state.current_mode = mode
    if "cam_id" not in st.session_state:
        st.session_state.cam_id = 0

    # 2. DETECT MODULE SWITCH: Trigger a safe hardware reset sequence
    if mode != st.session_state.current_mode:
        st.session_state.current_mode = mode
        st.session_state.cam_id += 1        # Increments key to force a clean component rebuild
        st.session_state.is_switching = True # Activates the cooling-off frame
        st.rerun()

    # 3. THE COOLING-OFF FRAME: Gives your Mac OS time to release the webcam lock
    if st.session_state.get("is_switching", False):
        st.session_state.is_switching = False
        st.markdown("""
        <div style="background:#0D1220;border:1px solid #1E3A52;border-radius:12px;padding:40px;text-align:center;margin-top:20px;">
          <div style="margin:0 auto 16px auto;width:40px;height:40px;border:4px solid #1E3A52;border-top-color:#38BDF8;border-radius:50%;animation:spin 1s linear infinite;"></div>
          <div style="color:#F1F5F9;font-size:16px;font-weight:600;">Reconfiguring AI Pipeline Hardware...</div>
          <div style="color:#64748B;font-size:13px;margin-top:4px;">Swapping tracking vectors safely...</div>
        </div>
        <style>@keyframes spin { to { transform: rotate(360deg); } }</style>
        """, unsafe_allow_html=True)
        
        import time
        time.sleep(0.6)  # The magic sweet spot to clear browser hardware contention
        st.rerun()

    # 4. MAIN RENDERING STREAM
    elif not run:
        # Your custom dark HTML offline view
        viewport.markdown("""
        <div style="background:#0D1220;border:1px dashed #1E3A52;border-radius:12px;padding:60px 40px;text-align:center;margin-top:20px;">
          <div style="font-size:48px;margin-bottom:16px;">📷</div>
          <div style="color:#F1F5F9;font-size:18px;font-weight:700;">Camera Pipeline Offline</div>
          <div style="color:#64748B;font-size:14px;margin-top:8px;">Enable the checkbox above to activate cloud processing loop.</div>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        # Render the fresh camera component with a clean versioned tracking key
        ctx = webrtc_streamer(
            key=f"omnivision-cloud-pipeline-v{st.session_state.cam_id}", 
            mode=WebRtcMode.SENDRECV,
            async_processing=True,
            rtc_configuration={
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {"urls": ["stun:stun1.l.google.com:19302"]},
                    {"urls": ["stun:stun.services.mozilla.com"]}
                ]
            },
            media_stream_constraints={
                "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
                "audio": False
            },
            video_processor_factory=lambda: OmniVisionCloudProcessor(mode),
        )

        # 3. Safely update the live background tracking thread variable
        if ctx and ctx.video_processor:
            ctx.video_processor.mode = mode

# Clean placeholders since layouts are cleanly printed onto the top of the stream directly[cite: 3]
# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC SIDEBAR DOCUMENTATION SYSTEM (HUD PANEL POPULATION)
# ─────────────────────────────────────────────────────────────────────────────

# 1. Standard HUD Connectivity Status Card
hud_main.markdown(f"""
<div class="hud-card">
  <div class="hud-label">📡 HUD Sync Status</div>
  <div class="hud-value" style="font-size: 18px; color: ##080F2B;">Core Sync Active</div>
  <div class="hud-sub">Real-time analysis overlays are burning directly onto the top margins of the camera viewport feed matrix.</div>
</div>
""", unsafe_allow_html=True)

# 2. Dynamic Module Feature Matrix Cards
if mode == "hand":
    hud_metrics.markdown("""
    <div class="hud-card" style="border-left: 4px solid #34D399;">
      <div class="hud-label">🖐️ Hand Decoder Engine Guide</div>
      <div style="color: #CBD5E1; font-size: 13px; margin-top: 10px; line-height: 1.7;">
        The vector-angle joint system decodes geometric configurations independently of rotation, tilt, or lens distance:<br><br>
        ✊ <b>Fist / Zero</b> — All internal finger joints folded.<br>
        ☝️ <b>One / Pointing</b> — Index finger extended fully.<br>
        ✌️ <b>Two / Peace / V</b> — Index and Middle fingers open.<br>
        🖐️ <b>Five / Open Hand</b> — All 5 digits extended wide.<br>
        🤘 <b>Rock On / Horns</b> — Index, Pinky, and Thumb open.<br>
        👍 <b>Thumbs Up</b> — Main thumb open, others closed.<br>
        👌 <b>OK / Excellent</b> — Thumb and Index tips close-contact loop.
      </div>
    </div>
    """, unsafe_allow_html=True)

elif mode == "palm":
    hud_metrics.markdown("""
    <div class="hud-card" style="border-left: 4px solid #FACC15;">
      <div class="hud-label">🔮 Palm Crease Computer Graphics Pipeline</div>
      <div style="color: #CBD5E1; font-size: 13px; margin-top: 10px; line-height: 1.7;">
        Processes raw dermal lighting frameworks sequentially to isolate line boundaries:<br><br>
        1️⃣ <b>Bilateral Filter</b> — Removes pixel sensor grain noise while locking hard edges.<br>
        2️⃣ <b>CLAHE Optimization</b> — Adaptive histogram normalization balancing across multiple skin tones.<br>
        3️⃣ <b>Black-Hat Operator</b> — Isolates dark linear features and crease valleys.<br>
        4️⃣ <b>Otsu Threshold</b> — Computes dynamic binarization markers.<br>
        5️⃣ <b>Canny Processing</b> — Compresses vectors into thin amber graphic displays.
      </div>
    </div>
    """, unsafe_allow_html=True)

elif mode == "face":
    hud_metrics.markdown("""
    <div class="hud-card" style="border-left: 4px solid #60A5FA;">
      <div class="hud-label">🎭 Facial Emotion Classification Matrix</div>
      <div style="color: #CBD5E1; font-size: 13px; margin-top: 10px; line-height: 1.7;">
        Evaluates facial matrices by calculating scales against the tracking profile's <b>Inter-Ocular Distance (IOD)</b>:<br><br>
        😲 <b>Surprised</b> — Vertical mouth stretch combined with open eyelid profiles.<br>
        👀 <b>Wide-Eyed</b> — Eyebrow heights clear threshold limits.<br>
        😊 <b>Happy / Smiling</b> — Horizontal mouth expansion with relaxed eye margins.<br>
        😡 <b>Angry</b> — Inner brow markers drawn close together downwards.<br>
        😔 <b>Sad / Frowning</b> — Inverted mouth corner coordinate mapping.<br>
        😐 <b>Neutral</b> — Default structural state baseline.
      </div>
    </div>
    """, unsafe_allow_html=True)

elif mode == "shape":
    hud_metrics.markdown("""
    <div class="hud-card" style="border-left: 4px solid #FACC15;">
      <div class="hud-label">👤 Structural Face Type Analytics Guide</div>
      <div style="color: #CBD5E1; font-size: 13px; margin-top: 10px; line-height: 1.7;">
        The algorithm reads facial landmark ratios relative to your biometric width matrix to categorize basic structural bone patterns:<br><br>
        🥚 <b>Oval Structural Shape</b> — Face height is moderately larger than cheekbone width, with forehead span wider than the jawline.<br>
        ⚪ <b>Round Symmetry Shape</b> — Face height and cheekbone span dimensions display sub-equal dimensions with smooth jaw curvature angles.<br>
        🔲 <b>Square Structural Shape</b> — Cross-sectional facial heights match widths closely, accompanied by sharp, broad lower jaw metrics.<br>
        💖 <b>Heart Shape</b> — Upper forehead widths measure significantly wider than narrow, prominent chin termination structures.<br>
        💎 <b>Diamond Shape</b> — Outermost cheekbones form the maximum dimension width, tapering sharply toward narrow forehead profiles and pointed chins.<br>
        ⏳ <b>Oblong / Rectangle Shape</b> — Elongated vertical structural frameworks where height metrics highly out-scale horizontal spans.
      </div>
    </div>
    """, unsafe_allow_html=True)

elif mode == "canvas":
    hud_metrics.markdown("""
    <div class="hud-card" style="border-left: 4px solid #FF00FF;">
      <div class="hud-label">✍️ Air Canvas Gesture Interlocking Rules</div>
      <div style="color: #CBD5E1; font-size: 13px; margin-top: 10px; line-height: 1.7;">
        Control your thread-isolated paint brushes with these hand movements:<br><br>
        ✏️ <b>Draw Mode</b> — Extend your <b>Index finger</b> while keeping your middle finger closed. The system tracks your finger tip vector to paint.<br>
        ✋ <b>Hover / Pause Mode</b> — Raise your middle finger or close your hand. This breaks the drawing path, letting you move around without painting.<br><br>
        💡 <i>Use the color picker and brush thickness controls in the sidebar to modify your digital ink setup!</i>
      </div>
    </div>
    """, unsafe_allow_html=True)

elif mode == "glasses":
    hud_metrics.markdown("""
    <div class="hud-card" style="border-left: 4px solid #FB923C;">
      <div class="hud-label">🕶️ Augmented Reality Synthesizer Specs</div>
      <div style="color: #CBD5E1; font-size: 13px; margin-top: 10px; line-height: 1.7;">
        Computes interactive depth positions using structural landmark anchors:<br><br>
        ▪️ <b>Scale Calibration</b> — Inter-Ocular space monitors real-time proximity shifts to scale the AR frame.<br>
        ▪️ <b>Angle Compensation</b> — Computes tilt slopes across eye center coordinates to rotate the asset matrix.<br>
        ▪️ <b>Asset Rendering</b> — Projects cybernetic neon eyewear lines onto coordinates 33, 263, and 61.
      </div>
    </div>
    """, unsafe_allow_html=True)

elif mode == "object":
    hud_metrics.markdown("""
    <div class="hud-card" style="border-left: 4px solid #00E5A0;">
      <div class="hud-label">🔍 Intelligent Object Classification Node</div>
      <div style="color: #CBD5E1; font-size: 13px; margin-top: 10px; line-height: 1.7;">
        Runs a highly optimized neural network graph to inspect item shapes:<br><br>
        ▪️ <b>Model Core</b> — Driven by an <i>EfficientNet-Lite0</i> convolutional architecture network.<br>
        ▪️ <b>Focal Target Area</b> — Keep target objects inside the center orange alignment box for optimal feature scanning.<br>
        ▪️ <b>Confidence Ratings</b> — Evaluates targets against thousands of classes to output accuracy percentage margins.
      </div>
    </div>
    """, unsafe_allow_html=True)

    
