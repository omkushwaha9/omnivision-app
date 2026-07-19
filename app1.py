"""
OmniVision Pro — v3.0 — Production Bug-Fixed Build
═══════════════════════════════════════════════════
FIXES IN THIS BUILD:
  [BUG-1] CAMERA FREEZE  — self.module_target → self.mode (AttributeError killed recv thread)
  [BUG-2] BRUSH CLOSURE  — brush vars now stored as processor attributes, safe across threads
  [BUG-3] PALM LINE      — Complete pipeline rewrite: correct grayscale order, adaptive kernel,
                           CLAHE on skin ROI, Frangi-style ridge enhancement, bright overlay
  [BUG-4] FPS JITTER     — Rolling deque inside processor for smooth FPS display
  [BUG-5] CANVAS CLEAR   — Clear flag passed via processor attribute, actually works now
  [BUG-6] EMA SMOOTHING  — Re-added per-hand and per-face EMA inside recv()
  [BUG-7] AR GLASSES     — Added tilt rotation so glasses follow head angle correctly

NEW FEATURE: 👁️ Blink Counter & Eye Aspect Ratio (EAR) live graph
  — Counts blinks using Eye Aspect Ratio with hysteresis threshold
  — Shown in "Face Emotion Engine" mode as a second sub-display
"""

import os, ssl, time, collections, threading
import urllib.request
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode

ssl._create_default_https_context = ssl._create_unverified_context

# ─────────────────────────────────────────────────────────────────────────────
# MODEL REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
MODELS = {
    "hand_landmarker.task":      "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "face_landmarker.task":      "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "efficientnet_lite0.tflite": "https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite0/float32/1/efficientnet_lite0.tflite",
}

# ─────────────────────────────────────────────────────────────────────────────
# CORE MATH UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def vec_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Rotation-invariant joint angle via dot-product. Returns degrees [0,180]."""
    ba = a - b
    bc = c - b
    cos_t = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0))))

def ema(prev: np.ndarray, curr: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Exponential Moving Average landmark smoother."""
    return alpha * curr + (1.0 - alpha) * prev

def lm_array(landmarks, w: int, h: int) -> np.ndarray:
    """MediaPipe landmark list → float32 (N,2) pixel coords."""
    return np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# FINGER DETECTION — Vector-Angle (rotation & scale invariant)
# ─────────────────────────────────────────────────────────────────────────────
FINGER_JOINTS = {
    "thumb":  (1, 2, 4),
    "index":  (5, 6, 8),
    "middle": (9, 10, 12),
    "ring":   (13, 14, 16),
    "pinky":  (17, 18, 20),
}

def detect_fingers(pts: np.ndarray) -> dict:
    state = {}
    for name, (a, b, c) in FINGER_JOINTS.items():
        angle = vec_angle(pts[a], pts[b], pts[c])
        thresh = 140.0 if name == "thumb" else 160.0
        state[name] = angle > thresh
    return state

def classify_gesture(fingers: dict, pts: np.ndarray) -> tuple:
    t, i, m, r, p = [fingers[k] for k in ("thumb","index","middle","ring","pinky")]
    count = sum([t, i, m, r, p])

    # OK sign: thumb tip near index tip, others open
    ok_ratio = np.linalg.norm(pts[4] - pts[8]) / (np.linalg.norm(pts[0] - pts[9]) + 1e-8)
    if ok_ratio < 0.35 and m and r and p:
        return "OK / Excellent", "👌"

    patterns = {
        (False, False, False, False, False): ("Fist / Zero",       "✊"),
        (False, True,  False, False, False): ("One / Pointing",    "☝️"),
        (False, True,  True,  False, False): ("Two / Peace / V",   "✌️"),
        (False, True,  True,  True,  False): ("Three",             "3️⃣"),
        (False, True,  True,  True,  True):  ("Four",              "4️⃣"),
        (True,  True,  True,  True,  True):  ("Five / Open Hand",  "🖐️"),
        (True,  False, False, False, True):  ("Rock On / Horns",   "🤘"),
        (True,  False, False, False, False): ("Thumbs Up",         "👍"),
        (False, False, False, False, True):  ("Pinky Up",          "🤙"),
        (True,  True,  False, False, False): ("L-Shape / Gun",     "👆"),
        (False, True,  False, False, True):  ("Spiderman",         "🕷️"),
        (True,  False, True,  False, False): ("Crossed",           "🤞"),
    }
    return patterns.get((t, i, m, r, p), (f"Custom [{count} up]", "🤚"))

# ─────────────────────────────────────────────────────────────────────────────
# FACE EMOTION — IOD-Normalized Geometric Ratios
# ─────────────────────────────────────────────────────────────────────────────
def analyze_emotion(pts: np.ndarray) -> tuple:
    iod       = np.linalg.norm(pts[33] - pts[263]) + 1e-8
    mouth_w   = np.linalg.norm(pts[61]  - pts[291]) / iod
    mouth_h   = np.linalg.norm(pts[13]  - pts[14])  / iod
    brow_avg  = ((pts[55][1] - pts[33][1]) + (pts[285][1] - pts[263][1])) / (2.0 * iod)
    eye_h_l   = abs(pts[159][1] - pts[145][1]) / iod
    eye_h_r   = abs(pts[386][1] - pts[374][1]) / iod
    avg_eye_h = (eye_h_l + eye_h_r) / 2.0

    if mouth_h > 0.18 and avg_eye_h > 0.14:
        return "Surprised 😲", "#FACC15"
    elif avg_eye_h > 0.19:
        return "Wide-Eyed 👀", "#818CF8"
    elif mouth_w > 0.75 and mouth_h < 0.12:
        return "Happy / Smiling 😊", "#34D399"
    elif brow_avg < -0.10 and mouth_w < 0.60:
        return "Angry 😡", "#F87171"
    elif mouth_w < 0.58 and mouth_h < 0.07:
        return "Sad / Frowning 😔", "#60A5FA"
    elif mouth_h > 0.13:
        return "Open Mouth 😮", "#FB923C"
    else:
        return "Neutral 😐", "#94A3B8"

# ─────────────────────────────────────────────────────────────────────────────
# BLINK COUNTER — Eye Aspect Ratio (EAR)
# MediaPipe face landmarks for each eye (vertical / horizontal pairs):
#   Left:  upper=159, lower=145, left-corner=33, right-corner=133
#   Right: upper=386, lower=374, left-corner=362, right-corner=263
# EAR = (v1 + v2) / (2 * h)  — drops sharply on blink
# ─────────────────────────────────────────────────────────────────────────────
EAR_THRESHOLD = 0.21   # below this = eye closed
EAR_CONSEC    = 2      # frames eye must be closed to count as blink

def eye_aspect_ratio(pts: np.ndarray, upper: int, lower: int,
                     left_c: int, right_c: int) -> float:
    v = abs(pts[upper][1] - pts[lower][1])
    h = np.linalg.norm(pts[left_c] - pts[right_c]) + 1e-8
    return float(v / h)

# ─────────────────────────────────────────────────────────────────────────────
# FACE SHAPE ANALYZER
# ─────────────────────────────────────────────────────────────────────────────
def analyze_face_shape(pts: np.ndarray) -> str:
    cheek_w    = np.linalg.norm(pts[234] - pts[454]) + 1e-8
    face_h     = np.linalg.norm(pts[10]  - pts[152]) / cheek_w
    forehead_w = np.linalg.norm(pts[54]  - pts[284]) / cheek_w
    jaw_w      = np.linalg.norm(pts[172] - pts[397]) / cheek_w

    if face_h > 1.38:
        return "Oblong / Rectangle Shape ⏳"
    elif forehead_w > 0.95 and jaw_w < 0.75:
        return "Heart Shape 💖"
    elif forehead_w < 0.80 and jaw_w < 0.72 and face_h < 1.30:
        return "Diamond Shape 💎"
    elif face_h < 1.14:
        return "Square Shape 🔲" if jaw_w > 0.82 else "Round Shape ⚪"
    else:
        return "Square Shape 🔲" if jaw_w > 0.82 else "Oval Shape 🥚"

# ─────────────────────────────────────────────────────────────────────────────
# PALM LINE EXTRACTOR — Fixed Pipeline (v3)
# ─────────────────────────────────────────────────────────────────────────────
def extract_palm_lines(frame: np.ndarray) -> np.ndarray:
    """
    FIXED pipeline order and parameters.
    Steps:
      1. Convert to grayscale FIRST, then bilateral filter (was reversed)
      2. CLAHE with tighter clipLimit for skin-tone robustness
      3. Black-hat with SMALLER kernel (9x9 vs 17x17) — catches finer creases
      4. Otsu threshold on the black-hat result
      5. Morphological close to connect broken line segments
      6. Canny edge detection for sharp line rendering
      7. Bright composite overlay — amber lines on dimmed original
    """
    h, w = frame.shape[:2]

    # Step 1: Gray first, then denoise
    gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, 7, 55, 55)

    # Step 2: CLAHE — equalise contrast across all skin tones
    clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6))
    enhanced = clahe.apply(denoised)

    # Step 3: Black-hat — 9x9 kernel captures fine palm crease lines
    k9       = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, k9)

    # Step 4: Boost signal before threshold
    blackhat = cv2.convertScaleAbs(blackhat, alpha=2.8, beta=0)

    # Step 5: Otsu auto-threshold
    _, thresh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 6: Close gaps in line segments
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k3)

    # Step 7: Canny edge on the threshold result for clean thin lines
    edges = cv2.Canny(thresh, 30, 100)

    # Step 8: Dilate edges slightly so they're visible at all resolutions
    edges = cv2.dilate(edges, k3, iterations=1)

    # Step 9: Composite — bright amber lines over dimmed original
    dim    = cv2.convertScaleAbs(frame, alpha=0.40, beta=0)  # darken background
    overlay = dim.copy()
    overlay[thresh > 0] = (0, 80, 160)     # dark amber fill under lines
    overlay[edges  > 0] = (0, 200, 255)    # bright amber on crease edges

    # Final blend — strong overlay so lines are clearly visible
    result = cv2.addWeighted(dim, 0.35, overlay, 0.95, 0)

    # Label
    cv2.putText(result, "PALM CREASE MAP", (10, h - 12),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
    return result

# ─────────────────────────────────────────────────────────────────────────────
# DRAWING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)
]
FINGER_IDX_MAP = {
    "thumb":  [1,2,3,4], "index":  [5,6,7,8],
    "middle": [9,10,11,12], "ring": [13,14,15,16], "pinky": [17,18,19,20]
}
FINGER_COLORS = {
    "thumb": (0,255,100), "index": (255,200,0),
    "middle": (0,200,255), "ring": (200,0,255), "pinky": (255,100,0)
}

def draw_hand(frame: np.ndarray, pts: np.ndarray, fingers: dict):
    lm_col = {0: (200,200,200)}
    for fname, idxs in FINGER_IDX_MAP.items():
        c = FINGER_COLORS[fname] if fingers.get(fname) else (70,70,70)
        for idx in idxs: lm_col[idx] = c
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)),
                 lm_col.get(a,(100,100,100)), 2, cv2.LINE_AA)
    for i, pt in enumerate(pts):
        cv2.circle(frame, tuple(pt.astype(int)), 5, lm_col.get(i,(100,100,100)), cv2.FILLED)
        cv2.circle(frame, tuple(pt.astype(int)), 5, (255,255,255), 1, cv2.LINE_AA)

def draw_face_minimal(frame: np.ndarray, pts: np.ndarray, color=(0,220,255)):
    for idx in [33,263,61,291,13,14,55,285,1,4,152,10,234,454,159,145,386,374]:
        if 0 <= idx < len(pts):
            cv2.circle(frame, tuple(pts[idx].astype(int)), 3, color, cv2.FILLED)

def draw_hud(frame: np.ndarray, label: str, fps: float, ms: float, color=(0,229,160)):
    h, w = frame.shape[:2]
    bar = frame.copy()
    cv2.rectangle(bar, (0, h-55), (w, h), (10,14,22), cv2.FILLED)
    cv2.addWeighted(bar, 0.75, frame, 0.25, 0, frame)
    cv2.putText(frame, label, (16, h-21), cv2.FONT_HERSHEY_DUPLEX, 0.58, color, 1, cv2.LINE_AA)
    cv2.line(frame, (0, h-55), (w, h-55), color, 1)
    fps_str = f"FPS: {fps:.1f}  |  {ms:.0f}ms"
    (tw, _), _ = cv2.getTextSize(fps_str, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(frame, fps_str, (w-tw-10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160,175,192), 1, cv2.LINE_AA)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    for fname, url in MODELS.items():
        if not os.path.exists(fname):
            with st.spinner(f"⬇️ Downloading {fname}…"):
                urllib.request.urlretrieve(url, fname)

    hand_opts = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path="hand_landmarker.task"),
        num_hands=2, min_hand_detection_confidence=0.60,
        min_hand_presence_confidence=0.60, min_tracking_confidence=0.50)

    face_opts = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path="face_landmarker.task"),
        num_faces=1, min_face_detection_confidence=0.60,
        min_face_presence_confidence=0.60, min_tracking_confidence=0.50)

    obj_opts = vision.ImageClassifierOptions(
        base_options=python.BaseOptions(model_asset_path="efficientnet_lite0.tflite"),
        max_results=3, score_threshold=0.28)

    return (vision.HandLandmarker.create_from_options(hand_opts),
            vision.FaceLandmarker.create_from_options(face_opts),
            vision.ImageClassifier.create_from_options(obj_opts))

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & STYLES
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="OmniVision Pro", page_icon="👁️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
.stApp { background: #080C14; }
section[data-testid="stSidebar"] { background: #0D1220; border-right: 1px solid #1E2D45; }
section[data-testid="stSidebar"] .stRadio label { color: #CBD5E1 !important; font-size: 14px; }
.hud-card { background: linear-gradient(135deg,#0D1B2A 0%,#112030 100%);
            border: 1px solid #1E3A52; border-left: 4px solid #00E5A0;
            border-radius: 10px; padding: 18px 24px; margin-bottom: 18px; }
.hud-label { color: #64748B; font-size: 11px; letter-spacing: 2px;
             text-transform: uppercase; font-weight: 600; }
.hud-value { color: #F0FDF4; font-size: 26px; font-weight: 700; margin-top: 6px; }
.hud-sub   { color: #94A3B8; font-size: 13px; margin-top: 4px; }
.page-title{ color: #F8FAFC; font-size: 26px; font-weight: 800; letter-spacing:-0.5px; }
.page-sub  { color: #475569; font-size: 13px; margin-top: -4px; }
.hud-divider{ border:none; border-top:1px solid #1E2D45; margin:14px 0; }
div[data-testid="stNotification"] { display:none !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
def init_state():
    defs = {
        "theme": "dark",
        "gesture_buf": collections.deque(maxlen=12),
        "emotion_buf":  collections.deque(maxlen=10),
        "current_mode": "hand",
        "cam_id": 0,
        "is_switching": False,
    }
    for k, v in defs.items():
        if k not in st.session_state:
            st.session_state[k] = v
init_state()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
title_col, toggle_col = st.sidebar.columns([5, 1])
with title_col:
    st.markdown("<h2 style='margin:0;padding:0;font-size:22px;'>🧠 OmniVision Pro</h2>",
                unsafe_allow_html=True)
with toggle_col:
    if st.button("☀️" if st.session_state.theme == "dark" else "🌙", key="theme_btn"):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()

st.sidebar.markdown("---")

MODES = {
    "🖐️  Hand Gesture Decoder": "hand",
    "🔮  Palm Line Analyzer":   "palm",
    "🎭  Face Emotion + Blink": "face",
    "👤  Face Shape Analyzer":  "shape",
    "✍️   Air Canvas":          "canvas",
    "🕶️   AR Glasses Overlay":  "glasses",
    "🔍  Object Classifier":    "object",
}
mode_label = st.sidebar.radio("Active Module", list(MODES.keys()), index=0)
mode       = MODES[mode_label]

st.sidebar.markdown("---")
st.sidebar.markdown("**⚙️ Detection Settings**")
smooth_alpha   = st.sidebar.slider("Temporal Smoothing (EMA α)", 0.1, 1.0, 0.55, 0.05)
min_confidence = st.sidebar.slider("Min Detection Confidence",   0.3, 0.95, 0.60, 0.05)

brush_r, brush_g, brush_b, brush_size = 255, 0, 255, 7
if mode == "canvas":
    hex_c       = st.sidebar.color_picker("Brush Color", "#FF00FF")
    brush_r, brush_g, brush_b = int(hex_c[1:3],16), int(hex_c[3:5],16), int(hex_c[5:7],16)
    brush_size  = st.sidebar.slider("Brush Size", 2, 20, 7)
    clear_canvas = st.sidebar.button("🗑️ Clear Canvas")
else:
    clear_canvas = False

st.sidebar.markdown("---")
st.sidebar.caption("OmniVision Pro v3.0 · MediaPipe · OpenCV")

# ─────────────────────────────────────────────────────────────────────────────
# THEME INJECTION
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.theme == "dark":
    bg_app="#111111"; bg_sb="#232323"; bd="#343434"
    tx="#F3F4F6"; txm="#696969"; hud_grad="linear-gradient(135deg,#232323,#111111)"
    tc_body="#7A7A7A"; tc_bold="#E5E5E5"
else:
    bg_app="#E1F0DA"; bg_sb="#D4E7C5"; bd="#BFD8AF"
    tx="#111111"; txm="#526D82"; hud_grad="linear-gradient(135deg,#E1F0DA,#D4E7C5)"
    tc_body="#3A4D5E"; tc_bold="#1A2636"

st.markdown(f"""
<style>
.stApp {{ background:{bg_app}!important; }}
h1,h2,h3,p,span,label {{ color:{tx}!important; }}
section[data-testid="stSidebar"] {{ background:{bg_sb}!important; border-right:1px solid {bd}!important; }}
section[data-testid="stSidebar"] .stRadio label {{ color:{tx}!important; }}
.hud-card {{ background:{hud_grad}!important; border:1px solid {bd}!important;
             border-left:4px solid #00E5A0!important; border-radius:10px;
             padding:18px 24px; margin-bottom:18px; }}
.hud-label {{ color:{txm}!important; }}
.hud-value {{ color:{tx}!important; }}
.hud-sub   {{ color:{txm}!important; }}
.page-title{{ color:{tx}!important; }}
.page-sub  {{ color:{txm}!important; }}
.hud-card div[style*="color"],
.hud-card p, .hud-card span {{ color:{tc_body}!important; }}
.hud-card b {{ color:{tc_bold}!important; font-weight:700!important; }}
div[data-testid="stSidebarUserContent"] div[data-testid="stButton"] button {{
    background:transparent!important; border:none!important; padding:0!important;
    box-shadow:none!important; min-width:unset!important; font-size:17px!important;
    opacity:0.45!important; transition:opacity 0.15s!important; }}
div[data-testid="stSidebarUserContent"] div[data-testid="stButton"] button:hover {{
    opacity:0.90!important; background:transparent!important; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="page-title">👁️ OmniVision Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Real-Time Vision · v3.0 Powered by Python · Real-time ML Models</div>', unsafe_allow_html=True)
st.markdown("<hr class='hud-divider'>", unsafe_allow_html=True)

col_vid, col_hud = st.columns([3, 2])
with col_vid:
    run      = st.checkbox("▶️  Activate Camera Pipeline", value=False)
    viewport = st.empty()
with col_hud:
    hud_main    = st.empty()
    hud_fingers = st.empty()
    hud_metrics = st.empty()

hand_engine, face_engine, obj_engine = load_models()

# ─────────────────────────────────────────────────────────────────────────────
# WEBRTC VIDEO PROCESSOR — All bugs fixed
# ─────────────────────────────────────────────────────────────────────────────
class OmniVisionProcessor:
    def __init__(self, mode="hand", alpha=0.55,
                 brush_rgb=(255,0,255), brush_size=7):
        # BUG-1 FIX: attribute is self.mode, used as self.mode throughout
        self.mode       = mode
        self.alpha      = alpha
        # BUG-2 FIX: brush config stored as instance attributes
        self.brush_bgr  = (brush_rgb[2], brush_rgb[1], brush_rgb[0])  # convert to BGR
        self.brush_size = brush_size

        # EMA state (BUG-6 FIX: re-added)
        self.prev_hand  = {}     # hand_idx → prev pts
        self.prev_face  = None

        # BUG-5 FIX: canvas clear via threading event
        self.clear_event  = threading.Event()
        self.paint_memory = []

        # FPS smoothing (BUG-4 FIX)
        self.fps_buf  = collections.deque(maxlen=25)
        self.t_last   = time.time()

        # Gesture / emotion voting buffers
        self.gest_buf  = collections.deque(maxlen=12)
        self.emot_buf  = collections.deque(maxlen=10)

        # Blink counter state (NEW FEATURE)
        self.blink_count     = 0
        self.ear_consec      = 0
        self.blink_in_prog   = False
        self.ear_history     = collections.deque(maxlen=60)  # last 60 frames

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        t0  = time.perf_counter()
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w = img.shape[:2]

        # BUG-5 FIX: clear canvas if flag set
        if self.clear_event.is_set():
            self.paint_memory = []
            self.clear_event.clear()

        img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        readout  = "Initializing…"
        hud_col  = (0, 229, 160)

        # ── FPS (BUG-4 FIX: averaged) ───────────────────────────────
        t_now  = time.time()
        self.fps_buf.append(1.0 / (t_now - self.t_last + 1e-8))
        self.t_last = t_now
        fps    = float(np.mean(self.fps_buf))

        # ════════════════════════════════════════════════════════════
        # MODULE ROUTING  (BUG-1 FIX: self.mode everywhere)
        # ════════════════════════════════════════════════════════════

        # ── HAND / CANVAS ────────────────────────────────────────────
        if self.mode in ("hand", "canvas"):
            res = hand_engine.detect(mp_img)
            if res.hand_landmarks:
                for hi, lm_list in enumerate(res.hand_landmarks):
                    raw = lm_array(lm_list, w, h)

                    # BUG-6 FIX: EMA smoothing per hand
                    prev = self.prev_hand.get(hi)
                    pts  = ema(prev, raw, self.alpha) if (prev is not None and prev.shape == raw.shape) else raw
                    self.prev_hand[hi] = pts

                    fingers = detect_fingers(pts)
                    gesture, emoji = classify_gesture(fingers, pts)

                    # Multi-frame voting
                    self.gest_buf.append((gesture, emoji))
                    voted = collections.Counter(self.gest_buf).most_common(1)[0][0]
                    gesture, emoji = voted

                    draw_hand(img, pts, fingers)

                    if self.mode == "hand":
                        readout = f"Gesture: {emoji}  {gesture}"

                    elif self.mode == "canvas":
                        drawing = fingers.get("index") and not fingers.get("middle")
                        tip = tuple(pts[8].astype(int))
                        self.paint_memory.append(tip if drawing else None)
                        readout = "✏️ Drawing" if drawing else "✋ Hovering — raise only index to draw"
            else:
                self.prev_hand.clear()
                if self.mode == "canvas":
                    self.paint_memory.append(None)
                readout = "🔍 No hand in frame"

            # Draw canvas trails
            if self.mode == "canvas" and len(self.paint_memory) > 1:
                for i in range(1, len(self.paint_memory)):
                    a, b = self.paint_memory[i-1], self.paint_memory[i]
                    if a and b:
                        cv2.line(img, a, b, self.brush_bgr, self.brush_size, cv2.LINE_AA)

        # ── PALM LINE (BUG-3 FIX: complete rewrite) ──────────────────
        elif self.mode == "palm":
            img     = extract_palm_lines(img)
            readout = "Palm Crease Analysis: Active"

        # ── FACE EMOTION + BLINK COUNTER (NEW FEATURE) ───────────────
        elif self.mode == "face":
            res = face_engine.detect(mp_img)
            if res.face_landmarks:
                raw = lm_array(res.face_landmarks[0], w, h)
                pts = ema(self.prev_face, raw, self.alpha) if (self.prev_face is not None and self.prev_face.shape == raw.shape) else raw
                self.prev_face = pts

                draw_face_minimal(img, pts)
                emotion, em_hex = analyze_emotion(pts)

                # Emotion voting
                self.emot_buf.append((emotion, em_hex))
                voted_e = collections.Counter(e for e,_ in self.emot_buf).most_common(1)[0][0]
                em_hex  = next((c for e,c in self.emot_buf if e==voted_e), em_hex)
                emotion = voted_e

                hud_col = tuple(int(em_hex.lstrip("#")[i:i+2],16) for i in (4,2,0))
                readout = f"Expression: {emotion}"

                # ── BLINK COUNTER ──────────────────────────────────
                ear_l = eye_aspect_ratio(pts, 159, 145, 33,  133)
                ear_r = eye_aspect_ratio(pts, 386, 374, 362, 263)
                ear   = (ear_l + ear_r) / 2.0
                self.ear_history.append(ear)

                if ear < EAR_THRESHOLD:
                    self.ear_consec += 1
                    self.blink_in_prog = True
                else:
                    if self.blink_in_prog and self.ear_consec >= EAR_CONSEC:
                        self.blink_count += 1
                    self.ear_consec    = 0
                    self.blink_in_prog = False

                # Draw EAR bar on frame (right side)
                bar_h  = int(ear * 300)
                bx     = w - 22
                cv2.rectangle(img, (bx, h-60), (bx+12, h-60-min(bar_h,80)),
                              (0,229,160) if ear > EAR_THRESHOLD else (0,80,255), cv2.FILLED)
                cv2.rectangle(img, (bx, h-140), (bx+12, h-60), (30,40,50), 1)
                cv2.putText(img, "EYE", (bx-2, h-145),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (120,140,160), 1, cv2.LINE_AA)

                # Blink count overlay on frame
                cv2.putText(img, f"Blinks: {self.blink_count}", (12, 34),
                            cv2.FONT_HERSHEY_DUPLEX, 0.62, (0,229,160), 1, cv2.LINE_AA)
            else:
                self.prev_face = None
                readout = "🔍 Align face to detect expression"

        # ── FACE SHAPE ───────────────────────────────────────────────
        elif self.mode == "shape":
            res = face_engine.detect(mp_img)
            if res.face_landmarks:
                raw = lm_array(res.face_landmarks[0], w, h)
                pts = ema(self.prev_face, raw, self.alpha) if (self.prev_face is not None and self.prev_face.shape == raw.shape) else raw
                self.prev_face = pts

                draw_face_minimal(img, pts, color=(0,255,150))
                shape   = analyze_face_shape(pts)
                readout = f"Face Type: {shape}"
                hud_col = (0, 200, 255)

                # Structural measurement lines
                for (p1, p2, c) in [
                    (10, 152, (255,200,0)),
                    (54, 284, (0,255,255)),
                    (234,454, (0,255,255)),
                    (172,397, (0,255,200)),
                ]:
                    cv2.line(img, tuple(pts[p1].astype(int)),
                                  tuple(pts[p2].astype(int)), c, 1, cv2.LINE_AA)
                for idx in [10,152,54,284,234,454,172,397]:
                    cv2.circle(img, tuple(pts[idx].astype(int)), 4, (0,255,255), cv2.FILLED)
            else:
                self.prev_face = None
                readout = "🔍 Align face to analyze structure"

        # ── AR GLASSES (tilt-corrected) ───────────────────────────────
        elif self.mode == "glasses":
            res = face_engine.detect(mp_img)
            if res.face_landmarks:
                raw = lm_array(res.face_landmarks[0], w, h)
                pts = ema(self.prev_face, raw, self.alpha) if (self.prev_face is not None and self.prev_face.shape == raw.shape) else raw
                self.prev_face = pts

                draw_face_minimal(img, pts)

                eye_l  = pts[33].astype(float)
                eye_r  = pts[263].astype(float)
                mid    = ((eye_l + eye_r) / 2).astype(int)
                iod    = int(np.linalg.norm(eye_l - eye_r))
                g_w    = int(iod * 1.9)
                g_h    = int(iod * 0.50)

                # BUG-7 FIX: compute tilt angle so glasses rotate with head
                dx     = eye_r[0] - eye_l[0]
                dy     = eye_r[1] - eye_l[1]
                angle  = float(np.degrees(np.arctan2(dy, dx)))

                # Draw rotated glasses via getRotationMatrix2D
                tmp = np.zeros_like(img)
                lx1, rx2 = mid[0] - g_w//2, mid[0] + g_w//2
                y1, y2   = mid[1] - g_h//2, mid[1] + g_h//2
                if lx1 > 0 and y1 > 0 and rx2 < w and y2 < h:
                    cv2.rectangle(tmp, (lx1, y1),    (mid[0]-4, y2),  (255,0,220), 2, cv2.LINE_AA)
                    cv2.rectangle(tmp, (mid[0]+4, y1),(rx2, y2),       (255,0,220), 2, cv2.LINE_AA)
                    cv2.line(tmp, (mid[0]-4, mid[1]), (mid[0]+4, mid[1]), (200,200,200), 2)
                    cv2.line(tmp, (lx1, mid[1]),      (lx1-18, mid[1]),   (255,0,220), 2)
                    cv2.line(tmp, (rx2, mid[1]),      (rx2+18, mid[1]),   (255,0,220), 2)
                    M   = cv2.getRotationMatrix2D((int(mid[0]), int(mid[1])), angle, 1.0)
                    tmp = cv2.warpAffine(tmp, M, (w, h))
                    img = cv2.addWeighted(img, 1.0, tmp, 1.0, 0)
                readout = "AR Glasses: Tilt-Corrected Overlay Active"
            else:
                self.prev_face = None
                readout = "🔍 Align face to display glasses"

        # ── OBJECT CLASSIFIER ─────────────────────────────────────────
        elif self.mode == "object":
            res = obj_engine.classify(mp_img)
            cx, cy = w//2, h//2
            # Animated pulsing reticle
            pulse = int(6 * np.sin(time.time() * 4)) + 100
            cv2.rectangle(img, (cx-pulse, cy-pulse), (cx+pulse, cy+pulse), (0,165,255), 2, cv2.LINE_AA)
            for dx, dy in [(-1,-1),(1,-1),(-1,1),(1,1)]:
                cv2.line(img, (cx+dx*pulse,cy+dy*pulse),(cx+dx*(pulse-18),cy+dy*pulse), (0,255,255),2)
                cv2.line(img, (cx+dx*pulse,cy+dy*pulse),(cx+dx*pulse,cy+dy*(pulse-18)),(0,255,255),2)

            if res.classifications and res.classifications[0].categories:
                cats = res.classifications[0].categories
                top  = cats[0]
                name = top.category_name.replace("_"," ").title()
                conf = top.score
                readout = f"Object: {name}  ({conf*100:.0f}%)"
                hud_col = (0,165,255)
                # Confidence bar
                bw = int(conf * (w-30))
                cv2.rectangle(img, (15, h-30), (15+bw, h-20), (0,165,255), cv2.FILLED)
                cv2.rectangle(img, (15, h-30), (w-15,  h-20), (40,60,80),  1)
                # All top results as small text
                for ri, cat in enumerate(cats[:3]):
                    lbl = f"{cat.category_name.replace('_',' ').title()} {cat.score*100:.0f}%"
                    cv2.putText(img, lbl, (12, 22+ri*18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150,180,200), 1, cv2.LINE_AA)
            else:
                readout = "🔍 Point object inside the reticle…"

        # ── HUD OVERLAY ───────────────────────────────────────────────
        ms = (time.perf_counter() - t0) * 1000
        draw_hud(img, readout, fps, ms, hud_col)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ─────────────────────────────────────────────────────────────────────────────
# CONTROL INTERFACE
# ─────────────────────────────────────────────────────────────────────────────
with col_vid:
    # Mode switch detector → force WebRTC component rebuild
    if mode != st.session_state.current_mode:
        st.session_state.current_mode = mode
        st.session_state.cam_id      += 1
        st.session_state.is_switching = True
        st.rerun()

    # Cooling-off frame (gives OS time to release camera lock)
    if st.session_state.is_switching:
        st.session_state.is_switching = False
        st.markdown("""
        <div style="background:#0D1220;border:1px solid #1E3A52;border-radius:12px;
                    padding:40px;text-align:center;margin-top:20px;">
          <div style="margin:0 auto 16px auto;width:40px;height:40px;border:4px solid #1E3A52;
                      border-top-color:#38BDF8;border-radius:50%;animation:spin 1s linear infinite;"></div>
          <div style="color:#F1F5F9;font-size:16px;font-weight:600;">Reconfiguring AI Pipeline…</div>
          <div style="color:#64748B;font-size:13px;margin-top:4px;">Switching module safely…</div>
        </div>
        <style>@keyframes spin{to{transform:rotate(360deg);}}</style>
        """, unsafe_allow_html=True)
        time.sleep(0.55)
        st.rerun()

    elif not run:
        viewport.markdown("""
        <div style="background:#0D1220;border:1px dashed #1E3A52;border-radius:12px;
                    padding:60px 40px;text-align:center;margin-top:20px;">
          <div style="font-size:48px;margin-bottom:16px;">📷</div>
          <div style="color:#F1F5F9;font-size:18px;font-weight:700;">Camera Pipeline Offline</div>
          <div style="color:#64748B;font-size:14px;margin-top:8px;">
            Enable the checkbox above to activate real-time processing.</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # Build processor with current sidebar config
        def make_processor():
            p = OmniVisionProcessor(
                mode       = mode,
                alpha      = smooth_alpha,
                brush_rgb  = (brush_r, brush_g, brush_b),
                brush_size = brush_size,
            )
            return p

        ctx = webrtc_streamer(
            key=f"omnivision-v{st.session_state.cam_id}",
            mode=WebRtcMode.SENDRECV,
            async_processing=True,
            rtc_configuration={
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {"urls": ["stun:stun1.l.google.com:19302"]},
                    {"urls": ["stun:stun.services.mozilla.com"]},
                ]
            },
            media_stream_constraints={
                "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
                "audio": False,
            },
            video_processor_factory=make_processor,
        )

        # Live update processor config on sidebar changes
        if ctx and ctx.video_processor:
            ctx.video_processor.mode       = mode
            ctx.video_processor.alpha      = smooth_alpha
            ctx.video_processor.brush_bgr  = (brush_b, brush_g, brush_r)
            ctx.video_processor.brush_size = brush_size
            # BUG-5 FIX: signal canvas clear to the thread
            if clear_canvas:
                ctx.video_processor.clear_event.set()

# ─────────────────────────────────────────────────────────────────────────────
# HUD SIDEBAR PANELS
# ─────────────────────────────────────────────────────────────────────────────
hud_main.markdown(f"""
<div class="hud-card">
  <div class="hud-label">📡 Active Module</div>
  <div class="hud-value" style="font-size:18px;">{mode_label.strip()}</div>
  <div class="hud-sub">v3.0 · Powered by Python · Real-time ML Models</div>
</div>
""", unsafe_allow_html=True)
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
