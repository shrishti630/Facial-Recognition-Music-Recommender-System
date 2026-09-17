import os
import json
import time
import urllib.request
import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import cv2
import numpy as np
import webbrowser

# -------------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------------
st.set_page_config(
    page_title="Sangeet — 5 Core Emotions Music Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Styling (Clean Light Theme)
st.markdown("""
<style>
    /* Global Base */
    .stApp {
        background-color: #F8FAFC;
        color: #0F172A;
    }
    
    /* Clean Hero Container */
    .hero-banner {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 22px 28px;
        margin-bottom: 24px;
        box-shadow: 0 4px 16px -2px rgba(0, 0, 0, 0.04);
    }
    
    /* Genre / Mood tag pills */
    .mood-pill {
        display: inline-block;
        background: #F1F5F9;
        color: #334155;
        border: 1px solid #CBD5E1;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-right: 6px;
        margin-top: 4px;
    }
    
    /* Clean button styling */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
    }
    
    /* Hide Streamlit default chrome */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

try:
    from tensorflow.keras.models import load_model
except ImportError:
    from keras.models import load_model

import mediapipe as mp

# Support both Legacy MediaPipe (Python <= 3.12) and Modern MediaPipe Tasks API (Python 3.13+)
USE_LEGACY = hasattr(mp, "solutions") and hasattr(mp.solutions, "holistic")

if USE_LEGACY:
    mp_holistic = mp.solutions.holistic
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
else:
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

# Hand skeletal joint connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring finger
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17) # Pinky & Palm base
]

# -------------------------------------------------------------
# THE 5 CORE EMOTIONS (STRICTLY CONSTRAINED)
# -------------------------------------------------------------
FIVE_EMOTIONS = ["happy", "sad", "neutral", "surprise", "angry"]

EMOTION_METADATA = {
    "happy": {
        "emoji": "☀️",
        "title": "Happy & Joyful",
        "subtitle": "Radiating positive energy",
        "blurb": "Your expression is full of warmth and lightness. We're matching your state with uplifting melodies, feel-good rhythms, and cheerful pop anthems.",
        "color_bgr": (0, 200, 100),
        "hex": "#10B981",
        "genres": ["Feel-Good Pop", "Sunny Acoustics", "Disco Groove", "Upbeat Indie"],
        "tempo": "Brisk & lively • 115–128 BPM"
    },
    "sad": {
        "emoji": "🌧️",
        "title": "Gentle & Soulful (Sad)",
        "subtitle": "Taking a quiet, reflective moment",
        "blurb": "Quiet moments have their own gentle beauty. Here are comforting acoustic tracks, soulful storytelling, and slow melodies that offer warm company.",
        "color_bgr": (255, 160, 50),
        "hex": "#3B82F6",
        "genres": ["Mellow Acoustic", "Soulful Ballads", "Piano Solos", "Late Night Indie"],
        "tempo": "Slow & peaceful • 65–85 BPM"
    },
    "neutral": {
        "emoji": "☕",
        "title": "Calm & Centered (Neutral)",
        "subtitle": "Easygoing focus and peaceful flow",
        "blurb": "You look relaxed and centered. Perfect for mellow lo-fi beats, gentle café acoustics, and smooth rhythms to accompany your workflow.",
        "color_bgr": (220, 120, 160),
        "hex": "#8B5CF6",
        "genres": ["Lo-Fi Chill", "Quiet Jazz", "Study Beats", "Ambient Piano"],
        "tempo": "Easy & moderate • 75–95 BPM"
    },
    "surprise": {
        "emoji": "✨",
        "title": "Curious & Excited (Surprised)",
        "subtitle": "Sparked by something unexpected",
        "blurb": "Engaged and wide awake! Here are vibrant synths, fresh electronic drops, and dynamic melodies that keep the energy high.",
        "color_bgr": (0, 180, 250),
        "hex": "#F59E0B",
        "genres": ["Nu-Disco", "Electronic Pop", "Synthwave", "Future Beats"],
        "tempo": "Bouncy & dynamic • 120–135 BPM"
    },
    "angry": {
        "emoji": "⚡",
        "title": "Fiery & Intense (Angry)",
        "subtitle": "Channeling pure drive and intensity",
        "blurb": "Channel that fire into momentum. We're selecting punchy rock tracks, hard-hitting rhythms, and adrenaline-charged tracks.",
        "color_bgr": (50, 50, 240),
        "hex": "#EF4444",
        "genres": ["Alternative Rock", "Driving Beats", "Power Anthems", "Heavy Bass"],
        "tempo": "Driving & fast • 130–150 BPM"
    }
}

# 3 Hand Gestures available to try on camera
GESTURE_TEST_CATALOGUE = {
    "rock": {
        "emoji": "🤘",
        "name": "Rock Sign",
        "desc": "Extend index & pinky fingers",
        "vibe": "Rock & Alternative"
    },
    "Thumbsup": {
        "emoji": "👍",
        "name": "Thumbs Up",
        "desc": "Give a clear upward thumb",
        "vibe": "Top Viral Hits & Pop"
    },
    "hello": {
        "emoji": "👋",
        "name": "Wave Hello",
        "desc": "Open palm towards camera",
        "vibe": "Warm Morning Acoustics"
    }
}

@st.cache_resource
def get_model():
    return load_model("model.h5")

@st.cache_data
def get_labels():
    return np.load("labels.npy")

def ensure_task_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    face_task = os.path.join(base_dir, "face_landmarker.task")
    hand_task = os.path.join(base_dir, "hand_landmarker.task")

    if not os.path.exists(face_task):
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, face_task)
    if not os.path.exists(hand_task):
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, hand_task)
    return face_task, hand_task

@st.cache_resource
def get_detectors():
    if USE_LEGACY:
        hol = mp_holistic.Holistic()
        return {"mode": "legacy", "hol": hol}
    else:
        face_path, hand_path = ensure_task_models()
        base_face = mp_python.BaseOptions(model_asset_path=face_path)
        opt_face = mp_vision.FaceLandmarkerOptions(base_options=base_face, num_faces=1)
        face_det = mp_vision.FaceLandmarker.create_from_options(opt_face)

        base_hand = mp_python.BaseOptions(model_asset_path=hand_path)
        opt_hand = mp_vision.HandHandmarkerOptions = mp_vision.HandLandmarkerOptions(base_options=base_hand, num_hands=2)
        hand_det = mp_vision.HandLandmarker.create_from_options(opt_hand)
        return {"mode": "tasks", "face": face_det, "hand": hand_det}

model = get_model()
label = get_labels()
detectors = get_detectors()

def read_telemetry():
    """Reads latest mood prediction safely from disk."""
    try:
        if os.path.exists("detected_state.json"):
            with open("detected_state.json", "r") as f:
                return json.load(f)
    except Exception:
        pass
    try:
        raw = str(np.load("detected_emotion.npy")[0])
        return {"emotion": raw, "confidence": 0.0, "five_emotions": {}, "active_gesture": None}
    except Exception:
        return {"emotion": "", "confidence": 0.0, "five_emotions": {}, "active_gesture": None}


# -------------------------------------------------------------
# VIDEO PROCESSOR (Strictly 5 Emotions + Hand Tracking)
# -------------------------------------------------------------
class EmotionDetector:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.flip(frm, 1)
        h, w, _ = frm.shape

        lst = []
        face_detected = False
        left_hand_present = False
        right_hand_present = False

        # Multi-color visual definitions:
        COLOR_FACE = (255, 220, 0)       # Cyan for Face mesh
        COLOR_LEFT_HAND = (220, 0, 255)   # Vibrant Magenta for Left Hand
        COLOR_RIGHT_HAND = (0, 230, 120)  # Spring Green for Right Hand

        if detectors["mode"] == "legacy":
            hol = detectors["hol"]
            res = hol.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
            if res.face_landmarks:
                face_detected = True
                for i in res.face_landmarks.landmark:
                    lst.append(i.x - res.face_landmarks.landmark[1].x)
                    lst.append(i.y - res.face_landmarks.landmark[1].y)

                if res.left_hand_landmarks:
                    left_hand_present = True
                    for i in res.left_hand_landmarks.landmark:
                        lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                        lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
                else:
                    for _ in range(42):
                        lst.append(0.0)

                if res.right_hand_landmarks:
                    right_hand_present = True
                    for i in res.right_hand_landmarks.landmark:
                        lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                        lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
                else:
                    for _ in range(42):
                        lst.append(0.0)

                for lm in res.face_landmarks.landmark[::4]:
                    cv2.circle(frm, (int(lm.x * w), int(lm.y * h)), 1, COLOR_FACE, -1)

                if res.left_hand_landmarks:
                    l_pts = [(int(lm.x * w), int(lm.y * h)) for lm in res.left_hand_landmarks.landmark]
                    for p1, p2 in HAND_CONNECTIONS:
                        cv2.line(frm, l_pts[p1], l_pts[p2], COLOR_LEFT_HAND, 2)
                    for pt in l_pts:
                        cv2.circle(frm, pt, 4, (255, 255, 255), -1)

                if res.right_hand_landmarks:
                    r_pts = [(int(lm.x * w), int(lm.y * h)) for lm in res.right_hand_landmarks.landmark]
                    for p1, p2 in HAND_CONNECTIONS:
                        cv2.line(frm, r_pts[p1], r_pts[p2], COLOR_RIGHT_HAND, 2)
                    for pt in r_pts:
                        cv2.circle(frm, pt, 4, (255, 255, 255), -1)
        else:
            face_det = detectors["face"]
            hand_det = detectors["hand"]

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
            res_face = face_det.detect(mp_image)
            res_hand = hand_det.detect(mp_image)

            if res_face.face_landmarks:
                face_detected = True
                face_lms = res_face.face_landmarks[0]
                ref_face = face_lms[1]

                for lm in face_lms[:468]:
                    lst.append(lm.x - ref_face.x)
                    lst.append(lm.y - ref_face.y)

                left_hand = None
                right_hand = None
                if res_hand.hand_landmarks:
                    for idx, handedness in enumerate(res_hand.handedness):
                        lbl = handedness[0].category_name
                        if lbl == "Left" and left_hand is None:
                            left_hand = res_hand.hand_landmarks[idx]
                        elif lbl == "Right" and right_hand is None:
                            right_hand = res_hand.hand_landmarks[idx]

                # Draw Left Hand (Magenta lines + white dots)
                if left_hand:
                    left_hand_present = True
                    ref_l = left_hand[8]
                    for lm in left_hand:
                        lst.append(lm.x - ref_l.x)
                        lst.append(lm.y - ref_l.y)

                    l_pts = [(int(lm.x * w), int(lm.y * h)) for lm in left_hand]
                    for p1, p2 in HAND_CONNECTIONS:
                        cv2.line(frm, l_pts[p1], l_pts[p2], COLOR_LEFT_HAND, 2)
                    for pt in l_pts:
                        cv2.circle(frm, pt, 4, (255, 255, 255), -1)
                else:
                    for _ in range(42):
                        lst.append(0.0)

                # Draw Right Hand (Spring Green lines + white dots)
                if right_hand:
                    right_hand_present = True
                    ref_r = right_hand[8]
                    for lm in right_hand:
                        lst.append(lm.x - ref_r.x)
                        lst.append(lm.y - ref_r.y)

                    r_pts = [(int(lm.x * w), int(lm.y * h)) for lm in right_hand]
                    for p1, p2 in HAND_CONNECTIONS:
                        cv2.line(frm, r_pts[p1], r_pts[p2], COLOR_RIGHT_HAND, 2)
                    for pt in r_pts:
                        cv2.circle(frm, pt, 4, (255, 255, 255), -1)
                else:
                    for _ in range(42):
                        lst.append(0.0)

                # Draw Face Mesh (Cyan dots)
                for lm in face_lms[:468:4]:
                    cv2.circle(frm, (int(lm.x * w), int(lm.y * h)), 1, COLOR_FACE, -1)

        # ---------------------------------------------------------
        # PREDICTION FILTERED STRICTLY TO THE 5 BASIC EMOTIONS
        # ---------------------------------------------------------
        top_emotion = ""
        confidence = 0.0
        mood_accent_color = (255, 255, 255)
        active_gesture = None
        five_emotions = {}

        if len(lst) == 1020:
            lst_arr = np.array(lst).reshape(1, -1)
            try:
                probs = model.predict(lst_arr, verbose=0)[0]
                label_list = list(label)

                # 1. Filter probabilities strictly across the 5 basic emotions
                five_indices = [label_list.index(e) for e in FIVE_EMOTIONS if e in label_list]
                five_probs = np.array([probs[i] for i in five_indices])
                prob_sum = np.sum(five_probs)

                if prob_sum > 0:
                    five_norm = five_probs / prob_sum
                else:
                    five_norm = five_probs

                # Store normalized percentages for the 5 basic emotions
                for idx, emo in enumerate(FIVE_EMOTIONS):
                    five_emotions[emo] = round(float(five_norm[idx]) * 100, 1)

                # The detected emotion is STRICTLY the highest among the 5 emotions
                best_five_idx = int(np.argmax(five_norm))
                top_emotion = FIVE_EMOTIONS[best_five_idx]
                confidence = float(five_norm[best_five_idx]) * 100.0

                if top_emotion in EMOTION_METADATA:
                    mood_accent_color = EMOTION_METADATA[top_emotion]["color_bgr"]

                # 2. Check for optional hand gestures (rock, thumbs up, hello)
                for g in ["rock", "Thumbsup", "hello"]:
                    if g in label_list:
                        g_idx = label_list.index(g)
                        if probs[g_idx] > 0.35:
                            active_gesture = g
                            break

                # Save state safely
                payload = {
                    "emotion": top_emotion,
                    "confidence": round(confidence, 1),
                    "five_emotions": five_emotions,
                    "active_gesture": active_gesture,
                    "face_detected": True,
                    "hands": {"left": left_hand_present, "right": right_hand_present},
                    "timestamp": time.time()
                }
                with open("detected_state.tmp", "w") as f:
                    json.dump(payload, f)
                os.replace("detected_state.tmp", "detected_state.json")
                np.save("detected_emotion.npy", np.array([top_emotion]))
            except Exception:
                pass

        # ---------------------------------------------
        # CAMERA HUD OVERLAY (STRICTLY 5 EMOTIONS)
        # ---------------------------------------------
        overlay = frm.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (15, 20, 30), -1)
        cv2.rectangle(overlay, (0, h - 40), (w, h), (15, 20, 30), -1)
        cv2.addWeighted(overlay, 0.72, frm, 0.28, 0, frm)

        if face_detected and top_emotion:
            readable_title = EMOTION_METADATA.get(top_emotion, {}).get("title", top_emotion.title())
            cv2.putText(frm, f"EMOTION: {readable_title.upper()}", (20, 34), cv2.FONT_HERSHEY_DUPLEX, 0.82, mood_accent_color, 2)
            cv2.putText(frm, f"Confidence: {confidence:.1f}%", (20, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 225, 235), 1)

            # Confidence progress bar on camera
            bar_start = 220
            bar_len = max(50, w - bar_start - 30)
            fill_len = int(bar_len * (confidence / 100.0))
            cv2.rectangle(frm, (bar_start, 50), (bar_start + bar_len, 64), (35, 45, 60), -1)
            if fill_len > 0:
                cv2.rectangle(frm, (bar_start, 50), (bar_start + fill_len, 64), mood_accent_color, -1)
        else:
            cv2.putText(frm, "CENTER YOUR FACE IN CAMERA", (20, 38), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 140, 255), 2)
            cv2.putText(frm, "Detecting 5 basic emotions...", (20, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 190, 205), 1)

        # Bottom Bar Status (Gesture awareness + hand tracking)
        face_str = "FACE: LOCKED" if face_detected else "FACE: NONE"
        face_col = (0, 220, 120) if face_detected else (0, 100, 255)
        cv2.putText(frm, face_str, (20, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, face_col, 1)

        if active_gesture and active_gesture in GESTURE_TEST_CATALOGUE:
            g_meta = GESTURE_TEST_CATALOGUE[active_gesture]
            gesture_str = f"GESTURE: {g_meta['name'].upper()} DETECTED"
            cv2.putText(frm, gesture_str, (180, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 240, 255), 1)
        else:
            l_txt = "L-HAND: ACTIVE" if left_hand_present else "L-HAND: OFF"
            r_txt = "R-HAND: ACTIVE" if right_hand_present else "R-HAND: OFF"
            cv2.putText(frm, f"{l_txt} | {r_txt}", (180, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 205, 215), 1)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")


# -------------------------------------------------------------
# HERO HEADER
# -------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 15px;">
        <div>
            <h1 style="margin: 0; font-size: 2.2rem; font-weight: 800; color: #0F172A; letter-spacing: -0.02em;">
                🎵 Sangeet AI Studio
            </h1>
            <p style="margin: 6px 0 0 0; color: #64748B; font-size: 1.05rem;">
                Express your natural mood across the 5 core emotions to curate your soundtrack.
            </p>
        </div>
        <div style="display: flex; gap: 8px; flex-wrap: wrap;">
            <span class="mood-pill">☀️ Happy</span>
            <span class="mood-pill">🌧️ Sad</span>
            <span class="mood-pill">☕ Neutral</span>
            <span class="mood-pill">✨ Surprise</span>
            <span class="mood-pill">⚡ Angry</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Fetch latest state
telemetry = read_telemetry()
current_emotion = telemetry.get("emotion", "")
current_conf = telemetry.get("confidence", 0.0)
five_emotions = telemetry.get("five_emotions", {})
active_gesture = telemetry.get("active_gesture", None)

# Default to Neutral or detected emotion
if current_emotion not in FIVE_EMOTIONS:
    current_emotion = "neutral"

mood_info = EMOTION_METADATA.get(current_emotion, EMOTION_METADATA["neutral"])


# -------------------------------------------------------------
# MAIN 2-COLUMN STUDIO LAYOUT
# -------------------------------------------------------------
col_left, col_right = st.columns([1.1, 1], gap="large")

# LEFT COLUMN: Live Camera, 3 Gestures to Try, & 5 Basic Emotions Breakdown
with col_left:
    # Camera Mirror Card
    with st.container(border=True):
        st.subheader("📹 Live Camera Mirror")
        st.caption("Detecting facial expressions and hands in real time.")

        RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

        webrtc_streamer(
            key="sangeet-camera-streamer",
            desired_playing_state=True,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=EmotionDetector
        )

        st.caption("Tracking Colors: **Cyan** = Face Mesh | **Magenta** = Left Hand | **Green** = Right Hand")

    # INTERACTIVE 3 HAND GESTURES TO TRY (Live Feedback)
    with st.container(border=True):
        st.subheader("🖐️ Try These 3 Hand Gestures")
        st.caption("Show these hand gestures to the camera and watch the system recognize them live:")

        g1, g2, g3 = st.columns(3)

        with g1:
            st.markdown("### 🤘 Rock Sign")
            st.caption("**How to try:** Extend index & pinky")
            st.caption("Vibe: **Classic Rock**")
            if active_gesture == "rock":
                st.success("🟢 DETECTED LIVE!")
            else:
                st.info("Try this gesture")

        with g2:
            st.markdown("### 👍 Thumbs Up")
            st.caption("**How to try:** Upward thumb to camera")
            st.caption("Vibe: **Top Viral Hits**")
            if active_gesture == "Thumbsup":
                st.success("🟢 DETECTED LIVE!")
            else:
                st.info("Try this gesture")

        with g3:
            st.markdown("### 👋 Wave Hello")
            st.caption("**How to try:** Open palm to camera")
            st.caption("Vibe: **Warm Acoustics**")
            if active_gesture == "hello":
                st.success("🟢 DETECTED LIVE!")
            else:
                st.info("Try this gesture")

    # 5 BASIC EMOTIONS BREAKDOWN CARD
    with st.container(border=True):
        st.subheader("📊 5 Basic Emotions Balance")
        st.caption("Real-time probability breakdown strictly across the 5 core emotions:")

        for emo_key in FIVE_EMOTIONS:
            emo_pct = five_emotions.get(emo_key, 0.0)
            meta = EMOTION_METADATA[emo_key]
            c_lbl, c_bar, c_val = st.columns([2.8, 4.2, 1])
            with c_lbl:
                st.write(f"{meta['emoji']} **{meta['title'].split('(')[0].strip()}**")
            with c_bar:
                st.progress(min(1.0, max(0.0, emo_pct / 100.0)))
            with c_val:
                st.write(f"**{emo_pct:.0f}%**")

# RIGHT COLUMN: Mood Hero & Music Recommendation Hub
with col_right:
    # Detected Emotion Card (Strictly one of the 5 core emotions)
    with st.container(border=True):
        col_m_emoji, col_m_text = st.columns([1, 4])
        with col_m_emoji:
            st.markdown(f"<div style='font-size: 3.8rem; text-align: center; padding-top: 5px;'>{mood_info['emoji']}</div>", unsafe_allow_html=True)
        with col_m_text:
            st.caption(f"DETECTED EMOTION • {current_conf:.0f}% CONFIDENCE")
            st.subheader(mood_info["title"])
            st.markdown(f"<span style='color: {mood_info['hex']}; font-weight: 600;'>{mood_info['subtitle']}</span>", unsafe_allow_html=True)

        st.write(mood_info["blurb"])
        st.write(f"**Sound Profile:** {mood_info['tempo']}")

        # Render genre pills cleanly
        pills_html = " ".join([f"<span class='mood-pill'>{g}</span>" for g in mood_info["genres"]])
        st.markdown(pills_html, unsafe_allow_html=True)

    # Preferences Container
    with st.container(border=True):
        st.subheader("🎧 Music Preferences")

        c_lang, c_art = st.columns(2)
        with c_lang:
            lang = st.text_input("Language / Region", value="Hindi")
        with c_art:
            artist = st.text_input("Favorite Artist", value="Arijit Singh")

        st.caption("Popular Artists:")
        popular_artists = ["Arijit Singh", "Taylor Swift", "Diljit Dosanjh", "Coldplay"]
        p_cols = st.columns(4)
        for i, a_name in enumerate(popular_artists):
            if p_cols[i].button(a_name, key=f"btn_artist_{i}", use_container_width=True):
                artist = a_name
                st.rerun()

    # Recommendations Container
    with st.container(border=True):
        st.subheader("🚀 Curated Recommendations")

        # YouTube and Spotify queries using the strict 5 emotions
        search_query = f"{lang} {current_emotion} songs {artist}".strip()
        youtube_url = f"https://www.youtube.com/results?search_query={search_query.replace(' ', '+')}"
        spotify_url = f"https://open.spotify.com/search/{search_query.replace(' ', '%20')}"

        btn_c1, btn_c2 = st.columns([2, 1])
        with btn_c1:
            get_rec_clicked = st.button("🎵 Curate Songs for My Emotion", use_container_width=True, type="primary")
        with btn_c2:
            if st.button("Reset Emotion 🔄", use_container_width=True):
                np.save("detected_emotion.npy", np.array(["neutral"]))
                if os.path.exists("detected_state.json"):
                    os.remove("detected_state.json")
                st.rerun()

        if get_rec_clicked:
            try:
                webbrowser.open(youtube_url)
            except Exception:
                pass
            st.success(f"🎉 Playlist curated for **{mood_info['title']}**! Choose where to listen:")

        st.markdown("<div style='margin-top: 10px;'>", unsafe_allow_html=True)
        link_col1, link_col2 = st.columns(2)
        with link_col1:
            st.link_button("▶️ Listen on YouTube", youtube_url, use_container_width=True)
        with link_col2:
            st.link_button("🟢 Listen on Spotify", spotify_url, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

st.write("---")
st.markdown(
    "<div style='text-align: center; color: #94A3B8; font-size: 0.85rem; padding: 10px 0;'>"
    "Sangeet AI Studio • 5 Core Emotions (Angry, Happy, Sad, Surprised, Neutral) Powered by MediaPipe & TensorFlow"
    "</div>",
    unsafe_allow_html=True
)
