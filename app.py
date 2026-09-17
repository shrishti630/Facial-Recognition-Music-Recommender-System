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
# PAGE CONFIGURATION (CLEAN MODERN LIGHT THEME)
# -------------------------------------------------------------
st.set_page_config(
    page_title="Sangeet — Music by Mood",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Styling (Scoped without breaking markdown parsers)
st.markdown("""
<style>
    /* Global Background and Typography */
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
# MOOD CATALOGUE WITH COLOR CODING
# -------------------------------------------------------------
MOOD_CATALOGUE = {
    "happy": {
        "emoji": "☀️",
        "title": "Bright & Joyful",
        "subtitle": "Radiating positive energy",
        "blurb": "Your expression is full of warmth and lightness. We're matching your state with uplifting melodies, feel-good rhythms, and cheerful anthems.",
        "color_bgr": (0, 200, 100),
        "hex": "#10B981",
        "genres": ["Feel-Good Pop", "Sunny Acoustics", "Disco Groove", "Indie Upbeat"],
        "tempo": "Brisk & lively • 115–128 BPM"
    },
    "sad": {
        "emoji": "🌧️",
        "title": "Gentle & Reflective",
        "subtitle": "Taking a quiet, soulful moment",
        "blurb": "Quiet moments have their own beauty. Here are comforting acoustic tracks, soulful storytelling, and melodies that offer gentle company.",
        "color_bgr": (255, 160, 50),
        "hex": "#3B82F6",
        "genres": ["Mellow Acoustic", "Soulful Ballads", "Piano Solos", "Late Night Indie"],
        "tempo": "Slow & peaceful • 65–85 BPM"
    },
    "angry": {
        "emoji": "⚡",
        "title": "Fiery & High Voltage",
        "subtitle": "Channeling pure drive and intensity",
        "blurb": "Channel that fire into momentum. We're selecting punchy rock tracks, hard-hitting rhythms, and adrenaline-charged tracks.",
        "color_bgr": (50, 50, 240),
        "hex": "#EF4444",
        "genres": ["Alternative Rock", "Driving Beats", "Power Anthems", "Heavy Bass"],
        "tempo": "Driving & fast • 130–150 BPM"
    },
    "surprise": {
        "emoji": "✨",
        "title": "Curious & Excited",
        "subtitle": "Sparked by something unexpected",
        "blurb": "Engaged and wide awake! Here are vibrant synths, fresh electronic drops, and dynamic melodies that keep the energy flowing.",
        "color_bgr": (0, 180, 250),
        "hex": "#F59E0B",
        "genres": ["Nu-Disco", "Electronic Pop", "Synthwave", "Future Beats"],
        "tempo": "Bouncy & dynamic • 120–135 BPM"
    },
    "neutral": {
        "emoji": "☕",
        "title": "Calm & Centered",
        "subtitle": "Easygoing focus and peaceful flow",
        "blurb": "You look relaxed and centered. Perfect for mellow lo-fi beats, gentle café acoustics, and smooth rhythms to accompany your workflow.",
        "color_bgr": (220, 120, 160),
        "hex": "#8B5CF6",
        "genres": ["Lo-Fi Chill", "Quiet Jazz", "Study Beats", "Ambient Piano"],
        "tempo": "Easy & moderate • 75–95 BPM"
    },
    "rock": {
        "emoji": "🎸",
        "title": "Playful & Rebellious",
        "subtitle": "Ready for loud guitars and good times",
        "blurb": "You've thrown the rock sign! Get ready for roaring guitars, punchy drum breaks, and unmistakable rock-and-roll attitude.",
        "color_bgr": (200, 50, 220),
        "hex": "#EC4899",
        "genres": ["Classic Rock", "Indie Anthems", "Garage Rock", "Grunge"],
        "tempo": "Punchy & driving • 120–140 BPM"
    },
    "Thumbsup": {
        "emoji": "👍",
        "title": "Good Vibes Approved",
        "subtitle": "Feeling great and ready to groove",
        "blurb": "Thumbs up! You're in a great groove. We're serving up crowd favorites, chart-topping hits, and songs you'll want to sing along with.",
        "color_bgr": (240, 200, 0),
        "hex": "#06B6D4",
        "genres": ["Top Hits", "Sing-Along Jams", "Feel-Good R&B", "Modern Pop"],
        "tempo": "Catchy & steady • 105–120 BPM"
    },
    "hello": {
        "emoji": "👋",
        "title": "Warm & Welcoming",
        "subtitle": "Starting the session off right",
        "blurb": "Hello there! Let's start on a bright note with inviting acoustic strings, sunny morning melodies, and upbeat coffeehouse tunes.",
        "color_bgr": (180, 220, 0),
        "hex": "#14B8A6",
        "genres": ["Warm Acoustic", "Morning Folk", "Breezy Pop", "Indie Coffeehouse"],
        "tempo": "Light & breezy • 90–110 BPM"
    },
    "No": {
        "emoji": "🍃",
        "title": "Quiet Peace",
        "subtitle": "Unwinding and stepping back",
        "blurb": "Taking a breath away from the noise. We're selecting soothing ambient sounds, peaceful instruments, and music to help you decompress.",
        "color_bgr": (160, 160, 160),
        "hex": "#64748B",
        "genres": ["Minimalist Strings", "Deep Focus", "Tranquil Ambient", "Meditation"],
        "tempo": "Slow & meditative • 50–70 BPM"
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
        opt_hand = mp_vision.HandLandmarkerOptions(base_options=base_hand, num_hands=2)
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
        return {"emotion": raw, "confidence": 0.0, "top3": []}
    except Exception:
        return {"emotion": "", "confidence": 0.0, "top3": []}


# -------------------------------------------------------------
# VIDEO PROCESSOR WITH MULTI-COLOR DOTS (Face vs Hands)
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

        # Color definitions for Face vs Hands:
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

                # Custom multi-colored landmarks
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

        # Run Prediction
        top_pred = ""
        confidence = 0.0
        mood_accent_color = (255, 255, 255)

        if len(lst) == 1020:
            lst_arr = np.array(lst).reshape(1, -1)
            try:
                probs = model.predict(lst_arr, verbose=0)[0]
                top_idx = int(np.argmax(probs))
                top_pred = str(label[top_idx])
                confidence = float(probs[top_idx]) * 100.0

                top_indices = np.argsort(probs)[::-1]
                top3_info = [(str(label[i]), round(float(probs[i]) * 100, 1)) for i in top_indices[:3]]

                if top_pred in MOOD_CATALOGUE:
                    mood_accent_color = MOOD_CATALOGUE[top_pred]["color_bgr"]

                # Save state safely
                payload = {
                    "emotion": top_pred,
                    "confidence": round(confidence, 1),
                    "top3": top3_info,
                    "face_detected": True,
                    "hands": {"left": left_hand_present, "right": right_hand_present},
                    "timestamp": time.time()
                }
                with open("detected_state.tmp", "w") as f:
                    json.dump(payload, f)
                os.replace("detected_state.tmp", "detected_state.json")
                np.save("detected_emotion.npy", np.array([top_pred]))
            except Exception:
                pass

        # ---------------------------------------------
        # CAMERA HUD OVERLAY (Refined, High-Contrast)
        # ---------------------------------------------
        overlay = frm.copy()
        # Top HUD Bar
        cv2.rectangle(overlay, (0, 0), (w, 80), (15, 20, 30), -1)
        # Bottom Telemetry Bar
        cv2.rectangle(overlay, (0, h - 40), (w, h), (15, 20, 30), -1)
        cv2.addWeighted(overlay, 0.72, frm, 0.28, 0, frm)

        if face_detected and top_pred:
            readable_title = MOOD_CATALOGUE.get(top_pred, {}).get("title", top_pred.title())
            cv2.putText(frm, f"MOOD: {readable_title.upper()}", (20, 34), cv2.FONT_HERSHEY_DUPLEX, 0.85, mood_accent_color, 2)
            cv2.putText(frm, f"Confidence: {confidence:.1f}%", (20, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 225, 235), 1)

            # Confidence bar on camera
            bar_start = 220
            bar_len = max(50, w - bar_start - 30)
            fill_len = int(bar_len * (confidence / 100.0))
            cv2.rectangle(frm, (bar_start, 50), (bar_start + bar_len, 64), (35, 45, 60), -1)
            if fill_len > 0:
                cv2.rectangle(frm, (bar_start, 50), (bar_start + fill_len, 64), mood_accent_color, -1)
        else:
            cv2.putText(frm, "CENTER YOUR FACE IN CAMERA", (20, 38), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 140, 255), 2)
            cv2.putText(frm, "Looking for facial landmarks...", (20, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 190, 205), 1)

        # Bottom Bar Status
        face_str = "FACE: LOCKED" if face_detected else "FACE: NONE"
        face_col = (0, 220, 120) if face_detected else (0, 100, 255)
        cv2.putText(frm, face_str, (20, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, face_col, 1)

        l_txt = "L-HAND: ACTIVE" if left_hand_present else "L-HAND: OFF"
        cv2.putText(frm, l_txt, (180, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_LEFT_HAND if left_hand_present else (150, 150, 160), 1)

        r_txt = "R-HAND: ACTIVE" if right_hand_present else "R-HAND: OFF"
        cv2.putText(frm, r_txt, (350, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_RIGHT_HAND if right_hand_present else (150, 150, 160), 1)

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
                Music that understands how you feel. Express a mood or hand gesture to curate your soundtrack.
            </p>
        </div>
        <div style="display: flex; gap: 8px;">
            <span class="mood-pill">✨ 468-Point Face Mesh</span>
            <span class="mood-pill">🖐️ Dual Hand Tracking</span>
            <span class="mood-pill">🎧 Instant Recommendations</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Fetch latest state
telemetry = read_telemetry()
current_emotion = telemetry.get("emotion", "")
current_conf = telemetry.get("confidence", 0.0)
current_top3 = telemetry.get("top3", [])

mood_info = MOOD_CATALOGUE.get(current_emotion, {
    "emoji": "🎧",
    "title": "Awaiting Your Expression",
    "subtitle": "Look into the camera to begin",
    "blurb": "Allow camera access and face the camera. Whether you're feeling joyful, calm, or energetic, we'll recommend songs that fit your exact mood.",
    "hex": "#EA580C",
    "genres": ["Feel-Good Pop", "Acoustic", "Lo-Fi Chill", "Classic Rock"],
    "tempo": "Ready to tune in"
})


# -------------------------------------------------------------
# MAIN 2-COLUMN STUDIO LAYOUT (Native Streamlit Containers)
# -------------------------------------------------------------
col_left, col_right = st.columns([1.1, 1], gap="large")

# LEFT COLUMN: Live Camera Feed & Expression Distribution
with col_left:
    with st.container(border=True):
        st.subheader("📹 Live Camera Mirror")
        st.caption("The AI detects your facial expressions and hand gestures in real time.")

        RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

        webrtc_streamer(
            key="sangeet-camera-streamer",
            desired_playing_state=True,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=EmotionDetector
        )

        st.caption("Color Guide: **Cyan** = Face Mesh | **Magenta** = Left Hand | **Green** = Right Hand")

    # Expression Balance Card
    with st.container(border=True):
        st.subheader("📊 Detected Expression Balance")

        if current_top3:
            for emo_code, pct in current_top3:
                info = MOOD_CATALOGUE.get(emo_code, {"emoji": "🎵", "title": emo_code.title()})
                c_lbl, c_bar, c_pct = st.columns([2.5, 4.5, 1])
                with c_lbl:
                    st.write(f"{info['emoji']} **{info['title']}**")
                with c_bar:
                    st.progress(min(1.0, max(0.0, pct / 100.0)))
                with c_pct:
                    st.write(f"**{pct:.0f}%**")
        else:
            st.info("💡 Turn on your camera above to see live expression balance.")

        st.caption("💡 Try smiling, raising your eyebrows, or making a thumbs up 👍 or rock sign 🤘!")

# RIGHT COLUMN: Mood Hero & Music Recommendation Hub
with col_right:
    # Detected Mood Card (Native Streamlit container with clean styling)
    with st.container(border=True):
        col_m_emoji, col_m_text = st.columns([1, 4])
        with col_m_emoji:
            st.markdown(f"<div style='font-size: 3.8rem; text-align: center; padding-top: 5px;'>{mood_info['emoji']}</div>", unsafe_allow_html=True)
        with col_m_text:
            st.caption(f"DETECTED MOOD • {current_conf:.0f}% CONFIDENCE")
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

        keyword = current_emotion if current_emotion else "feel good"
        search_query = f"{lang} {keyword} songs {artist}".strip()
        youtube_url = f"https://www.youtube.com/results?search_query={search_query.replace(' ', '+')}"
        spotify_url = f"https://open.spotify.com/search/{search_query.replace(' ', '%20')}"

        btn_c1, btn_c2 = st.columns([2, 1])
        with btn_c1:
            get_rec_clicked = st.button("🎵 Curate Songs for My Mood", use_container_width=True, type="primary")
        with btn_c2:
            if st.button("Reset Mood 🔄", use_container_width=True):
                np.save("detected_emotion.npy", np.array([""]))
                if os.path.exists("detected_state.json"):
                    os.remove("detected_state.json")
                st.rerun()

        if get_rec_clicked:
            if not current_emotion:
                st.warning("⚠️ Look into the camera for a moment to let us capture your mood first!")
            else:
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
    "Sangeet AI Studio • Facial Mesh & Gesture Recognition Powered by MediaPipe and TensorFlow"
    "</div>",
    unsafe_allow_html=True
)
