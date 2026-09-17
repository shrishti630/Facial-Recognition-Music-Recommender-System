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
# PAGE CONFIGURATION (WARM EDITORIAL LIGHT THEME)
# -------------------------------------------------------------
st.set_page_config(
    page_title="Sangeet — Music by Mood",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------------------------------------
# HUMANIZED EDITORIAL LIGHT STYLING
# -------------------------------------------------------------
st.markdown("""
<style>
    /* Global Base */
    .stApp {
        background-color: #FAF8F5;
        color: #1E293B;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Editorial Header */
    .editorial-header {
        padding: 24px 0 16px 0;
        border-bottom: 1px solid #EAE6DF;
        margin-bottom: 30px;
    }
    .editorial-title {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, serif;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #0F172A;
        margin: 0;
    }
    .editorial-sub {
        font-size: 1.05rem;
        color: #64748B;
        margin-top: 6px;
        line-height: 1.5;
    }
    
    /* Clean Cards */
    .clean-card {
        background: #FFFFFF;
        border: 1px solid #EAE6DF;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 22px;
        box-shadow: 0 4px 20px -4px rgba(0, 0, 0, 0.03), 0 2px 6px -2px rgba(0, 0, 0, 0.02);
    }
    
    .card-label {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8;
        margin-bottom: 12px;
    }
    
    /* Mood Card */
    .mood-hero-card {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 26px;
        margin-bottom: 22px;
        border: 1.5px solid #E2E8F0;
        box-shadow: 0 6px 24px -4px rgba(0, 0, 0, 0.04);
    }
    
    /* Genre Pills */
    .genre-chip {
        display: inline-block;
        background: #F8FAFC;
        color: #334155;
        border: 1px solid #E2E8F0;
        border-radius: 100px;
        padding: 5px 14px;
        font-size: 0.82rem;
        font-weight: 500;
        margin: 4px 6px 4px 0;
    }
    
    /* Progress styling */
    .stProgress > div > div > div > div {
        background-color: #EA580C;
    }
    
    /* Clean inputs & buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    /* Hide Streamlit default headers */
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

# -------------------------------------------------------------
# HUMANIZED MOOD CATALOGUE
# -------------------------------------------------------------
MOOD_CATALOGUE = {
    "happy": {
        "emoji": "☀️",
        "title": "Bright & Joyful",
        "subtitle": "Radiating positive energy",
        "blurb": "Your expression is full of warmth and lightness. We're matching your state with uplifting melodies, feel-good rhythms, and cheerful anthems.",
        "badge_bg": "#FEF3C7",
        "badge_color": "#92400E",
        "accent": "#D97706",
        "genres": ["Feel-Good Pop", "Sunny Acoustics", "Disco Groove", "Indie Upbeat"],
        "tempo": "Brisk & lively • 115–128 BPM"
    },
    "sad": {
        "emoji": "🌧️",
        "title": "Gentle & Reflective",
        "subtitle": "Taking a quiet, soulful moment",
        "blurb": "Quiet moments have their own beauty. Here are comforting acoustic tracks, soulful storytelling, and melodies that offer gentle, warm company.",
        "badge_bg": "#DBEAFE",
        "badge_color": "#1E40AF",
        "accent": "#2563EB",
        "genres": ["Mellow Acoustic", "Soulful Ballads", "Piano Solos", "Late Night Indie"],
        "tempo": "Slow & peaceful • 65–85 BPM"
    },
    "angry": {
        "emoji": "⚡",
        "title": "Fiery & High Voltage",
        "subtitle": "Channeling pure drive and intensity",
        "blurb": "Channel that fire into momentum. We're selecting punchy rock tracks, hard-hitting rhythms, and adrenaline-charged tracks.",
        "badge_bg": "#FEE2E2",
        "badge_color": "#991B1B",
        "accent": "#DC2626",
        "genres": ["Alternative Rock", "Driving Beats", "Power Anthems", "Heavy Bass"],
        "tempo": "Driving & fast • 130–150 BPM"
    },
    "surprise": {
        "emoji": "✨",
        "title": "Curious & Excited",
        "subtitle": "Sparked by something unexpected",
        "blurb": "Engaged and wide awake! Here are vibrant synths, fresh electronic drops, and dynamic melodies that keep the energy flowing.",
        "badge_bg": "#FFEDD5",
        "badge_color": "#9A3412",
        "accent": "#EA580C",
        "genres": ["Nu-Disco", "Electronic Pop", "Synthwave", "Future Beats"],
        "tempo": "Bouncy & dynamic • 120–135 BPM"
    },
    "neutral": {
        "emoji": "☕",
        "title": "Calm & Centered",
        "subtitle": "Easygoing focus and peaceful flow",
        "blurb": "You look relaxed and centered. Perfect for mellow lo-fi beats, gentle café acoustics, and smooth rhythms to accompany your workflow.",
        "badge_bg": "#F3E8FF",
        "badge_color": "#6B21A8",
        "accent": "#7C3AED",
        "genres": ["Lo-Fi Chill", "Quiet Jazz", "Study Beats", "Ambient Piano"],
        "tempo": "Easy & moderate • 75–95 BPM"
    },
    "rock": {
        "emoji": "🎸",
        "title": "Playful & Rebellious",
        "subtitle": "Ready for loud guitars and good times",
        "blurb": "You've thrown the rock sign! Get ready for roaring guitars, punchy drum breaks, and unmistakable rock-and-roll attitude.",
        "badge_bg": "#FCE7F3",
        "badge_color": "#9D174D",
        "accent": "#DB2777",
        "genres": ["Classic Rock", "Indie Anthems", "Garage Rock", "Grunge"],
        "tempo": "Punchy & driving • 120–140 BPM"
    },
    "Thumbsup": {
        "emoji": "👍",
        "title": "Good Vibes Approved",
        "subtitle": "Feeling great and ready to groove",
        "blurb": "Thumbs up! You're in a great groove. We're serving up crowd favorites, chart-topping hits, and songs you'll want to sing along with.",
        "badge_bg": "#CCFBF1",
        "badge_color": "#115E59",
        "accent": "#0D9488",
        "genres": ["Top Hits", "Sing-Along Jams", "Feel-Good R&B", "Modern Pop"],
        "tempo": "Catchy & steady • 105–120 BPM"
    },
    "hello": {
        "emoji": "👋",
        "title": "Warm & Welcoming",
        "subtitle": "Starting the session off right",
        "blurb": "Hello there! Let's start on a bright note with inviting acoustic strings, sunny morning melodies, and upbeat coffeehouse tunes.",
        "badge_bg": "#E0F2FE",
        "badge_color": "#075985",
        "accent": "#0284C7",
        "genres": ["Warm Acoustic", "Morning Folk", "Breezy Pop", "Indie Coffeehouse"],
        "tempo": "Light & breezy • 90–110 BPM"
    },
    "No": {
        "emoji": "🍃",
        "title": "Quiet Peace",
        "subtitle": "Unwinding and stepping back",
        "blurb": "Taking a breath away from the noise. We're selecting soothing ambient sounds, peaceful instruments, and music to help you decompress.",
        "badge_bg": "#F1F5F9",
        "badge_color": "#334155",
        "accent": "#475569",
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
# VIDEO PROCESSOR (MINIMAL, ELEGANT HUMAN OVERLAY)
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

                if left_hand:
                    left_hand_present = True
                    ref_l = left_hand[8]
                    for lm in left_hand:
                        lst.append(lm.x - ref_l.x)
                        lst.append(lm.y - ref_l.y)
                        cv2.circle(frm, (int(lm.x * w), int(lm.y * h)), 2, (180, 140, 240), -1)
                else:
                    for _ in range(42):
                        lst.append(0.0)

                if right_hand:
                    right_hand_present = True
                    ref_r = right_hand[8]
                    for lm in right_hand:
                        lst.append(lm.x - ref_r.x)
                        lst.append(lm.y - ref_r.y)
                        cv2.circle(frm, (int(lm.x * w), int(lm.y * h)), 2, (140, 200, 240), -1)
                else:
                    for _ in range(42):
                        lst.append(0.0)

                # Very subtle, refined facial landmark dots (not harsh wireframes)
                for lm in face_lms[:468:6]:
                    cv2.circle(frm, (int(lm.x * w), int(lm.y * h)), 1, (255, 255, 255), -1)

        # Run Prediction
        top_pred = ""
        confidence = 0.0

        if len(lst) == 1020:
            lst_arr = np.array(lst).reshape(1, -1)
            try:
                probs = model.predict(lst_arr, verbose=0)[0]
                top_idx = int(np.argmax(probs))
                top_pred = str(label[top_idx])
                confidence = float(probs[top_idx]) * 100.0

                top_indices = np.argsort(probs)[::-1]
                top3_info = [(str(label[i]), round(float(probs[i]) * 100, 1)) for i in top_indices[:3]]

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
        # MINIMAL, HUMANIZED FLOATING BANNER (NOT A SCI-FI HUD)
        # ---------------------------------------------
        overlay = frm.copy()
        # Soft dark charcoal floating pill at top
        cv2.rectangle(overlay, (20, 18), (w - 20, 72), (24, 28, 38), -1)
        cv2.addWeighted(overlay, 0.70, frm, 0.30, 0, frm)

        if face_detected and top_pred:
            readable_title = MOOD_CATALOGUE.get(top_pred, {}).get("title", top_pred.title())
            cv2.putText(frm, f"Current feeling: {readable_title}", (36, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(frm, f"{confidence:.0f}% confidence", (w - 180, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 210, 225), 1)

            # Clean thin indicator line at bottom of pill
            bar_w = int((w - 72) * (confidence / 100.0))
            cv2.line(frm, (36, 62), (36 + bar_w, 62), (234, 88, 12), 2)
        else:
            cv2.putText(frm, "Center your face in the camera view...", (36, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (220, 225, 235), 1)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")


# -------------------------------------------------------------
# EDITORIAL HEADER (Humanized, natural tone)
# -------------------------------------------------------------
st.markdown("""
<div class="editorial-header">
    <div style="display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap; gap: 15px;">
        <div>
            <span style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.12em; color: #EA580C;">
                Intelligent Mood Listening
            </span>
            <h1 class="editorial-title">Sangeet</h1>
            <p class="editorial-sub">
                Music that understands how you feel. Take a glance at your camera and let your mood curate the soundtrack.
            </p>
        </div>
        <div style="color: #64748B; font-size: 0.88rem; background: #FFFFFF; border: 1px solid #EAE6DF; border-radius: 12px; padding: 8px 16px;">
            ✨ Powered by facial landmark reading & deep learning
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Fetch latest mood data
telemetry = read_telemetry()
current_emotion = telemetry.get("emotion", "")
current_conf = telemetry.get("confidence", 0.0)
current_top3 = telemetry.get("top3", [])

mood_data = MOOD_CATALOGUE.get(current_emotion, {
    "emoji": "🎵",
    "title": "Awaiting Your Expression",
    "subtitle": "Look into the camera to begin",
    "blurb": "Position your face in the camera frame. Whether you're feeling joyful, calm, or energetic, we'll recommend songs that fit your exact state of mind.",
    "badge_bg": "#F1F5F9",
    "badge_color": "#475569",
    "accent": "#EA580C",
    "genres": ["Acoustic", "Pop", "Indie", "Lo-Fi", "Rock"],
    "tempo": "Tailored to your mood"
})


# -------------------------------------------------------------
# MAIN 2-COLUMN EDITORIAL LAYOUT
# -------------------------------------------------------------
left_col, right_col = st.columns([1.1, 1], gap="large")

# LEFT: Camera Stream & Expression Breakdown
with left_col:
    st.markdown('<div class="clean-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-label">Live Camera Mirror</div>', unsafe_allow_html=True)
    st.caption("Allow camera access below. The camera detects your natural expressions and hand gestures in real time.")

    RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    webrtc_streamer(
        key="editorial-camera",
        desired_playing_state=True,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=EmotionDetector
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Expression Breakdown Card
    st.markdown('<div class="clean-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-label">Expression Breakdown</div>', unsafe_allow_html=True)

    if current_top3:
        for emo_code, pct in current_top3:
            info = MOOD_CATALOGUE.get(emo_code, {"emoji": "🎵", "title": emo_code.title()})
            c_name, c_bar, c_val = st.columns([2.5, 4.5, 1])
            with c_name:
                st.write(f"{info['emoji']} **{info['title']}**")
            with c_bar:
                st.progress(min(1.0, max(0.0, pct / 100.0)))
            with c_val:
                st.write(f"{pct:.0f}%")
    else:
        st.markdown("""
        <p style="color: #64748B; font-size: 0.95rem; margin: 0;">
            Turn on the camera above. Your expression analysis will update automatically.
        </p>
        """, unsafe_allow_html=True)

    st.caption("Tip: Try smiling naturally, raising your eyebrows, or giving a thumbs up 👍 or rock sign 🤘!")
    st.markdown('</div>', unsafe_allow_html=True)

# RIGHT: Editorial Soundtrack & Recommendations
with right_col:
    # Editorial Mood Hero Card
    accent_color = mood_data.get("accent", "#EA580C")
    st.markdown(f"""
    <div class="mood-hero-card" style="border-left: 5px solid {accent_color};">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <span style="background: {mood_data['badge_bg']}; color: {mood_data['badge_color']}; font-size: 0.8rem; font-weight: 700; padding: 4px 12px; border-radius: 20px;">
                Detected Mood • {current_conf:.0f}% Match
            </span>
            <span style="font-size: 0.85rem; color: #64748B;">{mood_data['tempo']}</span>
        </div>
        
        <div style="display: flex; align-items: flex-start; gap: 18px; margin: 15px 0;">
            <div style="font-size: 3.4rem; line-height: 1; padding-top: 4px;">{mood_data['emoji']}</div>
            <div>
                <h2 style="margin: 0; font-size: 1.7rem; font-weight: 700; color: #0F172A; letter-spacing: -0.01em;">
                    {mood_data['title']}
                </h2>
                <div style="color: {accent_color}; font-weight: 600; font-size: 0.92rem; margin-top: 2px;">
                    {mood_data['subtitle']}
                </div>
                <p style="margin: 8px 0 0 0; color: #475569; font-size: 0.94rem; line-height: 1.55;">
                    {mood_data['blurb']}
                </p>
            </div>
        </div>
        
        <div style="margin-top: 16px; padding-top: 14px; border-top: 1px solid #F1F5F9;">
            <div style="font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: #94A3B8; margin-bottom: 6px;">
                Recommended Genres & Styles
            </div>
            {" ".join([f'<span class="genre-chip">{g}</span>' for g in mood_data['genres']])}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Music Personalization Box
    st.markdown('<div class="clean-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-label">Fine-tune Your Music</div>', unsafe_allow_html=True)

    p_col1, p_col2 = st.columns(2)
    with p_col1:
        lang = st.text_input("Language or Region", value="Hindi")
    with p_col2:
        artist = st.text_input("Preferred Artist", value="Arijit Singh")

    st.caption("Quick Select Artists:")
    artist_chips = ["Arijit Singh", "Taylor Swift", "Diljit Dosanjh", "Coldplay"]
    chip_row = st.columns(4)
    for idx, name in enumerate(artist_chips):
        if chip_row[idx].button(name, key=f"art_{idx}", use_container_width=True):
            artist = name
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # Action Hub
    st.markdown('<div class="clean-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-label">Listen & Explore</div>', unsafe_allow_html=True)

    # Search Query Construction
    mood_keyword = current_emotion if current_emotion else "peaceful melody"
    search_query = f"{lang} {mood_keyword} songs {artist}".strip()
    youtube_url = f"https://www.youtube.com/results?search_query={search_query.replace(' ', '+')}"
    spotify_url = f"https://open.spotify.com/search/{search_query.replace(' ', '%20')}"

    btn_col1, btn_col2 = st.columns([2, 1])
    with btn_col1:
        recommend_clicked = st.button("🎵 Curate Songs for My Mood", use_container_width=True, type="primary")
    with btn_col2:
        if st.button("Reset Mood", use_container_width=True):
            np.save("detected_emotion.npy", np.array([""]))
            if os.path.exists("detected_state.json"):
                os.remove("detected_state.json")
            st.rerun()

    if recommend_clicked:
        if not current_emotion:
            st.warning("Please look at the camera for a moment to let us sense your mood first!")
        else:
            try:
                webbrowser.open(youtube_url)
            except Exception:
                pass
            st.success(f"Soundtrack curated for **{mood_data['title']}**! Choose your player below:")

    st.markdown("<div style='margin-top: 14px;'>", unsafe_allow_html=True)
    l_c1, l_c2 = st.columns(2)
    with l_c1:
        st.link_button("▶️ Open on YouTube", youtube_url, use_container_width=True)
    with l_c2:
        st.link_button("🟢 Open on Spotify", spotify_url, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; color: #94A3B8; font-size: 0.85rem; padding: 25px 0 15px 0;">
    Sangeet • An emotional listening experience powered by MediaPipe & Deep Learning
</div>
""", unsafe_allow_html=True)
