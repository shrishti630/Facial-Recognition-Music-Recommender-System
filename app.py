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

# Page Configuration
st.set_page_config(
    page_title="Sangeet - AI Music Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Styling
st.markdown("""
<style>
    /* Global styles */
    .stApp {
        background-color: #0A0D14;
        color: #F1F5F9;
    }
    
    /* Header hero */
    .hero-container {
        background: linear-gradient(135deg, rgba(30, 27, 75, 0.7) 0%, rgba(15, 23, 42, 0.8) 50%, rgba(10, 13, 20, 0.9) 100%);
        border: 1px solid rgba(139, 92, 246, 0.25);
        border-radius: 20px;
        padding: 24px 30px;
        margin-bottom: 25px;
        box-shadow: 0 12px 36px rgba(0, 0, 0, 0.45);
        backdrop-filter: blur(12px);
    }
    
    /* Card containers */
    .glass-card {
        background: rgba(19, 23, 34, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
    }
    
    /* Mood Card with dynamic neon glow */
    .mood-card {
        background: rgba(19, 23, 34, 0.85);
        border-radius: 18px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        transition: all 0.3s ease;
    }
    
    /* Tag Pills */
    .mood-tag {
        display: inline-block;
        background: rgba(139, 92, 246, 0.15);
        color: #C4B5FD;
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.82rem;
        font-weight: 500;
        margin: 3px;
    }
    
    /* Hide default Streamlit chrome */
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

# Mood metadata mapping
MOOD_MAP = {
    "happy": {
        "emoji": "😊",
        "title": "Joyful & Energetic",
        "color": (0, 220, 120),
        "hex": "#10B981",
        "vibe": "Upbeat feel-good melodies, cheerful pop anthems, and dance beats",
        "tags": ["#HappyVibes", "#FeelGood", "#DancePop", "#SunnyHits"]
    },
    "sad": {
        "emoji": "🌧️",
        "title": "Soulful & Melancholic",
        "color": (255, 160, 60),
        "hex": "#3B82F6",
        "vibe": "Deep acoustic ballads, gentle piano melodies, and reflective lyrics",
        "tags": ["#Soulful", "#Acoustic", "#DeepEmotions", "#SlowBallad"]
    },
    "angry": {
        "emoji": "🔥",
        "title": "Intense & High Voltage",
        "color": (60, 60, 255),
        "hex": "#EF4444",
        "vibe": "Hard rock riffs, adrenaline-pumping beats, and heavy energy",
        "tags": ["#HardRock", "#WorkoutPump", "#Adrenaline", "#HeavyBass"]
    },
    "surprise": {
        "emoji": "⚡",
        "title": "Excited & Wonder",
        "color": (0, 180, 255),
        "hex": "#F59E0B",
        "vibe": "Electric synth drops, progressive EDM, and catchy build-ups",
        "tags": ["#Synthwave", "#EDMDrops", "#PartyBeats", "#FutureBass"]
    },
    "neutral": {
        "emoji": "☕",
        "title": "Calm & Centered",
        "color": (230, 130, 180),
        "hex": "#8B5CF6",
        "vibe": "Chilled Lo-Fi beats, soft jazz café acoustics, and study rhythms",
        "tags": ["#LoFiChill", "#Coffeehouse", "#StudyBeats", "#SmoothJazz"]
    },
    "rock": {
        "emoji": "🤘",
        "title": "Rock & Rebellion",
        "color": (200, 50, 220),
        "hex": "#EC4899",
        "vibe": "Classic rock anthems, distorted guitars, and alternative grooves",
        "tags": ["#ClassicRock", "#Alternative", "#GuitarSolo", "#IndieGroove"]
    },
    "Thumbsup": {
        "emoji": "👍",
        "title": "Approved & Groovy",
        "color": (255, 200, 0),
        "hex": "#06B6D4",
        "vibe": "Top trending viral hits, irresistible groove, and radio favorites",
        "tags": ["#TopHits", "#ViralJam", "#GroovyVibes", "#TrendingNow"]
    },
    "hello": {
        "emoji": "👋",
        "title": "Warm & Welcoming",
        "color": (160, 220, 0),
        "hex": "#14B8A6",
        "vibe": "Bright acoustic guitars, sunny morning tunes, and uplifting indie folk",
        "tags": ["#MorningAcoustic", "#WarmVibes", "#IndieFolk", "#SunnyDays"]
    },
    "No": {
        "emoji": "🛑",
        "title": "Peaceful & Tranquil",
        "color": (180, 180, 180),
        "hex": "#6B7280",
        "vibe": "Deep meditation music, ambient soundscapes, and calming piano",
        "tags": ["#Tranquil", "#DeepFocus", "#AmbientSound", "#InnerPeace"]
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
    """Reads latest prediction state safely."""
    try:
        if os.path.exists("detected_state.json"):
            with open("detected_state.json", "r") as f:
                data = json.load(f)
                return data
    except Exception:
        pass
    try:
        raw = str(np.load("detected_emotion.npy")[0])
        return {"emotion": raw, "confidence": 0.0, "top3": []}
    except Exception:
        return {"emotion": "", "confidence": 0.0, "top3": []}

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

                mp_drawing.draw_landmarks(frm, res.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
                if res.left_hand_landmarks:
                    mp_drawing.draw_landmarks(frm, res.left_hand_landmarks, mp_hands.HAND_CONNECTIONS)
                if res.right_hand_landmarks:
                    mp_drawing.draw_landmarks(frm, res.right_hand_landmarks, mp_hands.HAND_CONNECTIONS)
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
                        cv2.circle(frm, (int(lm.x * w), int(lm.y * h)), 3, (255, 0, 150), -1)
                else:
                    for _ in range(42):
                        lst.append(0.0)

                if right_hand:
                    right_hand_present = True
                    ref_r = right_hand[8]
                    for lm in right_hand:
                        lst.append(lm.x - ref_r.x)
                        lst.append(lm.y - ref_r.y)
                        cv2.circle(frm, (int(lm.x * w), int(lm.y * h)), 3, (0, 200, 255), -1)
                else:
                    for _ in range(42):
                        lst.append(0.0)

                # High-tech subtle cyan face points
                for lm in face_lms[:468:3]:
                    cv2.circle(frm, (int(lm.x * w), int(lm.y * h)), 1, (240, 200, 0), -1)

        # Predict when all 1,020 landmarks are available
        top_pred = "Scanning..."
        confidence = 0.0
        top_color = (200, 200, 200)

        if len(lst) == 1020:
            lst_arr = np.array(lst).reshape(1, -1)
            try:
                probs = model.predict(lst_arr, verbose=0)[0]
                top_idx = int(np.argmax(probs))
                top_pred = str(label[top_idx])
                confidence = float(probs[top_idx]) * 100.0

                top_indices = np.argsort(probs)[::-1]
                top3_info = [(str(label[i]), round(float(probs[i]) * 100, 1)) for i in top_indices[:3]]

                # Fetch color from mood map
                if top_pred in MOOD_MAP:
                    top_color = MOOD_MAP[top_pred]["color"]

                # Save telemetry safely to JSON
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

                # Also save numpy for backward compatibility
                np.save("detected_emotion.npy", np.array([top_pred]))
            except Exception:
                pass

        # -----------------------------
        # DRAW HEADS-UP DISPLAY (HUD)
        # -----------------------------
        overlay = frm.copy()
        cv2.rectangle(overlay, (0, 0), (w, 85), (10, 13, 20), -1)
        cv2.rectangle(overlay, (0, h - 45), (w, h), (10, 13, 20), -1)
        cv2.addWeighted(overlay, 0.75, frm, 0.25, 0, frm)

        if face_detected:
            # Top HUD: Emotion Title & Confidence Bar
            cv2.putText(frm, f"MOOD: {top_pred.upper()}", (20, 36), cv2.FONT_HERSHEY_DUPLEX, 0.9, top_color, 2)
            cv2.putText(frm, f"CONFIDENCE: {confidence:.1f}%", (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 225, 235), 1)

            # Interactive confidence meter bar on frame
            bar_start_x = 240
            bar_max_w = max(50, w - bar_start_x - 30)
            fill_w = int(bar_max_w * (confidence / 100.0))
            cv2.rectangle(frm, (bar_start_x, 52), (bar_start_x + bar_max_w, 68), (30, 35, 48), -1)
            if fill_w > 0:
                cv2.rectangle(frm, (bar_start_x, 52), (bar_start_x + fill_w, 68), top_color, -1)
        else:
            cv2.putText(frm, "LOOK DIRECTLY AT CAMERA", (20, 38), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 120, 255), 2)
            cv2.putText(frm, "Face Tracking: Searching for face...", (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 185, 195), 1)

        # Bottom Telemetry HUD
        face_status_str = "FACE: LOCKED" if face_detected else "FACE: LOST"
        face_status_col = (0, 220, 120) if face_detected else (0, 100, 255)
        cv2.putText(frm, face_status_str, (20, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, face_status_col, 1)

        hand_info = f"HANDS: L={'ON' if left_hand_present else 'OFF'} | R={'ON' if right_hand_present else 'OFF'}"
        cv2.putText(frm, hand_info, (170, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 205, 215), 1)

        cv2.putText(frm, "SANGEET AI HUD", (w - 160, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 120, 255), 1)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")


# -------------------------------------------------------------
# HERO BANNER
# -------------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap;">
        <div>
            <h1 style="margin: 0; font-size: 2.3rem; font-weight: 800; background: linear-gradient(90deg, #8B5CF6, #38BDF8, #34D399); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                🎵 Sangeet AI Studio
            </h1>
            <p style="margin: 8px 0 0 0; color: #94A3B8; font-size: 1.05rem;">
                Real-time Facial Emotion & Gesture Recognition paired with instant Music Recommendations.
            </p>
        </div>
        <div style="display: flex; gap: 10px; margin-top: 10px;">
            <span class="mood-tag">✨ 468 Face Landmarks</span>
            <span class="mood-tag">🖐️ Dual Hand Gestures</span>
            <span class="mood-tag">⚡ Live Telemetry</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Fetch latest state
telemetry = read_telemetry()
current_emotion = telemetry.get("emotion", "")
current_conf = telemetry.get("confidence", 0.0)
current_top3 = telemetry.get("top3", [])
mood_info = MOOD_MAP.get(current_emotion, {
    "emoji": "🎧",
    "title": "Awaiting Detection",
    "hex": "#64748B",
    "vibe": "Look at the camera and express an emotion or gesture to get music matched to your mood.",
    "tags": ["#MusicPlayer", "#Explore", "#Personalized"]
})

# -------------------------------------------------------------
# MAIN DASHBOARD: 2-COLUMN LAYOUT
# -------------------------------------------------------------
col_cam, col_ctrl = st.columns([1.15, 1], gap="medium")

# LEFT COLUMN: Live Camera & Vision Telemetry
with col_cam:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📹 Live Camera Feed & AI Telemetry")
    st.caption("Allow camera access. The AI models analyze your facial landmarks and gestures in real time.")

    RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    webrtc_streamer(
        key="emotion-studio-streamer",
        desired_playing_state=True,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=EmotionDetector
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Live Probabilities Breakdown Card
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### 📊 Model Probability Distribution")
    if current_top3:
        for emo_name, emo_pct in current_top3:
            emo_meta = MOOD_MAP.get(emo_name, {"emoji": "🎵", "hex": "#8B5CF6"})
            c1, c2, c3 = st.columns([1, 4, 1])
            with c1:
                st.write(f"{emo_meta['emoji']} **{emo_name.title()}**")
            with c2:
                st.progress(min(1.0, max(0.0, emo_pct / 100.0)))
            with c3:
                st.write(f"**{emo_pct}%**")
    else:
        st.info("💡 Turn on your camera to see live probability distributions here.")
    st.markdown('</div>', unsafe_allow_html=True)

# RIGHT COLUMN: Music Recommender Controls & Mood Studio
with col_ctrl:
    # Detected Mood Card
    border_color = mood_info.get("hex", "#8B5CF6")
    st.markdown(f"""
    <div class="mood-card" style="border: 1.5px solid {border_color};">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; color: #94A3B8; font-weight: 600;">Current Vibe</span>
            <span style="background: {border_color}22; color: {border_color}; border: 1px solid {border_color}55; border-radius: 12px; padding: 2px 10px; font-size: 0.85rem; font-weight: 600;">
                Confidence: {current_conf:.1f}%
            </span>
        </div>
        <div style="display: flex; align-items: center; gap: 16px; margin: 16px 0;">
            <div style="font-size: 3.2rem; line-height: 1;">{mood_info['emoji']}</div>
            <div>
                <h2 style="margin: 0; font-size: 1.8rem; font-weight: 700; color: #FFFFFF;">{mood_info['title']}</h2>
                <p style="margin: 4px 0 0 0; color: #94A3B8; font-size: 0.95rem;">{mood_info['vibe']}</p>
            </div>
        </div>
        <div style="margin-top: 12px;">
            {" ".join([f'<span class="mood-tag">{tag}</span>' for tag in mood_info['tags']])}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Preferences Box
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### 🎧 Music Preferences")

    col_l, col_a = st.columns(2)
    with col_l:
        lang = st.text_input("Language / Region", value="Hindi")
    with col_a:
        artist = st.text_input("Favorite Artist", value="Arijit Singh")

    # Quick Artist Chips
    st.caption("Quick Select Artists:")
    chip_cols = st.columns(4)
    artists_list = ["Arijit Singh", "Taylor Swift", "Diljit Dosanjh", "Coldplay"]
    for i, a_name in enumerate(artists_list):
        if chip_cols[i].button(a_name, key=f"chip_{i}", use_container_width=True):
            artist = a_name
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # Action Hub
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### 🚀 Get Recommendations")

    # Generate tailored search query
    query_mood = current_emotion if current_emotion else "feel good"
    search_query = f"{lang} {query_mood} songs {artist}".strip()
    yt_url = f"https://www.youtube.com/results?search_query={search_query.replace(' ', '+')}"
    spotify_url = f"https://open.spotify.com/search/{search_query.replace(' ', '%20')}"

    c_btn1, c_btn2 = st.columns([2, 1])
    with c_btn1:
        rec_clicked = st.button("🎵 Recommend Music Now", use_container_width=True)
    with c_btn2:
        if st.button("🔄 Reset Mood", use_container_width=True):
            np.save("detected_emotion.npy", np.array([""]))
            if os.path.exists("detected_state.json"):
                os.remove("detected_state.json")
            st.rerun()

    if rec_clicked:
        if not current_emotion:
            st.warning("⚠️ Please look into the camera to capture your mood first!")
        else:
            try:
                webbrowser.open(yt_url)
            except Exception:
                pass
            st.success(f"🎉 Curated recommendations ready for **{mood_info['title']}**!")

    # Direct launch buttons
    st.markdown("<div style='margin-top: 15px;'>", unsafe_allow_html=True)
    link_col1, link_col2 = st.columns(2)
    with link_col1:
        st.link_button("▶️ Open in YouTube", yt_url, use_container_width=True)
    with link_col2:
        st.link_button("🟢 Open in Spotify", spotify_url, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

st.write("---")
st.markdown(
    "<div style='text-align: center; color: #64748B; font-size: 0.85rem; padding: 10px;'>"
    "Sangeet AI • Powered by MediaPipe 468-Point Mesh, Keras Deep Learning & Streamlit"
    "</div>",
    unsafe_allow_html=True
)
