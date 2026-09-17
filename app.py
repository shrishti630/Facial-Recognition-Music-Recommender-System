import os
import urllib.request
import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import cv2
import numpy as np
import webbrowser

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

col1, col2, col3 = st.columns([1, 6, 1])

with col2:
    st.image("./images/Black and Red Music Studio Logo.png", width=300)

st.title("Sangeet")
st.write("Sangeet is an emotion detection based music recommendation system. Allow camera access below to detect your emotion and get personalized song recommendations.")

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

try:
    detected_emotion = str(np.load("detected_emotion.npy")[0])
except Exception:
    detected_emotion = ""

class EmotionDetector:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.flip(frm, 1)
        h, w, _ = frm.shape

        lst = []

        if detectors["mode"] == "legacy":
            hol = detectors["hol"]
            res = hol.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
            if res.face_landmarks:
                for i in res.face_landmarks.landmark:
                    lst.append(i.x - res.face_landmarks.landmark[1].x)
                    lst.append(i.y - res.face_landmarks.landmark[1].y)

                if res.left_hand_landmarks:
                    for i in res.left_hand_landmarks.landmark:
                        lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                        lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
                else:
                    for _ in range(42):
                        lst.append(0.0)

                if res.right_hand_landmarks:
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
                cv2.putText(frm, "Face not detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            face_det = detectors["face"]
            hand_det = detectors["hand"]

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
            res_face = face_det.detect(mp_image)
            res_hand = hand_det.detect(mp_image)

            if res_face.face_landmarks:
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
                    ref_l = left_hand[8]
                    for lm in left_hand:
                        lst.append(lm.x - ref_l.x)
                        lst.append(lm.y - ref_l.y)
                        cv2.circle(frm, (int(lm.x * w), int(lm.y * h)), 2, (0, 0, 255), -1)
                else:
                    for _ in range(42):
                        lst.append(0.0)

                if right_hand:
                    ref_r = right_hand[8]
                    for lm in right_hand:
                        lst.append(lm.x - ref_r.x)
                        lst.append(lm.y - ref_r.y)
                        cv2.circle(frm, (int(lm.x * w), int(lm.y * h)), 2, (255, 0, 0), -1)
                else:
                    for _ in range(42):
                        lst.append(0.0)

                for lm in face_lms[:468:4]:
                    cv2.circle(frm, (int(lm.x * w), int(lm.y * h)), 1, (0, 255, 0), -1)
            else:
                cv2.putText(frm, "Face not detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if len(lst) == 1020:
            lst = np.array(lst).reshape(1, -1)
            try:
                pred = label[np.argmax(model.predict(lst, verbose=0))]
                cv2.putText(frm, pred, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                np.save("detected_emotion.npy", np.array([pred]))
            except Exception:
                pass

        return av.VideoFrame.from_ndarray(frm, format="bgr24")

col_lang, col_artist = st.columns(2)
with col_lang:
    lang = st.text_input("Enter your preferred language", value="Hindi")
with col_artist:
    artist = st.text_input("Enter your preferred artist", value="Arijit Singh")

st.subheader("Webcam Emotion Detection")
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

webrtc_streamer(
    key="emotion-webrtc",
    desired_playing_state=True,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=EmotionDetector
)

# Display current detected emotion
if detected_emotion:
    st.info(f"🎭 Current Detected Emotion: **{detected_emotion.upper()}**")

col_rec, col_clear = st.columns([2, 1])

with col_rec:
    btn = st.button("Recommend Music 🎵", use_container_width=True)
with col_clear:
    clear_btn = st.button("Reset Emotion 🔄", use_container_width=True)

if clear_btn:
    np.save("detected_emotion.npy", np.array([""]))
    st.rerun()

if btn:
    if not detected_emotion:
        st.warning("Please allow the camera and let it detect your emotion first!")
    else:
        search_query = f"{lang} {detected_emotion} songs {artist}".strip()
        youtube_url = f"https://www.youtube.com/results?search_query={search_query.replace(' ', '+')}"
        try:
            webbrowser.open(youtube_url)
        except Exception:
            pass
        st.success(f"Finding recommendations for emotion: **{detected_emotion}**!")
        st.link_button("▶️ Open YouTube Recommendations", youtube_url, use_container_width=True)

st.write('---')

# Streamlit Customisation
st.markdown(""" <style>
header {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)
