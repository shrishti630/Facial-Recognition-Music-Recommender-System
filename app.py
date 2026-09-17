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

try:
    import mediapipe.solutions.holistic as mp_holistic
    import mediapipe.solutions.hands as mp_hands
    import mediapipe.solutions.drawing_utils as mp_drawing
except (AttributeError, ModuleNotFoundError):
    try:
        from mediapipe.python.solutions import holistic as mp_holistic
        from mediapipe.python.solutions import hands as mp_hands
        from mediapipe.python.solutions import drawing_utils as mp_drawing
    except (AttributeError, ModuleNotFoundError):
        import mediapipe as mp
        mp_holistic = mp.solutions.holistic
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils

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

model = get_model()
label = get_labels()
hol = mp_holistic.Holistic()

try:
    detected_emotion = str(np.load("detected_emotion.npy")[0])
except Exception:
    detected_emotion = ""

class EmotionDetector:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.flip(frm, 1)

        res = hol.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        lst = []

        # Only predict if face landmarks are detected to avoid shape mismatch crash
        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            lst = np.array(lst).reshape(1, -1)

            try:
                pred = label[np.argmax(model.predict(lst, verbose=0))]
                cv2.putText(frm, pred, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                np.save("detected_emotion.npy", np.array([pred]))
            except Exception:
                pass

            mp_drawing.draw_landmarks(frm, res.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
        else:
            cv2.putText(frm, "Face not detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if res.left_hand_landmarks:
            mp_drawing.draw_landmarks(frm, res.left_hand_landmarks, mp_hands.HAND_CONNECTIONS)
        if res.right_hand_landmarks:
            mp_drawing.draw_landmarks(frm, res.right_hand_landmarks, mp_hands.HAND_CONNECTIONS)

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


