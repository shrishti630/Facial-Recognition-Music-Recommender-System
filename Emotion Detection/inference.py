import cv2
import numpy as np

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

model = load_model("model.h5")
label = np.load("labels.npy")

hol = mp_holistic.Holistic()

# Establishing connection to the webcam camera
cap = cv2.VideoCapture(0)

while True:
    lst = []

    ret, frm = cap.read()
    if not ret or frm is None:
        continue

    frm = cv2.flip(frm, 1)

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
            cv2.putText(frm, pred, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        except Exception:
            pass

        mp_drawing.draw_landmarks(frm, res.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    else:
        cv2.putText(frm, "No face detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if res.left_hand_landmarks:
        mp_drawing.draw_landmarks(frm, res.left_hand_landmarks, mp_hands.HAND_CONNECTIONS)
    if res.right_hand_landmarks:
        mp_drawing.draw_landmarks(frm, res.right_hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show image back to screen
    cv2.imshow("Emotion Detection", frm)

    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        cap.release()
        break