import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyautogui
import time

# ---------------- LOAD MODEL ----------------
model = joblib.load("models/base_model.pkl")
encoder = joblib.load("models/label_encoder.pkl")

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

screen_w, screen_h = pyautogui.size()

# ---------------- SMOOTHING ----------------
prev_x, prev_y = 0, 0
smoothening = 6

# ---------------- ACTION CONTROL ----------------
last_left_click = 0
last_scroll_time = 0  # <-- added for scroll timing
click_delay = 1
scroll_delay = 0.2   # scroll every 0.2 seconds

confidence_threshold = 0.85

print("Press 'q' to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # --------- EXTRACT LANDMARKS ---------
            landmarks = []
            wrist = hand_landmarks.landmark[0]

            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x - wrist.x)
                landmarks.append(lm.y - wrist.y)
                landmarks.append(lm.z - wrist.z)

            landmarks = np.array(landmarks).reshape(1, -1)

            # --------- PREDICTION ---------
            probabilities = model.predict_proba(landmarks)[0]
            max_prob = np.max(probabilities)
            pred = np.argmax(probabilities)

            gesture = encoder.inverse_transform([pred])[0]

            # Display prediction
            cv2.putText(frame, f"{gesture} {max_prob:.2f}",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

            # --------- ACTIONS ---------
            if max_prob > confidence_threshold:

                # MOVE
                if gesture == "palm":
                    x = int(hand_landmarks.landmark[8].x * screen_w)
                    y = int(hand_landmarks.landmark[8].y * screen_h)

                    curr_x = prev_x + (x - prev_x) / smoothening
                    curr_y = prev_y + (y - prev_y) / smoothening

                    pyautogui.moveTo(curr_x, curr_y)
                    prev_x, prev_y = curr_x, curr_y

                # LEFT CLICK (PINCH)
                elif gesture == "pinch":
                    if time.time() - last_left_click > click_delay:
                        pyautogui.click()
                        last_left_click = time.time()

                # SCROLL (PEACE)
                elif gesture == "peace":
                    finger_y = hand_landmarks.landmark[12].y
                    scroll_speed = int((0.5 - finger_y) * 200)

                    if abs(scroll_speed) > 5:
                        pyautogui.scroll(scroll_speed)

                # FIST → SCROLL DOWN
                elif gesture == "fist":
                    if time.time() - last_scroll_time > scroll_delay:
                        pyautogui.scroll(-40)  # negative = scroll down
                        last_scroll_time = time.time()

    cv2.imshow("Virtual Mouse", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()