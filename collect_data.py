import cv2
import mediapipe as mp
import csv
import os
import time

gesture_name = input("Enter gesture name (pinch, peace, fist, palm): ").strip()

if gesture_name == "":
    print("Gesture name cannot be empty!")
    exit()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

os.makedirs("dataset", exist_ok=True)
file_path = "dataset/Hand Gesture Landmark Coordinates Dataset.csv"

sample_count = 0
last_save_time = 0
save_interval = 0.2   # save every 0.2 seconds (5 samples per second)

print("Press 'q' to stop collecting")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    current_time = time.time()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Save only every 0.2 seconds
            if current_time - last_save_time > save_interval:

                landmarks = []
                wrist = hand_landmarks.landmark[0]

                for lm in hand_landmarks.landmark:
                    landmarks.append(lm.x - wrist.x)
                    landmarks.append(lm.y - wrist.y)
                    landmarks.append(lm.z - wrist.z)

                landmarks.append(gesture_name)

                with open(file_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(landmarks)

                sample_count += 1
                last_save_time = current_time
                print(f"Saved sample: {sample_count}")

    cv2.putText(frame, f"Samples: {sample_count}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.imshow("Collecting Data", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Total samples saved:", sample_count)