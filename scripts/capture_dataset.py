import cv2
import mediapipe as mp
import os
import pandas as pd

# ---------------- Paths ---------------- #
dataset_path = os.path.join(os.path.dirname(__file__), "..", "dataset")
os.makedirs(dataset_path, exist_ok=True)

# ---------------- Mediapipe Hands ---------------- #
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)
print("Press 'ESC' or 'q' to quit anytime.")
print("You can record multiple gestures one by one.")

while True:
    gesture_name = input("Enter gesture name (or 'done' to finish): ")
    if gesture_name.lower() == "done":
        break

    frames_data = []
    print(f"Recording gesture: {gesture_name}")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        # ---------------- Only save if hand is detected ---------------- #
        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)  # draw dots & connections
                landmarks = []
                for lm in handLms.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                frames_data.append(landmarks)

            print(f"\rFrames recorded: {len(frames_data)}", end="")

        cv2.imshow("Recording Gesture (Press 'q' or ESC to quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    if frames_data:
        df = pd.DataFrame(frames_data)
        df.to_csv(os.path.join(dataset_path, f"{gesture_name}.csv"), index=False)
        print(f"\n✅ Saved {gesture_name} with {len(frames_data)} frames.")

cap.release()
cv2.destroyAllWindows()
print("✅ Capture stopped safely")
