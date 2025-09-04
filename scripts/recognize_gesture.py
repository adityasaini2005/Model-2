import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import pandas as pd

# ---------------- Paths ---------------- #
models_path = os.path.join(os.path.dirname(__file__), "..", "models")

# ---------------- Load Models & Scaler ---------------- #
scaler = joblib.load(os.path.join(models_path, "scaler.pkl"))
label_encoder = joblib.load(os.path.join(models_path, "label_encoder.pkl"))
rf_model = joblib.load(os.path.join(models_path, "rf_model.pkl"))

# ---------------- Mediapipe Hands ---------------- #
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

def extract_hand_features(result):
    features = []
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks[:2]:
            for lm in handLms.landmark:
                features.extend([lm.x, lm.y, lm.z])
    # Pad or truncate to match scaler size
    expected_size = len(scaler.feature_names_in_)
    if len(features) < expected_size:
        features += [0.0] * (expected_size - len(features))
    elif len(features) > expected_size:
        features = features[:expected_size]
    return features

def predict_gesture(features):
    df = pd.DataFrame([features], columns=scaler.feature_names_in_)
    vector_scaled = scaler.transform(df)
    pred_idx = rf_model.predict(vector_scaled)[0]
    return label_encoder.inverse_transform([pred_idx])[0]

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    gesture_text = "No hand detected"
    if result.multi_hand_landmarks:
        features = extract_hand_features(result)
        gesture_text = predict_gesture(features)

        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, gesture_text, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    cv2.imshow("Gesture Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break  # Only closes window once

cap.release()
cv2.destroyAllWindows()
