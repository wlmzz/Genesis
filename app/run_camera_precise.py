#!/usr/bin/env python3
"""
Genesis - Ultra-Precise Head & Hands Tracking
Tracking ultra-preciso con MediaPipe Hands (21 landmarks per mano)
"""
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def draw_head_box(frame, keypoints):
    """Disegna box rosso sulla testa usando keypoints viso"""
    head_kpts = keypoints[0:5]  # nose, eyes, ears
    valid_kpts = [(int(kp[0]), int(kp[1])) for kp in head_kpts if kp[2] > 0.5]

    if len(valid_kpts) >= 2:
        xs = [kp[0] for kp in valid_kpts]
        ys = [kp[1] for kp in valid_kpts]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        # Espandi box del 60% per includere tutta la testa
        w, h = x2 - x1, y2 - y1
        margin = int(max(w, h) * 0.6)
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(frame.shape[1], x2 + margin)
        y2 = min(frame.shape[0], y2 + margin)

        # Box ROSSO spesso per la testa
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
        cv2.putText(frame, "TESTA", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

def draw_hand_landmarks(frame, hand_landmarks, handedness):
    """Disegna landmarks MediaPipe e bounding box per la mano"""
    # Ottieni coordinate di tutti i 21 landmarks
    h, w, _ = frame.shape
    landmarks = []
    for landmark in hand_landmarks.landmark:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        landmarks.append((x, y))

    # Calcola bounding box che contiene tutti i landmarks
    xs = [lm[0] for lm in landmarks]
    ys = [lm[1] for lm in landmarks]
    x1, y1 = max(0, min(xs) - 20), max(0, min(ys) - 20)
    x2, y2 = min(w, max(xs) + 20), min(h, max(ys) + 20)

    # Colore in base a sinistra/destra
    hand_label = handedness.classification[0].label
    if hand_label == "Left":
        color = (0, 255, 0)  # VERDE per mano sinistra
        label = "MANO SX"
    else:
        color = (255, 0, 0)  # BLU per mano destra
        label = "MANO DX"

    # Disegna bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
    cv2.putText(frame, label, (x1, y1-10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Disegna tutti i 21 landmarks e le connessioni
    mp_drawing.draw_landmarks(
        frame,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style()
    )

def main():
    print("=" * 70)
    print("Genesis - Ultra-Precise Head & Hands Tracking")
    print("=" * 70)
    print("🔴 Box ROSSO      = Testa (YOLOv8-pose)")
    print("🟢 Box VERDE + 21 punti = Mano Sinistra (MediaPipe)")
    print("🔵 Box BLU + 21 punti   = Mano Destra (MediaPipe)")
    print()
    print("Premi ESC per uscire")
    print("=" * 70)

    # Carica modello YOLO-pose per la testa
    yolo_model = YOLO("yolov8n-pose.pt")

    # MediaPipe Hands per tracking preciso mani
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Apri webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Errore: impossibile aprire la webcam")
        return

    # Aumenta risoluzione per maggiore precisione
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Converti BGR a RGB per MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 1. YOLO pose per testa
        yolo_results = yolo_model(frame, verbose=False)
        for result in yolo_results:
            if result.keypoints is not None and len(result.keypoints) > 0:
                for person_kpts in result.keypoints.data:
                    keypoints = person_kpts.cpu().numpy()
                    draw_head_box(frame, keypoints)

        # 2. MediaPipe per mani (ULTRA PRECISO - 21 landmarks per mano)
        hands_results = hands.process(frame_rgb)
        if hands_results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(
                hands_results.multi_hand_landmarks,
                hands_results.multi_handedness
            ):
                draw_hand_landmarks(frame, hand_landmarks, handedness)

        # Info in tempo reale
        cv2.putText(frame, "YOLOv8-pose + MediaPipe Hands", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Mostra frame
        cv2.imshow("Genesis - Ultra-Precise Tracking", frame)

        # ESC per uscire
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("\n✓ Tracking terminato")

if __name__ == "__main__":
    main()
