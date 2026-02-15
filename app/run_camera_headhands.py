#!/usr/bin/env python3
"""
Genesis - Head & Hands Tracking
Tracciamento specifico di testa e mani con bounding box colorati
"""
import cv2
import numpy as np
from ultralytics import YOLO

def draw_head_box(frame, keypoints):
    """Disegna box rosso sulla testa usando keypoints viso (0-4: nose, eyes, ears)"""
    # Keypoints: 0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear
    head_kpts = keypoints[0:5]  # Primi 5 keypoints per la testa

    # Filtra keypoints validi (confidence > 0.5)
    valid_kpts = [(int(kp[0]), int(kp[1])) for kp in head_kpts if kp[2] > 0.5]

    if len(valid_kpts) >= 2:
        # Calcola bounding box che contiene tutti i keypoints della testa
        xs = [kp[0] for kp in valid_kpts]
        ys = [kp[1] for kp in valid_kpts]

        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        # Espandi il box del 50% per includere tutta la testa
        w, h = x2 - x1, y2 - y1
        margin = int(max(w, h) * 0.5)

        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(frame.shape[1], x2 + margin)
        y2 = min(frame.shape[0], y2 + margin)

        # Disegna box ROSSO per la testa
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(frame, "HEAD", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def draw_hand_boxes(frame, keypoints):
    """Disegna box verdi sulle mani usando wrist keypoints (9-10: left_wrist, right_wrist)"""
    # Keypoints: 9=left_wrist, 10=right_wrist
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]

    hand_size = 80  # Dimensione box mani

    # Mano sinistra
    if left_wrist[2] > 0.5:  # confidence > 0.5
        x, y = int(left_wrist[0]), int(left_wrist[1])
        x1 = max(0, x - hand_size//2)
        y1 = max(0, y - hand_size//2)
        x2 = min(frame.shape[1], x + hand_size//2)
        y2 = min(frame.shape[0], y + hand_size//2)

        # Box VERDE per mano sinistra
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, "LEFT HAND", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Mano destra
    if right_wrist[2] > 0.5:
        x, y = int(right_wrist[0]), int(right_wrist[1])
        x1 = max(0, x - hand_size//2)
        y1 = max(0, y - hand_size//2)
        x2 = min(frame.shape[1], x + hand_size//2)
        y2 = min(frame.shape[0], y + hand_size//2)

        # Box BLU per mano destra
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.putText(frame, "RIGHT HAND", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

def main():
    print("=" * 60)
    print("Genesis - Head & Hands Tracking")
    print("=" * 60)
    print("Box Rosso = Testa")
    print("Box Verde = Mano Sinistra")
    print("Box Blu = Mano Destra")
    print()
    print("Premi ESC per uscire")
    print("=" * 60)

    # Carica modello YOLO-pose (small = più preciso)
    model = YOLO("yolov8s-pose.pt")

    # Apri webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Errore: impossibile aprire la webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO pose detection
        results = model(frame, verbose=False)

        for result in results:
            if result.keypoints is not None and len(result.keypoints) > 0:
                # Per ogni persona rilevata
                for person_kpts in result.keypoints.data:
                    keypoints = person_kpts.cpu().numpy()  # 17 keypoints x 3 (x, y, confidence)

                    # Disegna box sulla testa (ROSSO)
                    draw_head_box(frame, keypoints)

                    # Disegna box sulle mani (VERDE e BLU)
                    draw_hand_boxes(frame, keypoints)

        # Mostra frame
        cv2.imshow("Genesis - Head & Hands Tracking", frame)

        # ESC per uscire
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Tracking terminato")

if __name__ == "__main__":
    main()
