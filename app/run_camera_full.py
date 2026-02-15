#!/usr/bin/env python3
"""
Genesis - Full Body Tracking
Tracking completo: Volto (468 punti), Mani (21 punti x2), Dita, Gesti
"""
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.FaceMeshModule import FaceMeshDetector

# Gesture recognition
def detect_gesture(hand_landmarks):
    """Rileva gesti comuni dalla posizione delle dita"""
    if not hand_landmarks:
        return "None"

    lm_list = hand_landmarks["lmList"]

    # Conta dita alzate
    fingers_up = 0

    # Pollice (ID 4)
    if lm_list[4][0] > lm_list[3][0]:  # pollice verso destra
        fingers_up += 1

    # Altre dita (ID: 8, 12, 16, 20)
    for id in [8, 12, 16, 20]:
        if lm_list[id][1] < lm_list[id-2][1]:  # dito alzato
            fingers_up += 1

    # Classifica gesto
    if fingers_up == 0:
        return "PUGNO"
    elif fingers_up == 1:
        return "UNO"
    elif fingers_up == 2:
        return "DUE / PEACE"
    elif fingers_up == 3:
        return "TRE"
    elif fingers_up == 4:
        return "QUATTRO"
    elif fingers_up == 5:
        return "MANO APERTA"
    else:
        return "Unknown"

def main():
    print("=" * 70)
    print("Genesis - Full Body Tracking")
    print("=" * 70)
    print("Tracking Completo:")
    print("  - Volto: Face Mesh (468 landmarks)")
    print("  - Mani: 21 landmarks per mano (tutte le dita)")
    print("  - Gesti: Riconoscimento automatico")
    print()
    print("Controlli:")
    print("  - ESC: Esci")
    print("  - 'f': Toggle Face Mesh")
    print("  - 'h': Toggle Hand Landmarks")
    print("=" * 70)

    # Inizializza detectors
    hand_detector = HandDetector(
        detectionCon=0.8,
        maxHands=2,
        minTrackCon=0.5
    )

    face_mesh_detector = FaceMeshDetector(
        maxFaces=1
    )

    face_detector = FaceDetector(
        minDetectionCon=0.7
    )

    # Apri webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("❌ Errore: impossibile aprire la webcam")
        return

    # Opzioni visualizzazione
    show_face_mesh = True
    show_hand_landmarks = True

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # === FACE MESH (468 landmarks) ===
        if show_face_mesh:
            frame, faces = face_mesh_detector.findFaceMesh(frame, draw=True)
            if faces:
                # Disegna bounding box verde sul volto
                face = faces[0]
                x_coords = [lm[0] for lm in face]
                y_coords = [lm[1] for lm in face]
                x1, y1 = min(x_coords), min(y_coords)
                x2, y2 = max(x_coords), max(y_coords)

                # Box VERDE per il volto
                cv2.rectangle(frame, (x1-20, y1-20), (x2+20, y2+20), (0, 255, 0), 3)
                cv2.putText(frame, "VOLTO (468 punti)", (x1-20, y1-35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # === HAND TRACKING (21 landmarks per mano) ===
        hands, frame = hand_detector.findHands(frame, draw=show_hand_landmarks)

        if hands:
            for hand in hands:
                # Info mano
                hand_type = hand["type"]  # Left o Right
                lm_list = hand["lmList"]  # 21 landmarks
                bbox = hand["bbox"]  # Bounding box

                # Rileva gesto
                gesture = detect_gesture(hand)

                # Colore in base alla mano
                if hand_type == "Left":
                    color = (255, 0, 0)  # BLU per mano sinistra
                    label = f"MANO SX: {gesture}"
                else:
                    color = (0, 0, 255)  # ROSSO per mano destra
                    label = f"MANO DX: {gesture}"

                # Disegna bounding box
                x, y, w, h = bbox
                cv2.rectangle(frame, (x-20, y-20), (x+w+20, y+h+20), color, 4)

                # Mostra gesto riconosciuto
                cv2.putText(frame, label, (x-20, y-35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # Evidenzia le punte delle dita (ID: 4,8,12,16,20)
                finger_tips = [4, 8, 12, 16, 20]
                for tip_id in finger_tips:
                    cx, cy = lm_list[tip_id][0], lm_list[tip_id][1]
                    cv2.circle(frame, (cx, cy), 10, (0, 255, 255), cv2.FILLED)

        # Info in tempo reale
        info_text = f"FaceMesh: {'ON' if show_face_mesh else 'OFF'} | "
        info_text += f"Hands: {'ON' if show_hand_landmarks else 'OFF'}"
        cv2.putText(frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # FPS
        fps = cap.get(cv2.CAP_PROP_FPS)
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Mostra frame
        cv2.imshow("Genesis - Full Body Tracking", frame)

        # Controlli
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('f'):  # Toggle face mesh
            show_face_mesh = not show_face_mesh
            print(f"Face Mesh: {'ON' if show_face_mesh else 'OFF'}")
        elif key == ord('h'):  # Toggle hand landmarks
            show_hand_landmarks = not show_hand_landmarks
            print(f"Hand Landmarks: {'ON' if show_hand_landmarks else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Tracking terminato")

if __name__ == "__main__":
    main()
