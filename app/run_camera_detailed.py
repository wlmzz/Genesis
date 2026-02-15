#!/usr/bin/env python3
"""
Genesis - Detailed Face & Body Tracking
Tracking dettagliato: Volto (68 landmarks con Dlib) + Corpo/Mani (17 keypoints YOLO-pose)
"""
import cv2
import numpy as np
import dlib
from ultralytics import YOLO

# Nomi keypoints YOLO-pose (COCO format)
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

def draw_face_landmarks(frame, landmarks):
    """Disegna 68 landmark del volto con Dlib"""
    # Jaw line (0-16)
    for i in range(0, 16):
        cv2.line(frame, (landmarks[i][0], landmarks[i][1]),
                (landmarks[i+1][0], landmarks[i+1][1]), (0, 255, 0), 2)

    # Eyebrows (17-26)
    for i in range(17, 21):
        cv2.line(frame, (landmarks[i][0], landmarks[i][1]),
                (landmarks[i+1][0], landmarks[i+1][1]), (0, 255, 255), 2)
    for i in range(22, 26):
        cv2.line(frame, (landmarks[i][0], landmarks[i][1]),
                (landmarks[i+1][0], landmarks[i+1][1]), (0, 255, 255), 2)

    # Nose (27-35)
    for i in range(27, 30):
        cv2.line(frame, (landmarks[i][0], landmarks[i][1]),
                (landmarks[i+1][0], landmarks[i+1][1]), (255, 255, 0), 2)
    for i in range(30, 35):
        cv2.line(frame, (landmarks[i][0], landmarks[i][1]),
                (landmarks[i+1][0], landmarks[i+1][1]), (255, 255, 0), 2)

    # Eyes (36-47)
    for i in range(36, 41):
        cv2.line(frame, (landmarks[i][0], landmarks[i][1]),
                (landmarks[i+1][0], landmarks[i+1][1]), (255, 0, 255), 2)
    cv2.line(frame, (landmarks[41][0], landmarks[41][1]),
            (landmarks[36][0], landmarks[36][1]), (255, 0, 255), 2)

    for i in range(42, 47):
        cv2.line(frame, (landmarks[i][0], landmarks[i][1]),
                (landmarks[i+1][0], landmarks[i+1][1]), (255, 0, 255), 2)
    cv2.line(frame, (landmarks[47][0], landmarks[47][1]),
            (landmarks[42][0], landmarks[42][1]), (255, 0, 255), 2)

    # Mouth (48-67)
    for i in range(48, 59):
        cv2.line(frame, (landmarks[i][0], landmarks[i][1]),
                (landmarks[i+1][0], landmarks[i+1][1]), (0, 0, 255), 2)
    cv2.line(frame, (landmarks[59][0], landmarks[59][1]),
            (landmarks[48][0], landmarks[48][1]), (0, 0, 255), 2)

    for i in range(60, 67):
        cv2.line(frame, (landmarks[i][0], landmarks[i][1]),
                (landmarks[i+1][0], landmarks[i+1][1]), (0, 0, 255), 2)
    cv2.line(frame, (landmarks[67][0], landmarks[67][1]),
            (landmarks[60][0], landmarks[60][1]), (0, 0, 255), 2)

    # Disegna punti
    for i, (x, y) in enumerate(landmarks):
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

def draw_hand_detailed(frame, keypoints):
    """Disegna landmarks delle mani con dettagli"""
    # Wrist points
    left_wrist = keypoints[9]   # ID 9 = left_wrist
    right_wrist = keypoints[10]  # ID 10 = right_wrist

    hand_size = 100

    # Mano sinistra
    if left_wrist[2] > 0.5:
        x, y = int(left_wrist[0]), int(left_wrist[1])
        # Box BLU
        cv2.rectangle(frame, (x-hand_size, y-hand_size),
                     (x+hand_size, y+hand_size), (255, 0, 0), 3)
        # Punto polso
        cv2.circle(frame, (x, y), 8, (255, 255, 0), -1)
        cv2.putText(frame, "MANO SX", (x-hand_size, y-hand_size-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Mano destra
    if right_wrist[2] > 0.5:
        x, y = int(right_wrist[0]), int(right_wrist[1])
        # Box ROSSO
        cv2.rectangle(frame, (x-hand_size, y-hand_size),
                     (x+hand_size, y+hand_size), (0, 0, 255), 3)
        # Punto polso
        cv2.circle(frame, (x, y), 8, (255, 255, 0), -1)
        cv2.putText(frame, "MANO DX", (x-hand_size, y-hand_size-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

def main():
    print("=" * 70)
    print("Genesis - Detailed Face & Body Tracking")
    print("=" * 70)
    print("Tracking:")
    print("  - VOLTO: 68 landmarks (Dlib)")
    print("  - CORPO: 17 keypoints (YOLOv8-pose)")
    print("  - MANI: Wrist tracking dettagliato")
    print()
    print("Controlli:")
    print("  - ESC: Esci")
    print("  - 'f': Toggle Face Landmarks")
    print("  - 'b': Toggle Body Keypoints")
    print("=" * 70)

    # Dlib face detector e shape predictor
    face_detector = dlib.get_frontal_face_detector()

    # Download shape predictor se non esiste
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    try:
        face_predictor = dlib.shape_predictor(predictor_path)
    except:
        print("⚠️  Downloading face landmarks model...")
        import urllib.request
        url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        urllib.request.urlretrieve(url, predictor_path + ".bz2")
        import bz2
        with bz2.open(predictor_path + ".bz2") as f:
            with open(predictor_path, 'wb') as out:
                out.write(f.read())
        face_predictor = dlib.shape_predictor(predictor_path)
        print("✓ Model downloaded")

    # YOLO pose model
    yolo_model = YOLO("yolov8s-pose.pt")

    # Webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("❌ Errore: impossibile aprire la webcam")
        return

    show_face = True
    show_body = True

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # === FACE LANDMARKS (68 punti con Dlib) ===
        if show_face:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector(gray)

            for face in faces:
                landmarks = face_predictor(gray, face)
                landmarks_points = [(landmarks.part(i).x, landmarks.part(i).y)
                                   for i in range(68)]

                # Disegna face mesh
                draw_face_landmarks(frame, landmarks_points)

                # Box verde sul volto
                x1, y1 = face.left(), face.top()
                x2, y2 = face.right(), face.bottom()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame, "VOLTO (68 landmarks)", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # === BODY KEYPOINTS (17 punti con YOLO) ===
        if show_body:
            results = yolo_model(frame, verbose=False)
            for result in results:
                if result.keypoints is not None:
                    for person_kpts in result.keypoints.data:
                        keypoints = person_kpts.cpu().numpy()

                        # Disegna mani dettagliate
                        draw_hand_detailed(frame, keypoints)

                        # Disegna tutti i keypoints del corpo
                        for i, kpt in enumerate(keypoints):
                            if kpt[2] > 0.5:  # confidence
                                x, y = int(kpt[0]), int(kpt[1])
                                cv2.circle(frame, (x, y), 5, (255, 255, 0), -1)

        # Info
        info = f"Face: {'ON' if show_face else 'OFF'} | Body: {'ON' if show_body else 'OFF'}"
        cv2.putText(frame, info, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Mostra frame
        cv2.imshow("Genesis - Detailed Tracking", frame)

        # Controlli
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('f'):
            show_face = not show_face
            print(f"Face Landmarks: {'ON' if show_face else 'OFF'}")
        elif key == ord('b'):
            show_body = not show_body
            print(f"Body Keypoints: {'ON' if show_body else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Tracking terminato")

if __name__ == "__main__":
    main()
