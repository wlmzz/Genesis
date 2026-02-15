#!/usr/bin/env python3
"""
Genesis - Complete Tracking System
Tracking completo: Volto (68 landmarks) + Corpo (17 keypoints) + Oggetti (80 classi COCO)
"""
import cv2
import numpy as np
import dlib
from ultralytics import YOLO

# COCO class names (80 oggetti)
COCO_CLASSES = [
    'persona', 'bicicletta', 'auto', 'moto', 'aereo', 'autobus', 'treno', 'camion', 'barca',
    'semaforo', 'idrante', 'segnale stop', 'parchimetro', 'panchina', 'uccello', 'gatto', 'cane',
    'cavallo', 'pecora', 'mucca', 'elefante', 'orso', 'zebra', 'giraffa', 'zaino', 'ombrello',
    'borsa', 'cravatta', 'valigia', 'frisbee', 'sci', 'snowboard', 'palla', 'aquilone',
    'mazza baseball', 'guanto baseball', 'skateboard', 'tavola surf', 'racchetta tennis',
    'bottiglia', 'bicchiere vino', 'tazza', 'forchetta', 'coltello', 'cucchiaio', 'ciotola',
    'banana', 'mela', 'sandwich', 'arancia', 'broccoli', 'carota', 'hot dog', 'pizza', 'donut',
    'torta', 'sedia', 'divano', 'pianta', 'letto', 'tavolo', 'wc', 'tv', 'laptop', 'mouse',
    'telecomando', 'tastiera', 'cellulare', 'microonde', 'forno', 'tostapane', 'lavandino',
    'frigorifero', 'libro', 'orologio', 'vaso', 'forbici', 'orsacchiotto', 'asciugacapelli', 'spazzolino'
]

def draw_face_landmarks(frame, landmarks):
    """Disegna 68 landmark del volto"""
    # Jaw line
    for i in range(0, 16):
        cv2.line(frame, (landmarks[i][0], landmarks[i][1]),
                (landmarks[i+1][0], landmarks[i+1][1]), (0, 255, 0), 2)

    # Eyebrows
    for i in range(17, 21):
        cv2.line(frame, (landmarks[i][0], landmarks[i][1]),
                (landmarks[i+1][0], landmarks[i+1][1]), (0, 255, 255), 2)
    for i in range(22, 26):
        cv2.line(frame, (landmarks[i][0], landmarks[i][1]),
                (landmarks[i+1][0], landmarks[i+1][1]), (0, 255, 255), 2)

    # Nose
    for i in range(27, 30):
        cv2.line(frame, (landmarks[i][0], landmarks[i][1]),
                (landmarks[i+1][0], landmarks[i+1][1]), (255, 255, 0), 2)
    for i in range(30, 35):
        cv2.line(frame, (landmarks[i][0], landmarks[i][1]),
                (landmarks[i+1][0], landmarks[i+1][1]), (255, 255, 0), 2)

    # Eyes
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

    # Mouth
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
    for x, y in landmarks:
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

def draw_hand_boxes(frame, keypoints):
    """Disegna box sulle mani"""
    left_wrist = keypoints[9]   # left_wrist
    right_wrist = keypoints[10]  # right_wrist
    hand_size = 80

    # Mano sinistra (BLU)
    if left_wrist[2] > 0.5:
        x, y = int(left_wrist[0]), int(left_wrist[1])
        cv2.rectangle(frame, (x-hand_size, y-hand_size),
                     (x+hand_size, y+hand_size), (255, 100, 0), 3)
        cv2.circle(frame, (x, y), 8, (255, 255, 0), -1)
        cv2.putText(frame, "MANO SX", (x-hand_size, y-hand_size-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

    # Mano destra (ROSSO)
    if right_wrist[2] > 0.5:
        x, y = int(right_wrist[0]), int(right_wrist[1])
        cv2.rectangle(frame, (x-hand_size, y-hand_size),
                     (x+hand_size, y+hand_size), (0, 100, 255), 3)
        cv2.circle(frame, (x, y), 8, (255, 255, 0), -1)
        cv2.putText(frame, "MANO DX", (x-hand_size, y-hand_size-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)

def main():
    print("=" * 70)
    print("Genesis - Complete Tracking System")
    print("=" * 70)
    print("Tracking:")
    print("  - VOLTO: 68 landmarks faciali (Dlib)")
    print("  - CORPO: 17 keypoints (YOLOv8-pose)")
    print("  - MANI: Tracking dettagliato polsi")
    print("  - OGGETTI: 80 classi COCO (YOLOv8)")
    print()
    print("Controlli:")
    print("  - ESC: Esci")
    print("  - 'f': Toggle Face Landmarks")
    print("  - 'b': Toggle Body Keypoints")
    print("  - 'o': Toggle Object Detection")
    print("=" * 70)

    # Dlib face detector e shape predictor
    face_detector = dlib.get_frontal_face_detector()
    predictor_path = "shape_predictor_68_face_landmarks.dat"

    try:
        face_predictor = dlib.shape_predictor(predictor_path)
    except:
        print("⚠️  Shape predictor model not found!")
        print("Please ensure shape_predictor_68_face_landmarks.dat exists in the current directory")
        return

    # YOLO models
    print("\nCaricamento modelli...")
    pose_model = YOLO("yolov8s-pose.pt")  # Per corpo/mani
    object_model = YOLO("yolov8n.pt")     # Per oggetti
    print("✓ Modelli caricati")

    # Webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("❌ Errore: impossibile aprire la webcam")
        return

    print("\n✓ Webcam aperta - Tracking attivo!")
    print("Cerca la finestra 'Genesis - Complete Tracking'\n")

    # Opzioni visualizzazione
    show_face = True
    show_body = True
    show_objects = True

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

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
                cv2.putText(frame, "VOLTO (68 pts)", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # === BODY KEYPOINTS (17 punti con YOLO-pose) ===
        if show_body:
            pose_results = pose_model(frame, verbose=False)
            for result in pose_results:
                if result.keypoints is not None:
                    for person_kpts in result.keypoints.data:
                        keypoints = person_kpts.cpu().numpy()

                        # Disegna mani
                        draw_hand_boxes(frame, keypoints)

                        # Disegna keypoints corpo
                        for i, kpt in enumerate(keypoints):
                            if kpt[2] > 0.5:  # confidence
                                x, y = int(kpt[0]), int(kpt[1])
                                cv2.circle(frame, (x, y), 4, (255, 255, 0), -1)

        # === OBJECT DETECTION (ogni 3 frame per performance) ===
        if show_objects and frame_count % 3 == 0:
            obj_results = object_model(frame, verbose=False, conf=0.5)

            for result in obj_results:
                boxes = result.boxes
                for box in boxes:
                    # Coordinate
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Classe e confidenza
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    # Skip persone (già tracciamo con pose)
                    if cls == 0:  # classe "person"
                        continue

                    # Nome oggetto
                    if cls < len(COCO_CLASSES):
                        obj_name = COCO_CLASSES[cls]
                    else:
                        obj_name = f"Class {cls}"

                    # Colore ARANCIONE per oggetti
                    color = (0, 165, 255)

                    # Disegna box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Label
                    label = f"{obj_name} {conf:.2f}"
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1-label_h-10), (x1+label_w+10, y1), color, -1)
                    cv2.putText(frame, label, (x1+5, y1-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Info status
        status_y = 30
        cv2.putText(frame, f"Face: {'ON' if show_face else 'OFF'}", (10, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        status_y += 30
        cv2.putText(frame, f"Body: {'ON' if show_body else 'OFF'}", (10, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        status_y += 30
        cv2.putText(frame, f"Objects: {'ON' if show_objects else 'OFF'}", (10, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Mostra frame
        cv2.imshow("Genesis - Complete Tracking", frame)

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
        elif key == ord('o'):
            show_objects = not show_objects
            print(f"Object Detection: {'ON' if show_objects else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Tracking terminato")

if __name__ == "__main__":
    main()
