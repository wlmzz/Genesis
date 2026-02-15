#!/usr/bin/env python3
"""
Genesis - Complete Multi-Angle Tracking + Lip Reading
Volto multi-angolo (478 pts) + Mani (21 pts x2) + Oggetti + Lettura Labiale
"""
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from lip_reading import LipReader
from lip_reading_server import LipReadingDataServer
import redis

# COCO class names
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

# Hand landmark connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17)  # Palm
]

def draw_hand_landmarks(frame, hand_landmarks, handedness):
    """Disegna 21 landmark della mano con connessioni"""
    h, w, _ = frame.shape

    # Determina colore in base a mano sinistra/destra
    if handedness == "Left":
        color = (255, 0, 0)  # BLU per sinistra
        label = "MANO SINISTRA"
    else:
        color = (0, 0, 255)  # ROSSO per destra
        label = "MANO DESTRA"

    # Converti landmarks in coordinate pixel
    points = []
    for landmark in hand_landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        points.append((x, y))

    # Disegna connessioni
    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx < len(points) and end_idx < len(points):
            cv2.line(frame, points[start_idx], points[end_idx], color, 2)

    # Disegna landmark points
    for i, (x, y) in enumerate(points):
        # Punti delle punte delle dita più grandi
        if i in [4, 8, 12, 16, 20]:  # Tips
            cv2.circle(frame, (x, y), 8, (0, 255, 255), -1)
            cv2.circle(frame, (x, y), 9, color, 2)
        else:
            cv2.circle(frame, (x, y), 5, color, -1)

    # Bounding box attorno alla mano
    if points:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        # Espandi il box
        margin = 20
        x1, y1 = max(0, x1-margin), max(0, y1-margin)
        x2, y2 = min(w, x2+margin), min(h, y2+margin)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, label, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return points

def draw_face_mesh(frame, face_landmarks):
    """Disegna face mesh con 478 landmark - supporta profilo e angolazioni"""
    h, w, _ = frame.shape

    # Converti landmarks in coordinate pixel
    points = []
    for landmark in face_landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        points.append((x, y))

    # Disegna solo i punti principali per performance (ogni 3 punti)
    for i, (x, y) in enumerate(points):
        if i % 3 == 0:  # Riduci densità per performance
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    # Bounding box attorno al volto
    if points:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xs), max(ys)

        # Box VERDE
        cv2.rectangle(frame, (x1-10, y1-10), (x2+10, y2+10), (0, 255, 0), 3)
        cv2.putText(frame, "VOLTO (478 pts)", (x1-10, y1-25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

def check_object_in_hand(hand_bbox, obj_bbox, threshold=0.3):
    """Verifica se un oggetto è vicino/dentro una mano"""
    hx1, hy1, hx2, hy2 = hand_bbox
    ox1, oy1, ox2, oy2 = obj_bbox

    # Calcola centro oggetto
    obj_center_x = (ox1 + ox2) / 2
    obj_center_y = (oy1 + oy2) / 2

    # Espandi hand bbox per tolleranza
    margin = 50
    hx1_exp = hx1 - margin
    hy1_exp = hy1 - margin
    hx2_exp = hx2 + margin
    hy2_exp = hy2 + margin

    # Check se centro oggetto è vicino alla mano
    if hx1_exp <= obj_center_x <= hx2_exp and hy1_exp <= obj_center_y <= hy2_exp:
        return True

    return False

def main():
    print("=" * 70)
    print("Genesis - Complete Multi-Angle Tracking + Lip Reading")
    print("=" * 70)
    print("Tracking:")
    print("  - VOLTO: 478 landmarks MULTI-ANGOLO (profilo, frontale, laterale)")
    print("  - MANI: 21 landmarks per mano - TUTTE LE DITA")
    print("  - OGGETTI: 80 classi COCO + rilevamento in mano")
    print("  - LIP READING: Lettura labiale e riconoscimento parole")
    print()
    print("Controlli:")
    print("  - ESC: Esci")
    print("  - 'f': Toggle Face")
    print("  - 'h': Toggle Hands")
    print("  - 'o': Toggle Objects")
    print("  - 'l': Toggle Lip Reading")
    print("=" * 70)

    import urllib.request
    import os

    # Download MediaPipe models
    print("\nDownload MediaPipe models...")

    hand_model_path = "hand_landmarker.task"
    if not os.path.exists(hand_model_path):
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        print(f"Downloading hand model...")
        urllib.request.urlretrieve(url, hand_model_path)
        print(f"✓ Hand model downloaded")
    else:
        print(f"✓ Hand model exists")

    face_model_path = "face_landmarker.task"
    if not os.path.exists(face_model_path):
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        print(f"Downloading face model...")
        urllib.request.urlretrieve(url, face_model_path)
        print(f"✓ Face model downloaded")
    else:
        print(f"✓ Face model exists")

    # MediaPipe Hand Landmarker
    hand_base_options = python.BaseOptions(model_asset_path=hand_model_path)
    hand_options = vision.HandLandmarkerOptions(
        base_options=hand_base_options,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    hand_detector = vision.HandLandmarker.create_from_options(hand_options)

    # MediaPipe Face Landmarker (478 punti, multi-angolo)
    face_base_options = python.BaseOptions(model_asset_path=face_model_path)
    face_options = vision.FaceLandmarkerOptions(
        base_options=face_base_options,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    face_detector = vision.FaceLandmarker.create_from_options(face_options)

    # Lip Reading System
    print("\nInizializzazione Lip Reading...")
    lip_reader = LipReader(sequence_length=30, confidence_threshold=0.55)
    print("✓ Lip Reader ready")

    # Redis connection for dashboard integration
    redis_client = None
    lip_data_server = None
    try:
        redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        redis_client.ping()
        lip_data_server = LipReadingDataServer(redis_client)
        print("✓ Redis connected - Dashboard integration enabled")
    except Exception as e:
        print(f"⚠️  Redis not available - Dashboard integration disabled: {e}")

    # YOLO object detection
    print("\nCaricamento YOLO model...")
    object_model = YOLO("yolov8n.pt")
    print("✓ YOLO loaded")

    # Webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("❌ Errore webcam")
        return

    print("\n✓ Sistema attivo! Cerca finestra 'Genesis - Hand Tracking'\n")

    show_face = True
    show_hands = True
    show_objects = True
    show_lip_reading = True
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        display_frame = frame.copy()
        h, w, _ = frame.shape

        hand_bboxes = []

        # === HAND LANDMARKS (21 per mano) ===
        if show_hands:
            # Converti frame per MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Detect hands
            hand_result = hand_detector.detect(mp_image)

            if hand_result.hand_landmarks:
                for idx, hand_landmarks in enumerate(hand_result.hand_landmarks):
                    # Get handedness (Left/Right)
                    handedness = hand_result.handedness[idx][0].category_name

                    # Disegna landmarks
                    points = draw_hand_landmarks(display_frame, hand_landmarks, handedness)

                    # Salva bbox per check oggetti in mano
                    if points:
                        xs = [p[0] for p in points]
                        ys = [p[1] for p in points]
                        hand_bboxes.append((min(xs), min(ys), max(xs), max(ys)))

        # === FACE MESH (478 punti multi-angolo) ===
        lip_result = None
        if show_face or show_lip_reading:
            # Usa lo stesso rgb_frame preparato per le mani
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Detect face
            face_result = face_detector.detect(mp_image)

            if face_result.face_landmarks:
                for face_landmarks in face_result.face_landmarks:
                    # Disegna face mesh (478 punti)
                    if show_face:
                        draw_face_mesh(display_frame, face_landmarks)

                    # LIP READING - Analizza movimento labiale
                    if show_lip_reading:
                        lip_result = lip_reader.process_frame(face_landmarks)

                        # Send data to dashboard via Redis
                        if lip_data_server and lip_result:
                            lip_data_server.update_data(lip_result)

        # === LIP READING VISUALIZATION ===
        if show_lip_reading and lip_result:
            # Position box on right side to avoid overlap
            h, w, _ = display_frame.shape
            display_frame = lip_reader.draw_lip_reading_info(display_frame, lip_result, position=(w-380, 30))

        # === OBJECT TRACKING (persistente, senza lampeggiamento) ===
        if show_objects:
            # Usa track() invece di predict() per tracking persistente
            obj_results = object_model.track(frame, persist=True, verbose=False, conf=0.5)

            for result in obj_results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    # Tracking ID (se disponibile)
                    track_id = int(box.id[0]) if box.id is not None else None

                    # Skip persone
                    if cls == 0:
                        continue

                    # Nome oggetto
                    if cls < len(COCO_CLASSES):
                        obj_name = COCO_CLASSES[cls]
                    else:
                        obj_name = f"Class {cls}"

                    # Aggiungi ID se disponibile per tracking persistente
                    if track_id is not None:
                        obj_name = f"#{track_id} {obj_name}"

                    # Check se oggetto è in mano
                    in_hand = False
                    for hand_bbox in hand_bboxes:
                        if check_object_in_hand(hand_bbox, (x1, y1, x2, y2)):
                            in_hand = True
                            break

                    # Colore: VERDE se in mano, ARANCIONE altrimenti
                    if in_hand:
                        color = (0, 255, 0)  # VERDE
                        label = f"IN MANO: {obj_name} {conf:.2f}"
                    else:
                        color = (0, 165, 255)  # ARANCIONE
                        label = f"{obj_name} {conf:.2f}"

                    # Disegna box più spesso se in mano
                    thickness = 4 if in_hand else 2
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)

                    # Label
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(display_frame, (x1, y1-label_h-10), (x1+label_w+10, y1), color, -1)
                    cv2.putText(display_frame, label, (x1+5, y1-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Status info
        status_y = 30
        cv2.putText(display_frame, f"Face: {'ON' if show_face else 'OFF'} (478 pts multi-angolo)", (10, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        status_y += 30
        cv2.putText(display_frame, f"Hands: {'ON' if show_hands else 'OFF'} (21 pts x mano)", (10, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        status_y += 30
        cv2.putText(display_frame, f"Objects: {'ON' if show_objects else 'OFF'}", (10, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Genesis - Hand Tracking", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('f'):
            show_face = not show_face
            print(f"Face: {'ON' if show_face else 'OFF'}")
        elif key == ord('h'):
            show_hands = not show_hands
            print(f"Hands: {'ON' if show_hands else 'OFF'}")
        elif key == ord('o'):
            show_objects = not show_objects
            print(f"Objects: {'ON' if show_objects else 'OFF'}")
        elif key == ord('l'):
            show_lip_reading = not show_lip_reading
            print(f"Lip Reading: {'ON' if show_lip_reading else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Tracking terminato")

if __name__ == "__main__":
    main()
