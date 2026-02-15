#!/usr/bin/env python3
"""
Genesis PRO - Multi-Person + Emotion Recognition
Sistema completo con tracking multi-persona ed emozioni

FEATURES:
✓ Multi-Person Tracking (fino a 5 persone)
✓ Emotion Recognition (7 emozioni)
✓ CORPO: 33 landmarks per persona
✓ VOLTO: 478 landmarks per persona
✓ MANI: 21x2 landmarks per persona
✓ LIP READING: Lettura labiale (persona principale)
✓ OGGETTI: 80 classi COCO
✓ Redis Integration per dashboard
"""
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from lip_reading import LipReader
from lip_reading_server import LipReadingDataServer
from emotion_recognition import EmotionRecognizer
from multi_person_tracker import MultiPersonTracker
import redis
import json
import time

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

# MediaPipe Pose connections
POSE_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (15, 17), (15, 19), (15, 21),
    (16, 18), (16, 20), (16, 22),
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

def draw_pose_landmarks(frame, pose_landmarks, color=(255, 255, 0)):
    """Disegna 33 landmark del corpo"""
    h, w, _ = frame.shape
    points = []

    for landmark in pose_landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        visibility = landmark.visibility
        points.append((x, y, visibility))

    # Disegna connessioni
    for connection in POSE_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx < len(points) and end_idx < len(points):
            if points[start_idx][2] > 0.5 and points[end_idx][2] > 0.5:
                cv2.line(frame, points[start_idx][:2], points[end_idx][:2], color, 2)

    # Disegna punti
    for i, (x, y, visibility) in enumerate(points):
        if visibility > 0.5:
            if i in [0, 11, 12, 13, 14, 15, 16, 23, 24]:
                cv2.circle(frame, (x, y), 6, (0, 255, 255), -1)
            else:
                cv2.circle(frame, (x, y), 4, color, -1)

def draw_hand_landmarks(frame, hand_landmarks, handedness="Unknown", color=(255, 0, 0)):
    """Disegna 21 landmark mano con MASSIMA PRECISIONE"""
    h, w, _ = frame.shape

    # Determina colore in base a mano sinistra/destra
    if handedness == "Left":
        hand_color = (255, 0, 0)  # BLU per sinistra
        label = "MANO SINISTRA"
    elif handedness == "Right":
        hand_color = (0, 0, 255)  # ROSSO per destra
        label = "MANO DESTRA"
    else:
        hand_color = color
        label = "MANO"

    # Converti landmarks in coordinate pixel
    points = []
    for landmark in hand_landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        points.append((x, y))

    # Disegna connessioni (LINEE SPESSE)
    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx < len(points) and end_idx < len(points):
            cv2.line(frame, points[start_idx], points[end_idx], hand_color, 3)

    # Disegna landmark points (GRANDI E VISIBILI)
    for i, (x, y) in enumerate(points):
        # Punti delle punte delle dita PIÙ GRANDI
        if i in [4, 8, 12, 16, 20]:  # Tips
            cv2.circle(frame, (x, y), 10, (0, 255, 255), -1)
            cv2.circle(frame, (x, y), 11, hand_color, 2)
        else:
            cv2.circle(frame, (x, y), 6, hand_color, -1)

    # Bounding box attorno alla mano (DETTAGLIATO)
    if points:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        # Espandi il box
        margin = 25
        x1, y1 = max(0, x1-margin), max(0, y1-margin)
        x2, y2 = min(w, x2+margin), min(h, y2+margin)

        cv2.rectangle(frame, (x1, y1), (x2, y2), hand_color, 4)
        cv2.putText(frame, label, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, hand_color, 2)

    return points

def draw_face_mesh_detailed(frame, face_landmarks, color=(0, 255, 0)):
    """Disegna face mesh DETTAGLIATO con labbra, occhi, sopracciglia evidenziati"""
    h, w, _ = frame.shape

    # Converti tutti i landmarks
    points = []
    for landmark in face_landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        points.append((x, y))

    # === LABBRA (ben visibili) ===
    # Labbra esterne
    lips_outer = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
    for i in range(len(lips_outer) - 1):
        if lips_outer[i] < len(points) and lips_outer[i+1] < len(points):
            cv2.line(frame, points[lips_outer[i]], points[lips_outer[i+1]], (0, 0, 255), 2)  # ROSSO per labbra
    # Chiudi il contorno
    if lips_outer[0] < len(points) and lips_outer[-1] < len(points):
        cv2.line(frame, points[lips_outer[-1]], points[lips_outer[0]], (0, 0, 255), 2)

    # Labbra interne
    lips_inner = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
    for i in range(len(lips_inner) - 1):
        if lips_inner[i] < len(points) and lips_inner[i+1] < len(points):
            cv2.line(frame, points[lips_inner[i]], points[lips_inner[i+1]], (0, 0, 255), 1)
    if lips_inner[0] < len(points) and lips_inner[-1] < len(points):
        cv2.line(frame, points[lips_inner[-1]], points[lips_inner[0]], (0, 0, 255), 1)

    # === OCCHI (ben visibili) ===
    # Occhio sinistro
    left_eye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    for i in range(len(left_eye) - 1):
        if left_eye[i] < len(points) and left_eye[i+1] < len(points):
            cv2.line(frame, points[left_eye[i]], points[left_eye[i+1]], (255, 255, 0), 2)  # CIANO per occhi
    if left_eye[0] < len(points) and left_eye[-1] < len(points):
        cv2.line(frame, points[left_eye[-1]], points[left_eye[0]], (255, 255, 0), 2)

    # Occhio destro
    right_eye = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    for i in range(len(right_eye) - 1):
        if right_eye[i] < len(points) and right_eye[i+1] < len(points):
            cv2.line(frame, points[right_eye[i]], points[right_eye[i+1]], (255, 255, 0), 2)
    if right_eye[0] < len(points) and right_eye[-1] < len(points):
        cv2.line(frame, points[right_eye[-1]], points[right_eye[0]], (255, 255, 0), 2)

    # === SOPRACCIGLIA ===
    # Sopracciglio sinistro
    left_eyebrow = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
    for i in range(len(left_eyebrow) - 1):
        if left_eyebrow[i] < len(points) and left_eyebrow[i+1] < len(points):
            cv2.line(frame, points[left_eyebrow[i]], points[left_eyebrow[i+1]], color, 2)

    # Sopracciglio destro
    right_eyebrow = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
    for i in range(len(right_eyebrow) - 1):
        if right_eyebrow[i] < len(points) and right_eyebrow[i+1] < len(points):
            cv2.line(frame, points[right_eyebrow[i]], points[right_eyebrow[i+1]], color, 2)

    # === CONTORNO VISO ===
    face_oval = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    for i in range(len(face_oval) - 1):
        if face_oval[i] < len(points) and face_oval[i+1] < len(points):
            cv2.line(frame, points[face_oval[i]], points[face_oval[i+1]], color, 1)

    # Disegna tutti i punti (più densi)
    for i, (x, y) in enumerate(points):
        if i % 2 == 0:  # Ogni 2 punti invece di 5
            cv2.circle(frame, (x, y), 1, color, -1)

    # Bounding box
    if points:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xs), max(ys)

        cv2.rectangle(frame, (x1-10, y1-10), (x2+10, y2+10), color, 3)
        cv2.putText(frame, "VOLTO (478 pts)", (x1-10, y1-25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def main():
    print("=" * 80)
    print("Genesis PRO - Multi-Person + Emotion Recognition")
    print("=" * 80)
    print("FEATURES COMPLETE:")
    print("  ✓ Multi-Person Tracking (fino a 5 persone)")
    print("  ✓ Emotion Recognition (7 emozioni per persona)")
    print("  ✓ CORPO: 33 landmarks MediaPipe Pose")
    print("  ✓ VOLTO: 478 landmarks MediaPipe Face")
    print("  ✓ MANI: 21x2 landmarks per persona")
    print("  ✓ LIP READING: Lettura labiale persona principale")
    print("  ✓ OGGETTI: 80 classi COCO")
    print("  ✓ Redis Integration")
    print()
    print("Controlli:")
    print("  - ESC: Esci")
    print("  - 'p': Toggle Pose")
    print("  - 'f': Toggle Face")
    print("  - 'h': Toggle Hands")
    print("  - 'e': Toggle Emotions")
    print("  - 'o': Toggle Objects")
    print("  - 'l': Toggle Lip Reading")
    print("  - 'i': Toggle Person IDs")
    print("=" * 80)

    import urllib.request
    import os

    # Download MediaPipe models
    print("\nCaricamento modelli MediaPipe...")

    hand_model_path = "hand_landmarker.task"
    face_model_path = "face_landmarker.task"
    pose_model_path = "pose_landmarker_heavy.task"

    for model_name, model_path, url in [
        ("Hand", hand_model_path, "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"),
        ("Face", face_model_path, "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"),
        ("Pose", pose_model_path, "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"),
    ]:
        if not os.path.exists(model_path):
            print(f"  Downloading {model_name} model...")
            urllib.request.urlretrieve(url, model_path)
            print(f"  ✓ {model_name} downloaded")
        else:
            print(f"  ✓ {model_name} exists")

    # Initialize MediaPipe detectors
    print("\nInizializzazione detectors...")

    # Hand (PRECISIONE MASSIMA)
    hand_detector = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=hand_model_path),
            num_hands=10,  # Fino a 10 mani (5 persone x 2)
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        ))

    # Face
    face_detector = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=face_model_path),
            num_faces=5,  # Fino a 5 persone
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ))

    # Pose
    pose_detector = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=pose_model_path),
            num_poses=5,  # Fino a 5 persone
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ))

    print("✓ MediaPipe ready")

    # Emotion Recognition
    print("\nInizializzazione Emotion Recognition...")
    emotion_recognizer = EmotionRecognizer(confidence_threshold=0.5)
    print("✓ Emotion Recognizer ready")

    # Multi-Person Tracker
    print("\nInizializzazione Multi-Person Tracker...")
    person_tracker = MultiPersonTracker(max_people=5, iou_threshold=0.3)
    print("✓ Multi-Person Tracker ready")

    # Lip Reading (solo persona principale)
    print("\nInizializzazione Lip Reading...")
    lip_reader = LipReader(sequence_length=30, confidence_threshold=0.55)
    print("✓ Lip Reader ready")

    # Redis
    redis_client = None
    lip_data_server = None
    try:
        redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        redis_client.ping()
        lip_data_server = LipReadingDataServer(redis_client)
        print("✓ Redis connected")
    except Exception as e:
        print(f"⚠️  Redis not available: {e}")

    # YOLO
    print("\nCaricamento YOLO...")
    object_model = YOLO("yolov8n.pt")
    print("✓ YOLO ready")

    # Webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("❌ Errore webcam")
        return

    print("\n✓ Genesis PRO attivo! Finestra: 'Genesis PRO'\n")

    # Toggle options
    show_pose = True
    show_face = True
    show_hands = True
    show_emotions = True
    show_objects = True
    show_lip_reading = True
    show_person_ids = True

    frame_count = 0
    fps_start = time.time()
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        display_frame = frame.copy()
        h, w, _ = frame.shape

        # FPS calculation
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - fps_start)
            fps_start = time.time()

        # MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect all people
        pose_result = pose_detector.detect(mp_image)
        face_result = face_detector.detect(mp_image)
        hand_result = hand_detector.detect(mp_image)

        # Process emotions for each face
        emotion_results = []
        if show_emotions and face_result.face_landmarks:
            for face_landmarks in face_result.face_landmarks:
                emotion = emotion_recognizer.recognize_emotion(face_landmarks)
                emotion_results.append(emotion)

        # Update multi-person tracker
        people = person_tracker.update(
            pose_results=pose_result.pose_landmarks if pose_result.pose_landmarks else [],
            face_results=face_result.face_landmarks if face_result.face_landmarks else [],
            hand_results=hand_result.hand_landmarks if hand_result.hand_landmarks else [],
            emotion_results=emotion_results,
            frame_shape=frame.shape
        )

        # Draw each person with their color
        for person_id, person in people.items():
            color = person.color

            # Pose
            if show_pose and person.pose_landmarks:
                draw_pose_landmarks(display_frame, person.pose_landmarks, color=color)

            # Face
            if show_face and person.face_landmarks:
                draw_face_mesh_detailed(display_frame, person.face_landmarks, color=color)

            # Hands - disegnate direttamente da hand_result per preservare handedness
            # (verrà fatto dopo, fuori dal loop delle persone)

            # Emotion
            if show_emotions and person.emotion_history:
                latest_emotion = person.emotion_history[-1]
                # Disegna emotion vicino alla faccia
                x1, y1, x2, y2 = person.bbox
                emotion_pos = (x1, y2 + 30)

                emoji = emotion_recognizer.EMOTION_EMOJI.get(latest_emotion['emotion'], '')
                emotion_color = emotion_recognizer.EMOTION_COLORS.get(latest_emotion['emotion'], (255, 255, 255))

                text = f"{emoji} {latest_emotion['emotion']} ({latest_emotion['confidence']:.0%})"
                cv2.putText(display_frame, text, emotion_pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, emotion_color, 2)

        # Draw person bounding boxes with IDs
        if show_person_ids:
            person_tracker.draw_people(display_frame)

        # === HAND TRACKING PRECISO (direttamente da MediaPipe con handedness) ===
        if show_hands and hand_result.hand_landmarks:
            for idx, hand_landmarks in enumerate(hand_result.hand_landmarks):
                # Get handedness (Left/Right) da MediaPipe
                handedness = hand_result.handedness[idx][0].category_name if hand_result.handedness else "Unknown"

                # Disegna con MASSIMA PRECISIONE
                draw_hand_landmarks(display_frame, hand_landmarks, handedness=handedness)

        # Lip Reading (solo persona più vicina/grande)
        if show_lip_reading and people:
            # Trova persona con bbox più grande (più vicina)
            largest_person = max(people.values(),
                               key=lambda p: (p.bbox[2]-p.bbox[0]) * (p.bbox[3]-p.bbox[1]))

            if largest_person.face_landmarks:
                lip_result = lip_reader.process_frame(largest_person.face_landmarks)

                if lip_data_server and lip_result:
                    lip_data_server.update_data(lip_result)

                # Draw lip reading info in angolo alto destra
                display_frame = lip_reader.draw_lip_reading_info(
                    display_frame, lip_result, position=(w-380, 30))

        # Object Detection (MIGLIORATO - sempre attivo, non solo ogni 3 frame)
        if show_objects:
            try:
                obj_results = object_model.track(frame, persist=True, verbose=False, conf=0.4, iou=0.5)

                for result in obj_results:
                    if result.boxes is not None and len(result.boxes) > 0:
                        boxes = result.boxes
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                            cls = int(box.cls[0])
                            conf = float(box.conf[0])

                            if cls == 0:  # Skip person
                                continue

                            obj_name = COCO_CLASSES[cls] if cls < len(COCO_CLASSES) else f"Class {cls}"

                            # Tracking ID
                            track_id = int(box.id[0]) if box.id is not None else None
                            if track_id is not None:
                                obj_name = f"#{track_id} {obj_name}"

                            # Colore ARANCIONE brillante
                            obj_color = (0, 165, 255)
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), obj_color, 3)

                            # Label con background
                            label = f"{obj_name} {conf:.2f}"
                            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            cv2.rectangle(display_frame, (x1, y1-label_h-10), (x1+label_w+10, y1), obj_color, -1)
                            cv2.putText(display_frame, label, (x1+5, y1-5),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            except Exception as e:
                pass  # Silent fail per object detection

        # Stats overlay (alto sinistra)
        stats_y = 30
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, stats_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        stats_y += 25
        cv2.putText(display_frame, f"People: {len(people)}/5", (10, stats_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        stats_y += 25

        toggles = f"P:{show_pose} F:{show_face} H:{show_hands} E:{show_emotions} O:{show_objects} L:{show_lip_reading}"
        cv2.putText(display_frame, toggles, (10, stats_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Send multi-person data to Redis
        if redis_client:
            try:
                multi_person_data = {
                    'timestamp': time.time(),
                    'num_people': len(people),
                    'people': []
                }

                for person_id, person in people.items():
                    person_data = {
                        'id': int(person_id),
                        'emotion': person.get_dominant_emotion(),
                        'frames_tracked': int(person.frames_tracked)
                    }
                    multi_person_data['people'].append(person_data)

                redis_client.set('genesis:multi_person:current',
                               json.dumps(multi_person_data), ex=60)
            except Exception as e:
                pass  # Silent fail

        cv2.imshow("Genesis PRO - Multi-Person + Emotions", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('p'):
            show_pose = not show_pose
            print(f"Pose: {'ON' if show_pose else 'OFF'}")
        elif key == ord('f'):
            show_face = not show_face
            print(f"Face: {'ON' if show_face else 'OFF'}")
        elif key == ord('h'):
            show_hands = not show_hands
            print(f"Hands: {'ON' if show_hands else 'OFF'}")
        elif key == ord('e'):
            show_emotions = not show_emotions
            print(f"Emotions: {'ON' if show_emotions else 'OFF'}")
        elif key == ord('o'):
            show_objects = not show_objects
            print(f"Objects: {'ON' if show_objects else 'OFF'}")
        elif key == ord('l'):
            show_lip_reading = not show_lip_reading
            print(f"Lip Reading: {'ON' if show_lip_reading else 'OFF'}")
        elif key == ord('i'):
            show_person_ids = not show_person_ids
            print(f"Person IDs: {'ON' if show_person_ids else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Genesis PRO terminato")

if __name__ == "__main__":
    main()
