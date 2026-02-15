#!/usr/bin/env python3
"""
OWL-ViT Zero-Shot Object Detector
Rileva QUALSIASI oggetto usando descrizioni testuali
"""
import torch
import cv2
import numpy as np
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image

class OWLViTDetector:
    """
    Zero-shot object detection con OWL-ViT

    Esempio:
        detector = OWLViTDetector(
            text_queries=["cellulare", "bottiglia d'acqua", "penna", "chiavi auto"]
        )
        detections = detector.detect(frame)
    """

    def __init__(self,
                 text_queries=None,
                 model_name="google/owlvit-base-patch32",
                 confidence_threshold=0.05,
                 device=None):
        """
        Args:
            text_queries: Lista di oggetti da cercare (es. ["cellulare", "penna"])
            model_name: Modello OWL-ViT ("owlvit-base-patch32" o "owlvit-large-patch14")
            confidence_threshold: Soglia minima di confidenza (0.0-1.0)
            device: "cuda", "mps", o "cpu" (auto-detect se None)
        """
        # Device detection
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"OWL-ViT: Loading model '{model_name}' on {self.device}...")

        # Load model and processor
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Configuration
        self.text_queries = text_queries or [
            "phone", "bottle", "cup", "book", "pen", "keys",
            "wallet", "watch", "glasses", "bag"
        ]
        self.confidence_threshold = confidence_threshold

        print(f"✓ OWL-ViT ready (searching for: {', '.join(self.text_queries[:5])}...)")

    def set_queries(self, text_queries):
        """Aggiorna gli oggetti da cercare"""
        self.text_queries = text_queries
        print(f"OWL-ViT: Updated queries → {', '.join(text_queries)}")

    def detect(self, frame):
        """
        Rileva oggetti nel frame

        Args:
            frame: Frame OpenCV (BGR numpy array)

        Returns:
            list: [{'label': str, 'confidence': float, 'box': (x1,y1,x2,y2)}, ...]
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # Prepare inputs
        inputs = self.processor(
            text=self.text_queries,
            images=pil_image,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process
        target_sizes = torch.tensor([pil_image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            threshold=self.confidence_threshold,
            target_sizes=target_sizes
        )[0]

        # Convert to detections list
        detections = []

        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"].cpu().numpy()

        for box, score, label_idx in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            label = self.text_queries[label_idx]

            detections.append({
                'label': label,
                'confidence': float(score),
                'box': (int(x1), int(y1), int(x2), int(y2))
            })

        return detections

    def draw_detections(self, frame, detections):
        """
        Disegna le detection sul frame

        Args:
            frame: Frame OpenCV (modificato in-place)
            detections: Lista di detection da detect()
        """
        for det in detections:
            x1, y1, x2, y2 = det['box']
            label = det['label']
            conf = det['confidence']

            # Colore in base a confidenza
            if conf > 0.3:
                color = (0, 255, 0)  # Verde (alta confidenza)
            elif conf > 0.15:
                color = (0, 255, 255)  # Giallo (media)
            else:
                color = (0, 165, 255)  # Arancione (bassa)

            # Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label con background
            label_text = f"{label} {conf:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(frame, (x1, y1-label_h-10), (x1+label_w+10, y1), color, -1)
            cv2.putText(
                frame, label_text, (x1+5, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
            )

        return frame


# Preset query lists per scenari comuni
QUERY_PRESETS = {
    "office": [
        "phone", "laptop", "mouse", "keyboard", "pen", "notebook",
        "coffee cup", "water bottle", "headphones", "charger"
    ],
    "kitchen": [
        "cup", "plate", "fork", "knife", "spoon", "bottle",
        "glass", "bowl", "mug", "cutting board"
    ],
    "personal": [
        "phone", "smartphone", "cellulare", "wallet", "portafoglio",
        "keys", "chiavi", "watch", "orologio", "glasses", "occhiali",
        "bag", "borsa", "headphones", "cuffie", "airpods",
        "airpods case", "custodia airpods", "earbuds case",
        "charger", "caricatore", "credit card", "carta", "ID card"
    ],
    "tools": [
        "hammer", "screwdriver", "wrench", "pliers", "drill",
        "saw", "tape measure", "level", "utility knife", "scissors"
    ],
    "medical": [
        "thermometer", "stethoscope", "syringe", "pills", "bandage",
        "mask", "gloves", "blood pressure monitor", "otoscope"
    ],
    "all": [
        # Ultra-comprehensive - oggetti comuni (inglese + italiano + varianti)
        "phone", "smartphone", "cellulare", "mobile phone", "cell phone", "iphone", "android phone",
        "bottle", "bottiglia", "water bottle", "plastic bottle", "glass bottle",
        "cup", "tazza", "mug", "coffee cup", "glass", "bicchiere", "drinking glass",
        "book", "libro", "notebook", "quaderno", "notepad",
        "pen", "penna", "pencil", "matita", "marker", "pennarello",
        "keys", "chiavi", "car keys", "house keys", "key ring",
        "wallet", "portafoglio", "purse", "credit card", "card",
        "watch", "orologio", "smartwatch", "wristwatch", "apple watch",
        "glasses", "occhiali", "sunglasses", "eyeglasses", "spectacles",
        "bag", "borsa", "backpack", "zaino", "handbag", "purse",
        "laptop", "computer", "macbook", "notebook computer",
        "mouse", "computer mouse", "wireless mouse",
        "keyboard", "tastiera", "computer keyboard",
        "headphones", "cuffie", "earphones", "earbuds", "airpods",
        "airpods case", "custodia airpods", "earbuds case", "charging case", "earbud charging case",
        "white case", "small white box", "apple airpods case",
        "charger", "caricatore", "phone charger", "cable", "charging cable", "usb cable",
        "remote control", "telecomando", "tv remote", "remote",
        "scissors", "forbici", "knife", "coltello", "fork", "forchetta",
        "spoon", "cucchiaio", "plate", "piatto", "bowl", "ciotola",
        "lighter", "accendino", "cigarette", "sigaretta",
        "tissue", "fazzoletto", "napkin", "tovagliolo",
        "coin", "moneta", "money", "soldi", "euro", "banknote"
    ]
}


if __name__ == "__main__":
    """Test OWL-ViT detector"""
    import sys

    # Test con webcam
    print("Testing OWL-ViT with webcam...")

    detector = OWLViTDetector(
        text_queries=QUERY_PRESETS["personal"],
        confidence_threshold=0.1
    )

    cap = cv2.VideoCapture(0)

    print("\nControls:")
    print("  - ESC: Exit")
    print("  - 'o': Office objects")
    print("  - 'k': Kitchen objects")
    print("  - 'p': Personal objects")
    print("  - 't': Tools")
    print("  - 'a': All objects")

    current_preset = "personal"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect (ogni 3 frame per performance)
        if cap.get(cv2.CAP_PROP_POS_FRAMES) % 3 == 0:
            detections = detector.detect(frame)
            detector.draw_detections(frame, detections)

        # Info
        cv2.putText(frame, f"Preset: {current_preset}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("OWL-ViT Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('o'):
            detector.set_queries(QUERY_PRESETS["office"])
            current_preset = "office"
        elif key == ord('k'):
            detector.set_queries(QUERY_PRESETS["kitchen"])
            current_preset = "kitchen"
        elif key == ord('p'):
            detector.set_queries(QUERY_PRESETS["personal"])
            current_preset = "personal"
        elif key == ord('t'):
            detector.set_queries(QUERY_PRESETS["tools"])
            current_preset = "tools"
        elif key == ord('a'):
            detector.set_queries(QUERY_PRESETS["all"])
            current_preset = "all"

    cap.release()
    cv2.destroyAllWindows()
