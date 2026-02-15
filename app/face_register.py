#!/usr/bin/env python3
"""
Genesis - Face Registration Tool
Registra nuovi volti nel database per il riconoscimento facciale
"""
from __future__ import annotations
import argparse
import cv2
import os
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Registra un nuovo volto nel database Genesis")
    ap.add_argument("--person_id", required=True, help="ID persona (es: john_doe, person_001)")
    ap.add_argument("--cam", type=int, default=0, help="ID webcam")
    ap.add_argument("--faces_dir", default="data/faces", help="Directory database volti")
    args = ap.parse_args()

    person_dir = Path(args.faces_dir) / args.person_id
    person_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.cam)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    print(f"\n{'='*60}")
    print(f"Genesis - Registrazione volto per: {args.person_id}")
    print(f"{'='*60}\n")
    print("Istruzioni:")
    print("  - Posizionati davanti alla camera")
    print("  - Premi 'c' per catturare il volto")
    print("  - Premi 'q' o ESC per uscire\n")

    captured = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(display, "Premi 'c' per catturare", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Genesis - Face Register (ESC to quit)", display)
        k = cv2.waitKey(1) & 0xFF

        if k == ord('c') and len(faces) > 0:
            # Cattura il primo volto rilevato
            x, y, w, h = faces[0]
            face_img = frame[y:y+h, x:x+w]

            # Salva foto
            photo_path = person_dir / "photo.jpg"
            cv2.imwrite(str(photo_path), face_img)
            print(f"\n✓ Volto salvato: {photo_path}")
            captured = True
            break

        if k == 27 or k == ord('q'):  # ESC o 'q'
            break

    cap.release()
    cv2.destroyAllWindows()

    if captured:
        print(f"\n{'='*60}")
        print(f"✓ Registrazione completata per: {args.person_id}")
        print(f"{'='*60}")
        print("\nL'embedding verrà generato automaticamente al primo avvio del sistema.")
    else:
        print("\n✗ Nessun volto catturato")

if __name__ == "__main__":
    main()
