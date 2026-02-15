#!/usr/bin/env python3
"""
Genesis - Zone Editor
Editor visuale per definire zone di tracking
"""
from __future__ import annotations
import json
import argparse
import cv2

points = []
zones = {}

def click_cb(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])

def main():
    ap = argparse.ArgumentParser(description="Genesis - Editor visuale zone")
    ap.add_argument("--source", default="0", help="0 per webcam oppure path video mp4")
    ap.add_argument("--width", type=int, default=960, help="Larghezza frame")
    ap.add_argument("--out", default="configs/zones.json", help="File output JSON")
    args = ap.parse_args()

    src = 0 if args.source == "0" else args.source
    cap = cv2.VideoCapture(src)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("Impossibile leggere frame da sorgente")

    h, w = frame.shape[:2]
    if w != args.width:
        scale = args.width / w
        frame = cv2.resize(frame, (args.width, int(h * scale)))

    cv2.namedWindow("Genesis - Zone Editor")
    cv2.setMouseCallback("Genesis - Zone Editor", click_cb)

    print(f"\n{'='*60}")
    print("Genesis - Zone Editor")
    print(f"{'='*60}\n")
    print("Istruzioni:")
    print("  - Clic sinistro: aggiungi punto")
    print("  - t: salva zona corrente (ti chiede nome) e reset punti")
    print("  - r: reset punti")
    print("  - s: salva file zones.json e esci")
    print("  - q: esci senza salvare\n")

    while True:
        disp = frame.copy()

        # draw current polygon
        if len(points) >= 2:
            for i in range(1, len(points)):
                cv2.line(disp, tuple(points[i-1]), tuple(points[i]), (0, 255, 255), 2)
        for p in points:
            cv2.circle(disp, tuple(p), 5, (0, 255, 255), -1)

        # draw saved zones
        for name, poly in zones.items():
            for i in range(len(poly)):
                p1 = tuple(poly[i])
                p2 = tuple(poly[(i+1) % len(poly)])
                cv2.line(disp, p1, p2, (0, 255, 0), 2)
            cv2.putText(disp, name, (poly[0][0]+5, poly[0][1]-8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Genesis - Zone Editor", disp)
        k = cv2.waitKey(20) & 0xFF

        if k == ord("q"):
            break
        if k == ord("r"):
            points.clear()
        if k == ord("t"):
            if len(points) < 3:
                print("⚠ Servono almeno 3 punti per un poligono.")
                continue
            name = input("Nome zona (es. queue_area, cashier_area, entrance_area): ").strip()
            if not name:
                continue
            zones[name] = [[int(x), int(y)] for x, y in points]
            print(f"✓ Zona '{name}' salvata con {len(points)} punti")
            points.clear()
        if k == ord("s"):
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(zones, f, ensure_ascii=False, indent=2)
            print(f"\n✓ File salvato: {args.out}")
            print(f"  Zone definite: {list(zones.keys())}\n")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
