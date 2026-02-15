#!/usr/bin/env python3
"""
Genesis - Video File Analysis
Analisi da file video con facial recognition
"""
from __future__ import annotations
import argparse
import time
import os
import cv2
import yaml
from ultralytics import YOLO

from core.zones import load_zones, draw_zones
from core.analytics import GenesisAnalytics
from core.face_recognition import FaceRecognizer
from core.identity_tracker import IdentityTracker
from core.io_utils import ensure_dir, append_csv, append_jsonl

def resize_keep(frame, width: int):
    """Ridimensiona frame mantenendo aspect ratio"""
    h, w = frame.shape[:2]
    if w == width:
        return frame
    scale = width / w
    return cv2.resize(frame, (width, int(h * scale)))

def main():
    ap = argparse.ArgumentParser(description="Genesis - Video File Analysis")
    ap.add_argument("--video", required=True, help="Path al file video")
    ap.add_argument("--settings", default="configs/settings.yaml")
    ap.add_argument("--zones", default="configs/zones.json")
    ap.add_argument("--outdir", default="data/outputs")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.settings, "r", encoding="utf-8"))
    zones = load_zones(args.zones)

    ensure_dir(args.outdir)
    csv_path = os.path.join(args.outdir, "metrics.csv")
    jsonl_path = os.path.join(args.outdir, "metrics.jsonl")

    model = YOLO(cfg["detector"]["model"])
    cap = cv2.VideoCapture(args.video)

    analytics = GenesisAnalytics(zone_names=list(zones.keys()), queue_zone="queue_area")

    # FACIAL RECOGNITION SETUP
    face_recognizer = None
    identity_tracker = None
    if cfg.get("face_recognition", {}).get("enabled", False):
        face_recognizer = FaceRecognizer(
            faces_dir=cfg["identity"]["faces_dir"],
            model=cfg["face_recognition"]["model"],
            distance_metric=cfg["face_recognition"]["distance_metric"],
            threshold=cfg["face_recognition"]["recognition_threshold"]
        )
        identity_tracker = IdentityTracker(db_path=cfg["identity"]["database_path"])
        print(f"✓ Face recognition enabled: {len(face_recognizer.known_embeddings)} known faces loaded")

    export_every = float(cfg["metrics"]["export_interval_sec"])
    last_export = 0.0
    frame_count = 0

    print("\n" + "="*60)
    print("Genesis - Video Analysis Started")
    print("="*60)
    print(f"Video: {args.video}")
    print(f"Zones: {list(zones.keys())}")
    print(f"Face Recognition: {'Enabled' if face_recognizer else 'Disabled'}")
    print("\nPress ESC to quit\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = resize_keep(frame, cfg["video"]["resize_width"])

        res = model.track(
            source=frame,
            persist=bool(cfg["tracking"]["persist"]),
            conf=float(cfg["detector"]["conf"]),
            iou=float(cfg["detector"]["iou"]),
            verbose=False,
            classes=[0],
        )[0]

        track_centers = {}
        identities = {}

        if res.boxes is not None and res.boxes.id is not None:
            ids = res.boxes.id.cpu().numpy().astype(int)
            xyxy = res.boxes.xyxy.cpu().numpy()

            for tid, b in zip(ids, xyxy):
                x1, y1, x2, y2 = b
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                track_centers[tid] = (cx, cy)

                # FACIAL RECOGNITION
                if face_recognizer is not None and frame_count % cfg["face_recognition"].get("update_embeddings_every", 30) == 0:
                    face_img = frame[int(y1):int(y2), int(x1):int(x2)]

                    if face_img.shape[0] >= cfg["face_recognition"]["face_size_threshold"] and face_img.shape[1] >= cfg["face_recognition"]["face_size_threshold"]:
                        person_id, confidence = face_recognizer.recognize_face(face_img)

                        current_zone = None
                        for zname, poly in zones.items():
                            from core.zones import point_in_polygon
                            if point_in_polygon((cx, cy), poly):
                                current_zone = zname
                                break

                        if person_id:
                            identities[tid] = person_id
                            identity_tracker.update_identity(tid, person_id, 1.0 - confidence, current_zone)
                        else:
                            identities[tid] = f"unknown_{tid}"
                            identity_tracker.update_identity(tid, None, 1.0 - confidence, current_zone)
                else:
                    if identity_tracker:
                        state = identity_tracker.get_identity(tid)
                        if state:
                            identities[tid] = state.person_id
                        else:
                            identities[tid] = f"track_{tid}"
                    else:
                        identities[tid] = f"track_{tid}"

        now = time.time()
        frame_count += 1

        if identity_tracker:
            identity_tracker.cleanup_stale()

        analytics.update(track_centers, zones, now=now)
        snap = analytics.snapshot(track_centers, zones, now=now)

        if (now - last_export) >= export_every:
            row = {
                "ts": int(snap.ts),
                "people_total": snap.people_total,
                "queue_len": snap.queue_len,
                "avg_queue_wait_sec": round(snap.avg_queue_wait_sec, 2),
                **{f"zone_{k}": v for k, v in snap.people_by_zone.items()},
            }
            header = list(row.keys())
            append_csv(csv_path, row, header)
            append_jsonl(jsonl_path, row)
            last_export = now

        if cfg["video"]["draw"]:
            draw_zones(frame, zones)

            if res.boxes is not None and res.boxes.id is not None:
                ids = res.boxes.id.cpu().numpy().astype(int)
                xyxy = res.boxes.xyxy.cpu().numpy()
                for tid, b in zip(ids, xyxy):
                    x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                    person_id = identities.get(tid, f"track_{tid}")
                    color = (0, 255, 0) if not person_id.startswith("unknown") and not person_id.startswith("track") else (0, 165, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, person_id, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.putText(frame, f"People: {snap.people_total}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Queue: {snap.queue_len}  Avg wait: {int(snap.avg_queue_wait_sec)}s", (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if face_recognizer:
                known_count = sum(1 for p in identities.values() if not p.startswith("unknown") and not p.startswith("track"))
                cv2.putText(frame, f"Identified: {known_count}/{len(identities)}", (15, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Genesis - Video Analysis (ESC to quit)", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Video analysis completed")
    print(f"✓ Metrics saved to: {args.outdir}")

if __name__ == "__main__":
    main()
