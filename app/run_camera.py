#!/usr/bin/env python3
"""
Genesis - Live Camera Analysis
Analisi in tempo reale da webcam con facial recognition

Supports dual-mode operation:
- Sync mode (default): Traditional frame-by-frame processing
- Event-driven mode: Publishes events to Redis Streams for microservices
"""
from __future__ import annotations
import argparse
import time
import os
import cv2
import yaml
import logging
from ultralytics import YOLO

from core.zones import load_zones, draw_zones
from core.analytics import GenesisAnalytics
from core.face_recognition import FaceRecognizer
from core.identity_tracker import IdentityTracker
from core.io_utils import ensure_dir, append_csv, append_jsonl

# Event-driven imports (optional based on feature flag)
try:
    from infrastructure.events import (
        RedisEventProducer,
        FrameCapturedEvent,
        PersonDetectedEvent,
        FaceRecognizedEvent,
        ZoneEvent,
        BackpressureError,
    )
    from infrastructure.cache import FrameCache
    EVENT_DRIVEN_AVAILABLE = True
except ImportError:
    EVENT_DRIVEN_AVAILABLE = False
    logging.warning("Event-driven components not available - install redis and lz4 packages")

logger = logging.getLogger(__name__)

def resize_keep(frame, width: int):
    """Ridimensiona frame mantenendo aspect ratio"""
    h, w = frame.shape[:2]
    if w == width:
        return frame
    scale = width / w
    return cv2.resize(frame, (width, int(h * scale)))

def main():
    ap = argparse.ArgumentParser(description="Genesis - Live Camera Analysis")
    ap.add_argument("--cam", type=int, default=0, help="ID webcam")
    ap.add_argument("--settings", default="configs/settings.yaml", help="File configurazione")
    ap.add_argument("--zones", default="configs/zones.json", help="File zone")
    ap.add_argument("--outdir", default="data/outputs", help="Directory output")
    args = ap.parse_args()

    # Load config
    cfg = yaml.safe_load(open(args.settings, "r", encoding="utf-8"))
    zones = load_zones(args.zones)

    # Setup output
    ensure_dir(args.outdir)
    csv_path = os.path.join(args.outdir, "metrics.csv")
    jsonl_path = os.path.join(args.outdir, "metrics.jsonl")

    # YOLO model
    model = YOLO(cfg["detector"]["model"])
    cap = cv2.VideoCapture(args.cam)

    # Analytics
    analytics = GenesisAnalytics(zone_names=list(zones.keys()), queue_zone="queue_area")

    # EVENT-DRIVEN SETUP (if enabled)
    event_producer = None
    frame_cache = None
    event_driven_enabled = cfg.get("event_driven", {}).get("enabled", False)

    if event_driven_enabled and EVENT_DRIVEN_AVAILABLE:
        try:
            event_config = cfg["event_driven"]
            event_producer = RedisEventProducer(
                redis_url=event_config.get("redis_url", "redis://localhost:6379"),
                stream_prefix=event_config.get("stream_prefix", "genesis"),
                backpressure_enabled=event_config.get("backpressure", {}).get("enabled", True),
                max_pending=event_config.get("backpressure", {}).get("max_pending", 1000),
                block_threshold=event_config.get("backpressure", {}).get("block_threshold", 800),
            )

            # Frame cache for sharing frames
            if event_config.get("frame_cache", {}).get("enabled", True):
                frame_cache = FrameCache(
                    redis_url=event_config.get("redis_url", "redis://localhost:6379"),
                    ttl_seconds=event_config.get("frame_cache", {}).get("ttl_seconds", 60),
                    compress=True,
                )

            print(f"✓ Event-driven mode enabled (prefix: {event_config.get('stream_prefix', 'genesis')})")
        except Exception as e:
            print(f"⚠ Failed to initialize event-driven mode: {e}")
            print("  Falling back to sync mode")
            event_producer = None
            frame_cache = None
    elif event_driven_enabled and not EVENT_DRIVEN_AVAILABLE:
        print("⚠ Event-driven mode requested but dependencies not available")
        print("  Install with: pip install redis lz4")

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
    queue_over_since = None
    frame_count = 0

    print("\n" + "="*60)
    print("Genesis - Live Analysis Started")
    print("="*60)
    print(f"Mode: {'Event-Driven' if event_producer else 'Sync'}")
    print(f"Camera: {args.cam}")
    print(f"Zones: {list(zones.keys())}")
    print(f"Face Recognition: {'Enabled' if face_recognizer else 'Disabled'}")
    if event_producer:
        print(f"Frame Cache: {'Enabled' if frame_cache else 'Disabled'}")
    print("\nPress ESC to quit\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = resize_keep(frame, cfg["video"]["resize_width"])
        now = time.time()

        # Generate unique frame ID for event correlation
        frame_id = f"cam_{args.cam}_{frame_count}_{int(now * 1000)}"

        # PUBLISH: FrameCapturedEvent (if event-driven mode)
        if event_producer and frame_cache:
            try:
                # Cache frame for workers
                frame_cache.set(frame_id, frame)

                # Publish event
                event_producer.publish(FrameCapturedEvent(
                    camera_id=f"cam_{args.cam}",
                    frame_id=frame_id,
                    timestamp=now,
                    frame_shape=frame.shape,
                ))
            except BackpressureError as e:
                logger.warning(f"Backpressure detected: {e}")
                # Continue processing locally even if event publishing fails
            except Exception as e:
                logger.error(f"Failed to publish frame event: {e}")

        # YOLO tracking
        res = model.track(
            source=frame,
            persist=bool(cfg["tracking"]["persist"]),
            conf=float(cfg["detector"]["conf"]),
            iou=float(cfg["detector"]["iou"]),
            verbose=False,
            classes=[0],  # person only
        )[0]

        track_centers = {}
        identities = {}  # track_id -> person_id

        if res.boxes is not None and res.boxes.id is not None:
            ids = res.boxes.id.cpu().numpy().astype(int)
            xyxy = res.boxes.xyxy.cpu().numpy()

            for tid, b in zip(ids, xyxy):
                x1, y1, x2, y2 = b
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                track_centers[tid] = (cx, cy)

                # PUBLISH: PersonDetectedEvent (if event-driven mode)
                if event_producer:
                    try:
                        event_producer.publish(PersonDetectedEvent(
                            frame_id=frame_id,
                            track_id=int(tid),
                            bbox=(float(x1), float(y1), float(x2), float(y2)),
                            confidence=float(res.boxes.conf[list(ids).index(tid)]),
                            timestamp=now,
                        ))
                    except Exception as e:
                        logger.error(f"Failed to publish person event: {e}")

                # FACIAL RECOGNITION per ogni track
                if face_recognizer is not None and frame_count % cfg["face_recognition"].get("update_embeddings_every", 30) == 0:
                    # Estrai ROI volto
                    face_img = frame[int(y1):int(y2), int(x1):int(x2)]

                    # Verifica dimensione minima
                    if face_img.shape[0] >= cfg["face_recognition"]["face_size_threshold"] and face_img.shape[1] >= cfg["face_recognition"]["face_size_threshold"]:
                        person_id, confidence = face_recognizer.recognize_face(face_img)

                        # Determina zona corrente
                        current_zone = None
                        for zname, poly in zones.items():
                            from core.zones import point_in_polygon
                            if point_in_polygon((cx, cy), poly):
                                current_zone = zname
                                break

                        if person_id:
                            identities[tid] = person_id
                            identity_tracker.update_identity(tid, person_id, 1.0 - confidence, current_zone)

                            # PUBLISH: FaceRecognizedEvent (if event-driven mode)
                            if event_producer:
                                try:
                                    # Get embedding for event (Note: in production, extract from recognizer)
                                    embedding = [0.0] * 512  # Placeholder - in real implementation, get from face_recognizer
                                    event_producer.publish(FaceRecognizedEvent(
                                        track_id=int(tid),
                                        person_id=person_id,
                                        embedding=embedding,
                                        confidence=confidence,
                                        is_new_face=False,
                                        timestamp=now,
                                        current_zone=current_zone,
                                    ))
                                except Exception as e:
                                    logger.error(f"Failed to publish face event: {e}")
                        else:
                            # Volto sconosciuto
                            identities[tid] = f"unknown_{tid}"
                            identity_tracker.update_identity(tid, None, 1.0 - confidence, current_zone)

                            # PUBLISH: New face event (if event-driven mode)
                            if event_producer:
                                try:
                                    embedding = [0.0] * 512  # Placeholder
                                    event_producer.publish(FaceRecognizedEvent(
                                        track_id=int(tid),
                                        person_id=None,
                                        embedding=embedding,
                                        confidence=1.0 - confidence,
                                        is_new_face=True,
                                        timestamp=now,
                                        current_zone=current_zone,
                                    ))
                                except Exception as e:
                                    logger.error(f"Failed to publish new face event: {e}")

                            # Opzionale: salva volti sconosciuti
                            if cfg["face_recognition"].get("save_unknown_faces", False):
                                unknown_dir = ensure_dir("data/faces/unknown")
                                cv2.imwrite(f"{unknown_dir}/unknown_{tid}_{int(time.time())}.jpg", face_img)
                else:
                    # Recupera identità già assegnata
                    if identity_tracker:
                        state = identity_tracker.get_identity(tid)
                        if state:
                            identities[tid] = state.person_id
                        else:
                            identities[tid] = f"track_{tid}"
                    else:
                        identities[tid] = f"track_{tid}"

        frame_count += 1

        # Cleanup identità stale
        if identity_tracker:
            identity_tracker.cleanup_stale()

        # Update analytics
        analytics.update(track_centers, zones, now=now)
        snap = analytics.snapshot(track_centers, zones, now=now)

        # Alerts
        queue_len_thr = int(cfg["alerts"]["queue_len_threshold"])
        queue_dur_thr = int(cfg["alerts"]["queue_len_duration_sec"])
        avg_wait_thr = int(cfg["alerts"]["avg_wait_threshold_sec"])

        alert_msgs = []
        if snap.queue_len > queue_len_thr:
            queue_over_since = queue_over_since or now
            if (now - queue_over_since) > queue_dur_thr:
                alert_msgs.append(f"ALERT: coda alta ({snap.queue_len}) > {queue_len_thr} per > {queue_dur_thr}s")
        else:
            queue_over_since = None

        if snap.avg_queue_wait_sec > avg_wait_thr:
            alert_msgs.append(f"ALERT: attesa media coda {int(snap.avg_queue_wait_sec)}s > {avg_wait_thr}s")

        # Export metrics
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

        # Draw visualization
        if cfg["video"]["draw"]:
            draw_zones(frame, zones)

            # Draw bounding boxes con identità
            if res.boxes is not None and res.boxes.id is not None:
                ids = res.boxes.id.cpu().numpy().astype(int)
                xyxy = res.boxes.xyxy.cpu().numpy()
                for tid, b in zip(ids, xyxy):
                    x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])

                    # Colore in base a identità
                    person_id = identities.get(tid, f"track_{tid}")
                    color = (0, 255, 0) if not person_id.startswith("unknown") and not person_id.startswith("track") else (0, 165, 255)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, person_id, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Info generali
            cv2.putText(frame, f"People: {snap.people_total}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Queue: {snap.queue_len}  Avg wait: {int(snap.avg_queue_wait_sec)}s", (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if face_recognizer:
                known_count = sum(1 for p in identities.values() if not p.startswith("unknown") and not p.startswith("track"))
                cv2.putText(frame, f"Identified: {known_count}/{len(identities)}", (15, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            y = 120
            for m in alert_msgs[:3]:
                cv2.putText(frame, m, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                y += 25

            cv2.imshow("Genesis - Live Analysis (ESC to quit)", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # ESC
                break

    cap.release()
    cv2.destroyAllWindows()

    # Cleanup event-driven resources
    if event_producer:
        event_producer.close()
    if frame_cache:
        frame_cache.close()

    print("\n✓ Analysis stopped")

if __name__ == "__main__":
    main()
