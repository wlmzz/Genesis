#!/usr/bin/env python3
"""
Diagnostic script to check lip reading system status
"""
import redis
import json

print("=" * 60)
print("LIP READING DIAGNOSTIC")
print("=" * 60)

# Check Redis connection
try:
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    r.ping()
    print("✓ Redis is running")
except Exception as e:
    print(f"❌ Redis error: {e}")
    exit(1)

# Check for lip reading data
current_data = r.get("genesis:lip_reading:current")
if current_data:
    print("✓ Current lip reading data found:")
    print(json.dumps(json.loads(current_data), indent=2))
else:
    print("⚠️  No current lip reading data")
    print("\nPossible reasons:")
    print("  1. No face detected - position yourself in front of camera")
    print("  2. Lip reading toggled off - press 'L' key in camera window")
    print("  3. No mouth movement - try speaking or opening your mouth")

# Check history
history_count = r.llen("genesis:lip_reading:history")
print(f"\nLip reading history: {history_count} entries")

if history_count > 0:
    print("\nRecent words:")
    history = r.lrange("genesis:lip_reading:history", 0, 4)
    for item in history:
        data = json.loads(item)
        print(f"  - {data['word']} (confidence: {data['confidence']:.1%})")

print("\n" + "=" * 60)
print("INSTRUCTIONS:")
print("=" * 60)
print("1. Make sure camera window 'Genesis - Hand Tracking' is open")
print("2. Position your face in front of the camera")
print("3. Press 'L' to toggle lip reading ON")
print("4. Move your mouth or speak")
print("5. Refresh the dashboard at http://localhost:8501")
print("=" * 60)
