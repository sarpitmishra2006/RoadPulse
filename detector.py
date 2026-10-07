"""
detector.py - Multi-pothole detector with dashboard stream
- Reads frames from dashboard's /video_feed
- Detects MULTIPLE potholes per frame (draws box for each)
- Fires one alert per frame (with 15s cooldown)
- Saves image with all boxes drawn
"""

import cv2
import numpy as np
import time
import smbus2
import RPi.GPIO as GPIO
from datetime import datetime
import requests
import os
from PIL import Image
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

GPIO.setwarnings(False)

# === CONFIG ===
ROBOFLOW_API_KEY = "G4IwXXRbhqRiDPzC9cTE"
ROBOFLOW_MODEL_ID = "pothole-vhmow-naeby/1"
CONF_THRESHOLD = 0.30          # show box if confidence >= 0.30
ALERT_THRESHOLD = 0.75         # only fire alert if confidence >= 0.75
COOLDOWN_SECONDS = 15
STREAM_URL = "http://127.0.0.1:5000/video_feed"
DEFAULT_LAT = 13.322066
DEFAULT_LON = 80.151957
DASHBOARD_URL = "http://127.0.0.1:5000/api/detect"
IMAGE_DIR = "/home/pi/detections"
os.makedirs(IMAGE_DIR, exist_ok=True)

# === SENSOR SETUP ===
TRIG = 4
ECHO = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

MPU_ADDR = 0x68
bus = smbus2.SMBus(1)
bus.write_byte_data(MPU_ADDR, 0x6B, 0)

# === ROBOFLOW CLIENT ===
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=ROBOFLOW_API_KEY
).configure(InferenceConfiguration(api_key_transport="header"))
print("[detector] Roboflow ready:", ROBOFLOW_MODEL_ID)


def detect(frame):
    """Returns (found, max_conf, list_of_potholes)."""
    roi_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(roi_rgb)

    try:
        result = client.infer(pil_image, model_id=ROBOFLOW_MODEL_ID)
    except Exception as e:
        print("[detector] API error:", e)
        return False, 0.0, []

    preds = result.get("predictions", [])
    potholes = [p for p in preds if p.get("class", "").lower() == "pothole"]

    if not potholes:
        return False, 0.0, []

    max_conf = max(p["confidence"] for p in potholes)
    return max_conf >= CONF_THRESHOLD, max_conf, potholes


def draw_boxes(frame, potholes):
    """Draw a box for every detected pothole."""
    display = frame.copy()
    for p in potholes:
        x, y = int(p["x"]), int(p["y"])
        w, h = int(p["width"]), int(p["height"])
        c = p["confidence"]

        x1, y1 = x - w // 2, y - h // 2
        x2, y2 = x + w // 2, y + h // 2

        # Green for alert-worthy, yellow for below alert threshold
        if c >= ALERT_THRESHOLD:
            color = (0, 255, 0)      # green
        else:
            color = (0, 220, 255)    # yellow

        cv2.rectangle(display, (x1, y1), (x2, y2), color, 3)
        cv2.putText(display, f"pothole {c:.2f}", (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Overlay count
    if potholes:
        cv2.putText(display, f"Potholes: {len(potholes)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    return display


def measure_depth():
    GPIO.output(TRIG, False); time.sleep(0.05)
    GPIO.output(TRIG, True); time.sleep(0.00001)
    GPIO.output(TRIG, False)
    start = time.time(); stop = time.time(); timeout = time.time() + 0.04
    while GPIO.input(ECHO) == 0:
        start = time.time()
        if time.time() > timeout:
            return -1
    while GPIO.input(ECHO) == 1:
        stop = time.time()
        if time.time() > timeout:
            return -1
    return round((stop - start) * 34300 / 2, 1)


def read_mpu_z():
    try:
        data = bus.read_i2c_block_data(MPU_ADDR, 0x3B, 6)
        z = (data[4] << 8) | data[5]
        if z > 32768:
            z -= 65536
        return z
    except Exception:
        return 0


# === MAIN ===
print("[detector] Connecting to dashboard stream...")
cap = cv2.VideoCapture(STREAM_URL)
if not cap.isOpened():
    print("[detector] ERROR: Cannot open stream. Is dashboard running?")
    exit(1)

print("[detector] Stream connected.")
print(f"[detector] Visual threshold: {CONF_THRESHOLD} | Alert threshold: {ALERT_THRESHOLD}")
print("[detector] Press Ctrl+C to stop.\n")

last_alert = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.5)
            continue

        found, conf, potholes = detect(frame)
        display = draw_boxes(frame, potholes)

        print(f"[detector] Count: {len(potholes)} | MaxConf: {conf:.2f} | Alert: {conf >= ALERT_THRESHOLD}", end='\r')

        if conf >= ALERT_THRESHOLD and (time.time() - last_alert) > COOLDOWN_SECONDS:
            last_alert = time.time()

            print(f"\n{'='*55}")
            print(f"POTHOLE ALERT - {len(potholes)} pothole(s), max conf {conf:.2f}")
            print(f"{'='*55}")

            accel_z = read_mpu_z()
            depth = measure_depth()
            print(f"[sensors] MPU Z: {accel_z} | Depth: {depth} cm")

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_filename = f"pothole_{ts}.jpg"
            img_path = os.path.join(IMAGE_DIR, img_filename)
            cv2.imwrite(img_path, display)
            print(f"[image] Saved: {img_filename} (with {len(potholes)} box(es))")

            payload = {
                "timestamp": ts,
                "latitude": DEFAULT_LAT,
                "longitude": DEFAULT_LON,
                "depth_cm": depth if depth > 0 else None,
                "accel_z": accel_z,
                "confidence": round(conf, 2),
                "count": len(potholes),
                "image_filename": img_filename
            }

            try:
                r = requests.post(DASHBOARD_URL, json=payload, timeout=3)
                print(f"[dashboard] Sent (status {r.status_code})")
            except Exception as e:
                print(f"[dashboard] Offline: {e}")

            print(f"{'='*55}\n")

        time.sleep(0.5)

except KeyboardInterrupt:
    print("\n[detector] Stopped.")
finally:
    cap.release()
    GPIO.cleanup()
    print("[detector] Cleanup done.")
