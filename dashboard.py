"""
dashboard.py - FULL SYSTEM: camera + AI + sensors + dashboard + live feed with boxes
Alert threshold: 0.35
"""
import cv2
import threading
import time
import smbus2
import RPi.GPIO as GPIO
from datetime import datetime
import os
import json
from PIL import Image
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
from flask import Flask, request, jsonify, render_template_string, send_from_directory, Response

GPIO.setwarnings(False)

# === CONFIG ===
ROBOFLOW_API_KEY = "G4IwXXRbhqRiDPzC9cTE"
ROBOFLOW_MODEL_ID = "pothole-vhmow-naeby/1"
CONF_THRESHOLD = 0.35         # alert threshold — anything above this triggers
COOLDOWN_SECONDS = 10         # wait between alerts
CAMERA_ID = 0
DEFAULT_LAT = 13.322066
DEFAULT_LON = 80.151957
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

# === ROBOFLOW ===
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=ROBOFLOW_API_KEY
).configure(InferenceConfiguration(api_key_transport="header"))

# === GLOBAL STATE ===
latest_frame = None
raw_frame = None
frame_lock = threading.Lock()
detections = []
last_alert = 0


def measure_depth():
    GPIO.output(TRIG, False); time.sleep(0.05)
    GPIO.output(TRIG, True); time.sleep(0.00001)
    GPIO.output(TRIG, False)
    start = time.time(); stop = time.time(); timeout = time.time() + 0.04
    while GPIO.input(ECHO) == 0:
        start = time.time()
        if time.time() > timeout: return -1
    while GPIO.input(ECHO) == 1:
        stop = time.time()
        if time.time() > timeout: return -1
    return round((stop - start) * 34300 / 2, 1)


def read_mpu_z():
    try:
        data = bus.read_i2c_block_data(MPU_ADDR, 0x3B, 6)
        z = (data[4] << 8) | data[5]
        if z > 32768: z -= 65536
        return z
    except Exception:
        return 0


def camera_worker():
    """Reads camera at 1280x720, keeps the freshest raw frame."""
    global raw_frame
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        with frame_lock:
            raw_frame = frame.copy()
        time.sleep(0.03)


def ai_worker():
    """Sends frames to Roboflow, draws ALL detected potholes."""
    global latest_frame, last_alert
    while True:
        with frame_lock:
            frame = raw_frame.copy() if raw_frame is not None else None
        if frame is None:
            time.sleep(0.5)
            continue

        try:
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = client.infer(pil, model_id=ROBOFLOW_MODEL_ID)
            preds = result.get("predictions", [])
        except Exception as e:
            print("[ai] API error:", e)
            time.sleep(2)
            continue

        # Collect ALL potholes above the visual threshold
        potholes = [p for p in preds
                    if p.get("class", "").lower() == "pothole"
                    and p.get("confidence", 0) > 0.20]

        annotated = frame.copy()
        best_conf = 0.0

        for p in potholes:
            conf = p["confidence"]
            x, y = int(p["x"]), int(p["y"])
            w, h = int(p["width"]), int(p["height"])
            x1, y1 = x - w // 2, y - h // 2
            x2, y2 = x + w // 2, y + h // 2

            # Green if above alert threshold, yellow otherwise
            color = (0, 255, 0) if conf >= CONF_THRESHOLD else (0, 220, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            cv2.putText(annotated, f"{conf:.2f}", (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if conf > best_conf:
                best_conf = conf

        # Counter in top-left
        if potholes:
            cv2.putText(annotated, f"Potholes: {len(potholes)}", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        # Fire an alert if best confidence >= threshold
        if best_conf >= CONF_THRESHOLD and (time.time() - last_alert) > COOLDOWN_SECONDS:
            last_alert = time.time()

            accel_z = read_mpu_z()
            depth = measure_depth()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_filename = f"pothole_{ts}.jpg"
            cv2.imwrite(os.path.join(IMAGE_DIR, img_filename), annotated)

            entry = {
                "timestamp": ts,
                "latitude": DEFAULT_LAT,
                "longitude": DEFAULT_LON,
                "depth_cm": depth if depth > 0 else None,
                "accel_z": accel_z,
                "confidence": round(best_conf, 2),
                "count": len(potholes),
                "image_filename": img_filename
            }
            detections.append(entry)
            print(f"[ai] ALERT - {len(potholes)} pothole(s), "
                  f"max conf {best_conf:.2f}, depth {depth}cm")

        with frame_lock:
            latest_frame = annotated

        time.sleep(0.5)


def mjpeg_generator():
    """Serves annotated frames as MJPEG."""
    while True:
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None
        if frame is None:
            time.sleep(0.1)
            continue
        ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.05)


# === FLASK APP ===
app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Pothole Detection and Monitoring System</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        * { box-sizing: border-box; }
        body { margin: 0; font-family: Arial; background: #0e1117; color: #e6e6e6; }
        header { background: #1a1f2e; padding: 15px; border-bottom: 3px solid #ff5555;
                 display: flex; justify-content: space-between; align-items: center; }
        h1 { margin: 0; color: #ff5555; font-size: 20px; }
        .status { padding: 6px 12px; border-radius: 20px; font-size: 12px;
                  background: #22cc22; color: #000; font-weight: bold; }
        .status.alert { background: #ff2222; color: #fff; animation: pulse 1s infinite; }
        @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.6; } }
        #alert-banner { display: none; padding: 15px 20px; background: #ff2222;
                        color: #fff; font-weight: bold; font-size: 16px; }
        #alert-banner.show { display: block; }
        .live-section { padding: 12px; background: #1a1f2e; border-bottom: 3px solid #ff5555; }
        .live-section h2 { color: #ff5555; font-size: 16px; margin: 0 0 10px 0; }
        .live-section img { width: 100%; max-width: 720px; display: block;
                            margin: 0 auto; border-radius: 6px; background: #000;
                            border: 2px solid #ff5555; }
        .stats { display: flex; gap: 15px; padding: 12px 15px; background: #1a1f2e; }
        .stat { flex: 1; text-align: center; padding: 8px; background: #0e1117; border-radius: 6px; }
        .stat-num { font-size: 22px; color: #ff5555; font-weight: bold; }
        .stat-label { font-size: 11px; color: #888; margin-top: 2px; }
        #map { height: 35vh; width: 100%; }
        .detections { padding: 10px; }
        .detections h2 { color: #ff5555; font-size: 16px; margin: 10px 0; }
        .card { display: flex; gap: 12px; padding: 12px; margin: 8px 0;
                background: #1a1f2e; border-radius: 6px;
                border-left: 5px solid #ff5555; align-items: center; }
        .card img { width: 120px; height: 90px; object-fit: cover;
                    border-radius: 4px; cursor: pointer; }
        .card .info { flex: 1; font-size: 13px; }
        .card .time { color: #ff7777; font-weight: bold; font-size: 14px; }
        .card .meta { color: #aaa; margin-top: 4px; font-size: 12px; }
    </style>
</head>
<body>
    <header>
        <h1>&#128679; AI Pothole Detection &amp; Monitoring System</h1>
        <div class="status" id="status-badge">&#9679; ACTIVE</div>
    </header>

    <div id="alert-banner"></div>

    <div class="live-section">
        <h2>&#128249; Live Camera Feed (with AI boxes)</h2>
        <img src="/video_feed" alt="Live camera feed">
    </div>

    <div class="stats">
        <div class="stat"><div class="stat-num">{{ count }}</div><div class="stat-label">Alerts</div></div>
        <div class="stat"><div class="stat-num">{{ avg_depth }}</div><div class="stat-label">Avg Depth cm</div></div>
        <div class="stat"><div class="stat-num">{{ avg_conf }}</div><div class="stat-label">Avg Confidence</div></div>
    </div>

    <div id="map"></div>

    <div class="detections">
        <h2>&#128276; Recent Pothole Alerts</h2>
        {% if detections|length == 0 %}
            <p style="color: #888; padding: 20px; text-align: center;">
                No potholes detected yet. Point the camera at a pothole image to test.
            </p>
        {% endif %}
        {% for d in detections %}
        <div class="card">
            {% if d.image_filename %}
            <img src="/detections/{{ d.image_filename }}" onclick="window.open(this.src)">
            {% endif %}
            <div class="info">
                <div class="time">&#9888;&#65039; {{ d.timestamp }}</div>
                <div class="meta">
                    &#128207; Depth: {{ d.depth_cm if d.depth_cm else 'N/A' }} cm |
                    &#128243; MPU: {{ d.accel_z }} |
                    &#127919; Conf: {{ d.confidence }} |
                    &#128220; Count: {{ d.count if d.count else 1 }}
                </div>
            </div>
        </div>
        {% endfor %}
    </div>

    <script>
        var map = L.map('map').setView([13.322066, 80.151957], 16);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
                    { attribution: '&copy; OpenStreetMap' }).addTo(map);
        var detections = {{ detections_json|safe }};
        detections.forEach(function(d) {
            if (d.latitude && d.longitude) {
                L.circleMarker([d.latitude, d.longitude], {
                    color: '#ff5555', radius: 8, fillOpacity: 0.8
                }).addTo(map).bindPopup(
                    "Pothole<br>Time: " + d.timestamp +
                    "<br>Depth: " + (d.depth_cm || 'N/A') + " cm" +
                    "<br>Conf: " + d.confidence);
            }
        });
        var lastCount = detections.length;
        setInterval(function() {
            fetch('/api/detections').then(r => r.json()).then(function(data) {
                if (data.length > lastCount) {
                    var d = data[data.length - 1];
                    var banner = document.getElementById('alert-banner');
                    banner.innerHTML = '&#128680; NEW POTHOLE - Depth: ' +
                        (d.depth_cm || 'N/A') + 'cm - Conf: ' + d.confidence;
                    banner.classList.add('show');
                    document.getElementById('status-badge').textContent = 'ALERT';
                    document.getElementById('status-badge').classList.add('alert');
                    lastCount = data.length;
                    setTimeout(() => location.reload(), 2000);
                }
            }).catch(() => {});
        }, 3000);
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    depths = [d['depth_cm'] for d in detections if d.get('depth_cm')]
    confs = [d['confidence'] for d in detections if d.get('confidence')]
    avg_depth = round(sum(depths) / len(depths), 1) if depths else 0
    avg_conf = round(sum(confs) / len(confs), 2) if confs else 0
    return render_template_string(
        HTML,
        detections=list(reversed(detections[-20:])),
        detections_json=json.dumps(detections),
        count=len(detections),
        avg_depth=avg_depth,
        avg_conf=avg_conf
    )


@app.route('/video_feed')
def video_feed():
    return Response(mjpeg_generator(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/detect', methods=['POST'])
def receive():
    return jsonify({"status": "internal_only"})


@app.route('/api/detections')
def get_detections():
    return jsonify(detections)


@app.route('/detections/<path:filename>')
def serve_image(filename):
    return send_from_directory(IMAGE_DIR, filename)


if __name__ == '__main__':
    threading.Thread(target=camera_worker, daemon=True).start()
    threading.Thread(target=ai_worker, daemon=True).start()
    print("[system] Dashboard + camera + AI + sensors in one process")
    print("[system] Alert threshold: 0.35")
    print("[system] Open http://192.168.150.228:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
