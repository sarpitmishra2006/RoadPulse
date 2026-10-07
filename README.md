# RoadPulse

AI-powered pothole detection and monitoring system using Raspberry Pi 4.

## Hardware
- Raspberry Pi 4B (8GB)
- USB Webcam
- MPU6050 (accelerometer/gyroscope)
- HC-SR04 (ultrasonic depth sensor)
- SIM800/900 GSM module (planned)

## Software Stack
- Python 3.13
- OpenCV
- Flask
- Roboflow Hosted API (pothole-vhmow-naeby/1)
- YOLOv8 (trained on 1,302 pothole images, 84.9% mAP)

## Features
- Real-time pothole detection via webcam
- Live camera feed with bounding boxes
- MPU6050 shock detection
- HC-SR04 depth measurement
- Web dashboard with map and alerts
- GPS location tagging (RMKCET default)

## Setup
1. Clone this repo
2. `python3 -m venv pothole-env && source pothole-env/bin/activate`
3. `pip install opencv-python-headless onnxruntime flask numpy requests smbus2 pyserial pillow inference-sdk RPi.GPIO`
4. Get Roboflow API key from https://roboflow.com
5. Update `dashboard.py` with your API key
6. `python3 dashboard.py`
7. Open http://<pi-ip>:5000

## Author
Arpit Mishra
