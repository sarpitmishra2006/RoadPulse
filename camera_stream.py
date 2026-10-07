"""
camera_stream.py - MJPEG stream server for live dashboard view
"""

import cv2
from flask import Flask, Response, render_template_string

app = Flask(__name__)
CAMERA_ID = 0

def generate_frames():
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 10)

    while True:
        success, frame = cap.read()
        if not success:
            continue
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    print("[camera_stream] Stream at http://192.168.150.228:5001/video_feed")
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
