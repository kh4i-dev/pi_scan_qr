# -*- coding: utf-8 -*-
"""
app6.2_hybrid.py
Phiên bản kết hợp AI + QR + WebSocket ổn định nhất
Tác giả: Kh4i-dev x ChatGPT
"""

import os, cv2, time, json, threading, logging
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, Response, jsonify, request
from flask_sock import Sock

# ========== MONKEY PATCH (Eventlet) ==========
import eventlet
eventlet.monkey_patch()

# ========== IMPORT AI + QR ==========
try:
    from pyzbar import pyzbar
    from ultralytics import YOLO
    AI_ENABLED = True
except ImportError:
    AI_ENABLED = False

# ========== MOCK GPIO ==========
try:
    import RPi.GPIO as RPiGPIO
except (ImportError, RuntimeError):
    RPiGPIO = None

class MockGPIO:
    BCM = BOARD = OUT = IN = HIGH = LOW = None
    def setmode(self, *a, **k): pass
    def setup(self, *a, **k): pass
    def output(self, *a, **k): pass
    def input(self, *a, **k): return 0
    def cleanup(self): pass

GPIO = RPiGPIO or MockGPIO()

# ========== LOGGER ==========
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler("system.log", encoding="utf-8")])

# ========== APP INIT ==========
app = Flask(__name__)
sock = Sock(app)
executor = ThreadPoolExecutor(max_workers=8)

CONFIG_FILE = "config.json"
CONFIG_LOCK = threading.Lock()
QUEUE_LOCK = threading.Lock()
LOG_LOCK = threading.Lock()

# ========== GLOBAL STATE ==========
queue = []
clients = set()
running = True

# ========== CAMERA ==========
CAMERA_INDEX = 0
camera = cv2.VideoCapture(CAMERA_INDEX)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
camera.set(cv2.CAP_PROP_FPS, 30)

if AI_ENABLED:
    model = YOLO("best.pt")  # bạn có thể đổi model tại đây

# ========== CONFIG ==========
def load_config():
    if not os.path.exists(CONFIG_FILE):
        cfg = {"push_delay": 1.5, "lanes": 3, "enable_auto_train": False}
        save_config(cfg)
        return cfg
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_config(cfg):
    with CONFIG_LOCK:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=4, ensure_ascii=False)

config = load_config()

# ========== BROADCAST ==========
def broadcast_log(msg):
    with LOG_LOCK:
        for ws in list(clients):
            try:
                ws.send(json.dumps({"log": msg}))
            except Exception:
                clients.discard(ws)

# ========== QR DETECTION (HYBRID) ==========
def detect_qr(frame):
    # 1️⃣ OpenCV QRCodeDetector
    qr_detector = cv2.QRCodeDetector()
    data, pts, _ = qr_detector.detectAndDecode(frame)
    if data:
        return data.strip()

    # 2️⃣ Pyzbar fallback
    if AI_ENABLED:
        try:
            barcodes = pyzbar.decode(frame)
            if barcodes:
                return barcodes[0].data.decode("utf-8").strip()
        except Exception:
            pass

    # 3️⃣ YOLO fallback
    if AI_ENABLED:
        try:
            results = model.predict(frame, verbose=False)
            for box in results[0].boxes:
                cls = results[0].names[int(box.cls)]
                if "qr" in cls.lower():
                    return "QR_DETECTED_AI"
        except Exception:
            pass

    return None

# ========== CAMERA THREAD ==========
def camera_thread():
    logging.info("[CAMERA] Thread started")
    while running:
        try:
            ret, frame = camera.read()
            if not ret:
                time.sleep(0.1)
                continue
            qr_data = detect_qr(frame)
            if qr_data:
                with QUEUE_LOCK:
                    queue.append(qr_data)
                broadcast_log(f"QR Detected: {qr_data}")
            time.sleep(0.05)
        except KeyboardInterrupt:
            break
        except Exception as e:
            logging.warning(f"[CAMERA] Error: {e}")
            time.sleep(0.2)
    logging.info("[CAMERA] Thread stopped")

# ========== VIDEO FEED ==========
def generate_video():
    while running:
        try:
            ret, frame = camera.read()
            if not ret:
                continue
            _, buffer = cv2.imencode(".jpg", frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except ConnectionAbortedError:
            logging.info("[VIDEO] Client closed connection.")
            break
        except Exception as e:
            logging.warning(f"[VIDEO] Error: {e}")
            break

@app.route("/video_feed")
def video_feed():
    return Response(generate_video(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# ========== ROUTES ==========
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/config")
def get_config():
    return jsonify(config)

@app.route("/api/reset_queue")
def reset_queue():
    with QUEUE_LOCK:
        queue.clear()
    broadcast_log("Queue reset")
    return jsonify({"status": "ok"})

# ========== WEBSOCKET ==========
@sock.route("/ws")
def ws_route(ws):
    clients.add(ws)
    logging.info("[WS] Client connected.")
    try:
        while True:
            try:
                msg = ws.receive(timeout=1)
                if not msg:
                    time.sleep(0.05)
                    continue
                if msg == "ping":
                    ws.send("pong")
            except Exception as e:
                if "Connection closed" in str(e):
                    break
    finally:
        clients.discard(ws)
        logging.info("[WS] Client disconnected.")

# ========== AUTO SAVE THREAD ==========
def autosave_thread():
    while running:
        time.sleep(60)
        save_config(config)
        logging.info("[CONFIG] Auto-saved")

# ========== STARTUP ==========
if __name__ == "__main__":
    threading.Thread(target=camera_thread, daemon=True).start()
    threading.Thread(target=autosave_thread, daemon=True).start()

    logging.info("=====================================")
    logging.info(" APP 6.2 HYBRID STARTED (AI + QR)")
    logging.info(" URL: http://0.0.0.0:3000")
    logging.info("=====================================")

    eventlet.wsgi.server(eventlet.listen(("0.0.0.0", 3000)), app)
