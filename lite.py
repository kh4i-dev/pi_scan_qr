# -*- coding: utf-8 -*-
import cv2, time, json, threading
import os
from flask import Flask, Response, send_from_directory
from flask_sock import Sock
import RPi.GPIO as GPIO
# =============================
# ⚙️ CẤU HÌNH CƠ BẢN
# =============================
CAMERA_INDEX = 0
ACTIVE_LOW = True
CYCLE_DELAY = 0.3       # Thời gian đẩy
SETTLE_DELAY = 0.2      # Thời gian chờ
DEBOUNCE = 0.1          # Chống nhiễu sensor
PUSH_DELAY = 0.5        # Thời gian trễ trước khi piston hoạt động
QUEUE_TIMEOUT = 10.0     # Timeout hàng chờ
main_running = True

# --- Khai báo 4 làn ---
LANES = [
    {"id": "A", "name": "Loại A", "sensor": 3,  "pull": 18, "push": 17},
    {"id": "B", "name": "Loại B", "sensor": 23, "pull": 14, "push": 27},
    {"id": "C", "name": "Loại C", "sensor": 24, "pull": 4,  "push": 22},
    {"id": "D", "name": "Loại D (đếm)", "sensor": None, "pull": None, "push": None},
]

# =============================
# 🔌 KHỞI TẠO GPIO
# =============================
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
for lane in LANES:
    if lane["push"]: GPIO.setup(lane["push"], GPIO.OUT)
    if lane["pull"]: GPIO.setup(lane["pull"], GPIO.OUT)
    if lane["sensor"]: GPIO.setup(lane["sensor"], GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Định nghĩa nhanh hai hàm bật/tắt relay
on  = lambda p: GPIO.output(p, GPIO.LOW if ACTIVE_LOW else GPIO.HIGH)
off = lambda p: GPIO.output(p, GPIO.HIGH if ACTIVE_LOW else GPIO.LOW)

def reset_relays():
    """Đưa tất cả relay về trạng thái an toàn (THU bật, ĐẨY tắt)."""
    for lane in LANES:
        if lane["pull"]: on(lane["pull"])
        if lane["push"]: off(lane["push"])
    print("[GPIO] ✅ Đã reset tất cả relay về mặc định.")

# =============================
# 🧩 TRẠNG THÁI TOÀN CỤC
# =============================
queue, queue_lock = [], threading.Lock()   # Hàng chờ QR
latest_frame, frame_lock = None, threading.Lock()  # Khung hình camera
ws_clients, ws_lock = set(), threading.Lock()       # Kết nối WebSocket
counts = [0]*len(LANES)                            # Bộ đếm từng làn

# =============================
# 🪶 HÀM HỖ TRỢ
# =============================
def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def broadcast(event):
    """Gửi sự kiện (log/state_update) đến tất cả client qua WS."""
    data = json.dumps(event)
    with ws_lock:
        for ws in list(ws_clients):
            try: ws.send(data)
            except: ws_clients.remove(ws)

# =============================
# 🎥 LUỒNG CAMERA
# =============================
def camera_thread():
    """Đọc hình từ camera và lưu vào biến latest_frame."""
    global latest_frame
    cam = cv2.VideoCapture(CAMERA_INDEX)
    cam.set(3, 640); cam.set(4, 480)
    if not cam.isOpened():
        log("❌ Không mở được camera.")
        return
    while main_running:
        ok, frame = cam.read()
        if ok:
            with frame_lock: latest_frame = frame
        time.sleep(0.03)
    cam.release()

# =============================
# 🤖 LUỒNG XỬ LÝ CHÍNH (QR + SENSOR)
# =============================
def main_loop():
    """Gộp quét QR + đọc cảm biến vào một vòng lặp chính."""
    det = cv2.QRCodeDetector()
    last_qr, last_time = "", 0
    last_sensor = {}
    timeout_ref = time.time()

    while main_running:
        # --- Đọc khung hình ---
        frame = None
        with frame_lock:
            if latest_frame is not None:
                frame = latest_frame.copy()
        if frame is None:
            time.sleep(0.05)
            continue

        # --- QUÉT QR ---
        data, _, _ = det.detectAndDecode(frame)
        if data and (data != last_qr or time.time() - last_time > 2):
            last_qr, last_time = data, time.time()
            qr = data.strip()
            found = next((i for i, l in enumerate(LANES) if l["id"] == qr), None)
            if found is not None:
                with queue_lock:
                    queue.append(found)
                log(f"📦 QR {qr} → thêm vào hàng chờ {queue}")
                broadcast({"type": "log", "log_type": "qr", "message": f"Phát hiện {qr}", "data": {"queue": queue}})
            else:
                log(f"⚠️ Mã QR không hợp lệ: {qr}")

        # --- TIMEOUT HÀNG CHỜ ---
        if queue and time.time() - timeout_ref > QUEUE_TIMEOUT:
            with queue_lock:
                dropped = queue.pop(0)
            log(f"⏰ Bỏ hàng chờ {LANES[dropped]['id']} do timeout.")
            broadcast({"type": "log", "log_type": "warn", "message": f"Hàng chờ {LANES[dropped]['id']} hết hạn.", "data": {"queue": queue}})
            timeout_ref = time.time()

        # --- ĐỌC SENSOR ---
        for i, lane in enumerate(LANES):
            if not lane["sensor"]: continue
            val = GPIO.input(lane["sensor"])
            if val == 0 and last_sensor.get(i, 1) == 1:
                # Cảm biến phát hiện vật
                threading.Thread(target=sort_cycle, args=(i,), daemon=True).start()
                with queue_lock:
                    if queue and queue[0] == i:
                        queue.pop(0)
                    broadcast({"type": "log", "log_type": "sort", "message": f"Sensor {lane['id']} kích hoạt", "data": {"queue": queue}})
                timeout_ref = time.time()
            last_sensor[i] = val

        time.sleep(0.02)

# =============================
# 🔁 CHU TRÌNH THU - ĐẨY
# =============================
def sort_cycle(i):
    """Thực thi 1 chu trình thu - đẩy của làn i."""
    lane = LANES[i]
    log(f"🚀 Bắt đầu {lane['name']} (delay {PUSH_DELAY}s)")
    time.sleep(PUSH_DELAY)  # Độ trễ trước khi chạy piston

    # Nếu là lane đếm (D)
    if not lane["pull"] or not lane["push"]:
        counts[i] += 1
        log(f"📊 Cập nhật đếm {lane['id']} = {counts[i]}")
        return

    # Chu trình piston
    off(lane["pull"]); time.sleep(SETTLE_DELAY)
    on(lane["push"]);  time.sleep(CYCLE_DELAY)
    off(lane["push"]); time.sleep(SETTLE_DELAY)
    on(lane["pull"])

    counts[i] += 1
    log(f"✅ Hoàn tất {lane['id']} → Tổng: {counts[i]}")

# =============================
# 🌐 FLASK + WEBSOCKET
# =============================
app = Flask(__name__, static_folder=".")
sock = Sock(app)

@app.route("/")
def index():
    return send_from_directory(os.path.dirname(__file__), "index_lite.html")

@app.route("/video_feed")
def video_feed():
    """Luồng video MJPEG gửi cho trình duyệt."""
    def gen():
        while main_running:
            frame = None
            with frame_lock:
                if latest_frame is not None:
                    frame = latest_frame.copy()
            if frame is None:
                time.sleep(0.1)
                continue
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ok: continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@sock.route("/ws")
def ws(ws):
    """WebSocket gửi log & nhận lệnh reset."""
    with ws_lock: ws_clients.add(ws)
    try:
        while True:
            msg = ws.receive()
            if not msg: break
            data = json.loads(msg)
            act = data.get("action")
            if act == "reset_count":
                for j in range(len(counts)): counts[j] = 0
                log("🧹 Đã reset toàn bộ bộ đếm.")
                broadcast({"type": "log", "log_type": "info", "message": "Đã reset toàn bộ đếm."})
            elif act == "reset_queue":
                with queue_lock: queue.clear()
                log("🧹 Reset hàng chờ.")
                broadcast({"type": "log", "log_type": "warn", "message": "Hàng chờ đã được reset.", "data": {"queue": []}})
    finally:
        with ws_lock: ws_clients.discard(ws)

# =============================
# 🏁 MAIN
# =============================
if __name__ == "__main__":
    try:
        reset_relays()
        threading.Thread(target=camera_thread, daemon=True).start()
        threading.Thread(target=main_loop, daemon=True).start()
        log(f"🚀 Hệ thống phân loại khởi động (PUSH_DELAY = {PUSH_DELAY}s, 4 làn)")
        app.run(host="0.0.0.0", port=3000)
    except KeyboardInterrupt:
        log("🛑 Dừng hệ thống (Ctrl+C).")
    finally:
        main_running = False
        GPIO.cleanup()
        print("✅ GPIO đã dọn dẹp. Tạm biệt!")
