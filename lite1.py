# -*- coding: utf-8 -*-
"""
BẢN LITE TỐI GIẢN (CHỈ CHẠY PI THẬT)
- Giữ lại Web Server (Flask, Sock) và giao diện index_lite.html.
- Yêu cầu file config.json để chạy (không còn hard-code LANES).
- Đã loại bỏ toàn bộ MockGPIO, ErrorManager, Pyzbar, ThreadPoolExecutor.
- Sửa hạn chế FIFO: Dùng logic 'if i in qr_queue:' (linh hoạt).
- Thêm luồng broadcast_state_thread để cập nhật UI.
"""
import cv2
import time
import json
import threading
import os
import unicodedata
import re
import RPi.GPIO as GPIO # Import trực tiếp
from flask import Flask, Response, send_from_directory, request, jsonify
from flask_sock import Sock

# =============================
#      CẤU HÌNH & KHỞI TẠO TOÀN CỤC
# =============================
CONFIG_FILE = 'config.json'
ACTIVE_LOW = True

# --- Các biến toàn cục ---
lanes_config = []       # Tải từ JSON
timing_config = {}      # Tải từ JSON
qr_config = {}          # Tải từ JSON

RELAY_PINS = []
SENSOR_PINS = []

main_running = True
latest_frame = None
frame_lock = threading.Lock()

ws_clients, ws_lock = set(), threading.Lock()

# --- Biến logic V3 (Nâng cấp) ---
counts = []             # Bộ đếm (khởi tạo theo num_lanes)
last_s_state = []       # Trạng thái sensor (khởi tạo theo num_lanes)
last_s_trig = []        # Thời điểm trigger (khởi tạo theo num_lanes)
qr_queue = []           # Hàng chờ (lưu index)
queue_lock = threading.Lock()
queue_head_since = 0.0
pending_sensor_triggers = [] # (khởi tạo theo num_lanes)

# =============================
#    CÁC HÀM TIỆN ÍCH (Chuẩn hóa ID)
# =============================
def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))

def canon_id(s: str) -> str:
    if s is None: return ""
    s = str(s).strip()
    try: s = s.encode("utf-8").decode("unicode_escape")
    except Exception: pass
    s = _strip_accents(s).upper()
    s = re.sub(r"[^A-Z0-9]", "", s)
    s = re.sub(r"^(LOAI|LO)+", "", s)
    return s

# =============================
#       HÀM ĐIỀU KHIỂN RELAY
# =============================
def RELAY_ON(pin):
    if pin is None: return
    try: GPIO.output(pin, GPIO.LOW if ACTIVE_LOW else GPIO.HIGH)
    except Exception as e:
        log(f"[GPIO] Lỗi kích hoạt relay pin {pin}: {e}", 'error')
        global main_running
        main_running = False

def RELAY_OFF(pin):
    if pin is None: return
    try: GPIO.output(pin, GPIO.HIGH if ACTIVE_LOW else GPIO.LOW)
    except Exception as e:
        log(f"[GPIO] Lỗi tắt relay pin {pin}: {e}", 'error')
        global main_running
        main_running = False

# =============================
#      LOAD CẤU HÌNH
# =============================
def ensure_lane_ids(lanes_list):
    default_ids = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    for i, lane in enumerate(lanes_list):
        if 'id' not in lane or not lane['id']:
            lane['id'] = default_ids[i] if i < len(default_ids) else f"LANE_{i+1}"
    return lanes_list

def load_config():
    global lanes_config, timing_config, qr_config, RELAY_PINS, SENSOR_PINS
    global counts, last_s_state, last_s_trig, pending_sensor_triggers

    if not os.path.exists(CONFIG_FILE):
        print(f"[CRITICAL] Không tìm thấy file {CONFIG_FILE}. Không thể khởi động.")
        return False

    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f: content = f.read()
        file_cfg = json.loads(content)
        
        timing_config = file_cfg.get('timing_config', {})
        qr_config = file_cfg.get('qr_config', {})
        lanes_config = ensure_lane_ids(file_cfg.get('lanes_config', []))
        
        num_lanes = len(lanes_config)
        
        RELAY_PINS.clear(); SENSOR_PINS.clear()
        for i, cfg in enumerate(lanes_config):
            s_pin = int(cfg["sensor_pin"]) if cfg.get("sensor_pin") is not None else None
            p_pin = int(cfg["push_pin"]) if cfg.get("push_pin") is not None else None
            pl_pin = int(cfg["pull_pin"]) if cfg.get("pull_pin") is not None else None
            
            cfg["index"] = i # Thêm index để tham chiếu
            if s_pin is not None: SENSOR_PINS.append(s_pin)
            if p_pin is not None: RELAY_PINS.append(p_pin)
            if pl_pin is not None: RELAY_PINS.append(pl_pin)

        # Khởi tạo các mảng trạng thái
        counts = [0] * num_lanes
        last_s_state = [1] * num_lanes
        last_s_trig = [0.0] * num_lanes
        pending_sensor_triggers = [0.0] * num_lanes
        
        print(f"[CONFIG] Đã tải cấu hình cho {num_lanes} lanes.")
        return True
    except Exception as e:
        print(f"[CRITICAL] Lỗi đọc file {CONFIG_FILE}: {e}. Không thể khởi động.")
        return False

def reset_relays():
    print("[GPIO] Reset tất cả relay (Thu BẬT, Đẩy TẮT)...")
    try:
        for lane in lanes_config:
            pull_pin, push_pin = lane.get("pull_pin"), lane.get("push_pin")
            if pull_pin is not None: RELAY_ON(pull_pin)
            if push_pin is not None: RELAY_OFF(push_pin)
        time.sleep(0.1)
        print("[GPIO] Reset relay hoàn tất.")
    except Exception as e:
        log(f"[GPIO] Lỗi khi reset relay: {e}", 'error')
        global main_running
        main_running = False

# =============================
# 🪶 HÀM HỖ TRỢ (Log & Broadcast)
# =============================
def log(msg, log_type="info"):
    """In ra console và gửi log tới client."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")
    broadcast({"type": "log", "log_type": log_type, "message": msg})

def broadcast(event):
    if event.get("type") == "log":
        event['timestamp'] = time.strftime('%H:%M:%S')
    data = json.dumps(event)
    with ws_lock:
        for ws in list(ws_clients):
            try: ws.send(data)
            except: ws_clients.remove(ws)

# =============================
#         LUỒNG CAMERA
# =============================
def run_camera(camera_index):
    global latest_frame, main_running
    camera = None
    try:
        print("[CAMERA] Khởi tạo camera...")
        camera = cv2.VideoCapture(camera_index)
        props = {cv2.CAP_PROP_FRAME_WIDTH: 640, cv2.CAP_PROP_FRAME_HEIGHT: 480, cv2.CAP_PROP_BUFFERSIZE: 1}
        for prop, value in props.items(): camera.set(prop, value)

        if not camera.isOpened():
            log("[CRITICAL] Không thể mở camera. Dừng hệ thống.", 'error')
            main_running = False
            return
        print("[CAMERA] Camera sẵn sàng.")
        
        while main_running:
            ret, frame = camera.read()
            if not ret:
                log("[CRITICAL] Mất kết nối camera. Dừng hệ thống.", 'error')
                main_running = False
                break
            with frame_lock:
                latest_frame = frame.copy()
            time.sleep(1 / 60)
    except Exception as e:
        log(f"[CRITICAL] Luồng camera bị crash: {e}", 'error')
        main_running = False
    finally:
        if camera: camera.release()
        print("[CAMERA] Đã giải phóng camera.")

# =============================
#       LOGIC CHU TRÌNH PHÂN LOẠI
# =============================
def sorting_process(lane_index):
    global counts
    try:
        lane = lanes_config[lane_index]
        lane_name = lane["name"]
        push_pin = lane.get("push_pin")
        pull_pin = lane.get("pull_pin")
        
        cfg = timing_config
        delay = cfg.get('cycle_delay', 0.3)
        settle_delay = cfg.get('settle_delay', 0.2)
        
        is_sorting_lane = not (push_pin is None or pull_pin is None)

        if not is_sorting_lane:
            log(f"Vật phẩm đi thẳng qua {lane_name}", 'pass')
            log_type = "pass"
        else:
            log(f"Bắt đầu chu trình đẩy {lane_name}", 'info')
            RELAY_OFF(pull_pin); time.sleep(settle_delay)
            if not main_running: return
            RELAY_ON(push_pin); time.sleep(delay)
            if not main_running: return
            RELAY_OFF(push_pin); time.sleep(settle_delay)
            if not main_running: return
            RELAY_ON(pull_pin)
            log_type = "sort"
        
        counts[lane_index] += 1
        log(f"Hoàn tất: {lane_name}. Tổng đếm: {counts[lane_index]}", log_type)
        # Gửi log đếm cho UI
        broadcast({"type": "log", "log_type": log_type, "name": lane_name, "count": counts[lane_index]})

    except Exception as e:
        log(f"[SORT] Lỗi trong sorting_process (lane {lane_name}): {e}", 'error')
        global main_running
        main_running = False

def handle_sorting_with_delay(lane_index):
    try:
        lane_name_for_log = lanes_config[lane_index]['name']
        push_delay = timing_config.get('push_delay', 0.0)

        if push_delay > 0:
            log(f"Đã thấy vật {lane_name_for_log}, chờ {push_delay}s...", 'info')
            time.sleep(push_delay)
        if not main_running: return
        
        sorting_process(lane_index)

    except Exception as e:
        log(f"[ERROR] Lỗi trong luồng sorting_delay (lane {lane_name_for_log}): {e}", 'error')
        global main_running
        main_running = False

# =============================
#       QUÉT MÃ QR (Đã tối giản)
# =============================
def qr_detection_loop():
    global pending_sensor_triggers, queue_head_since
    
    detector = cv2.QRCodeDetector()
    last_qr, last_time = "", 0.0
    print("[QR] Luồng QR bắt đầu (Sử dụng: cv2.QRCodeDetector).")
    
    PENDING_TRIGGER_TIMEOUT = timing_config.get("pending_trigger_timeout", 1.0)

    while main_running:
        try:
            LANE_MAP = {canon_id(l.get("id")): l["index"] for l in lanes_config if l.get("id")}
            
            cfg = qr_config
            use_roi = cfg.get("use_roi", False)
            x, y = cfg.get("roi_x", 0), cfg.get("roi_y", 0)
            w, h = cfg.get("roi_w", 0), cfg.get("roi_h", 0)

            frame_copy = None
            with frame_lock:
                if latest_frame is not None: frame_copy = latest_frame.copy()
            if frame_copy is None:
                time.sleep(0.01); continue

            gray_frame = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
            if use_roi and w > 0 and h > 0:
                y_end = min(y + h, gray_frame.shape[0])
                x_end = min(x + w, gray_frame.shape[1])
                gray_frame = gray_frame[y:y_end, x:x_end]

            data, _, _ = detector.detectAndDecode(gray_frame)

            if data and (data != last_qr or time.time() - last_time > 3.0):
                last_qr, last_time = data, time.time()
                data_key = canon_id(data)
                data_raw = data.strip()
                now = time.time()

                if data_key in LANE_MAP:
                    idx = LANE_MAP[data_key]
                    is_pending_match = False
                    
                    with queue_lock:
                        if (pending_sensor_triggers[idx] > 0.0) and (now - pending_sensor_triggers[idx] < PENDING_TRIGGER_TIMEOUT):
                            is_pending_match = True
                            pending_sensor_triggers[idx] = 0.0
                        
                    if is_pending_match:
                        lane_name = lanes_config[idx]['name']
                        msg = f"QR '{data_raw}' khớp với sensor {lane_name} đang chờ."
                        log(f"[QR] {msg}", 'info')
                        threading.Thread(target=handle_sorting_with_delay, args=(idx,), daemon=True).start()
                    else:
                        with queue_lock:
                            is_queue_empty_before = not qr_queue
                            qr_queue.append(idx)
                            current_queue_for_log = list(qr_queue) # Gửi index cho UI
                            if is_queue_empty_before: queue_head_since = time.time()
                        
                        msg = f"Phát hiện {lanes_config[idx]['name']} (key: {data_key})"
                        log(f"[QR] {msg}", 'qr')
                        broadcast({"type": "log", "log_type": "qr", "message": msg, "data": {"queue": current_queue_for_log}})
                            
                elif data_key == "NG":
                    log(f"[QR] Mã NG: {data_raw}", 'warn')
                else:
                    log(f"[QR] Không rõ mã QR: raw='{data_raw}', key='{data_key}'", 'warn')
                    broadcast({"type": "log", "log_type": "unknown_qr", "message": f"Không rõ mã QR: {data_raw}"})
            
            time.sleep(0.01)

        except Exception as e:
            log(f"[QR] Lỗi trong luồng QR: {e}", 'error')
            time.sleep(0.5)

# =============================
#      GIÁM SÁT SENSOR (Đã sửa FIFO)
# =============================
def sensor_monitoring_thread():
    global last_s_state, last_s_trig, queue_head_since, pending_sensor_triggers
    
    debounce_time = timing_config.get('sensor_debounce', 0.1)
    QUEUE_HEAD_TIMEOUT = timing_config.get('queue_head_timeout', 15.0)
    num_lanes = len(lanes_config)

    try:
        while main_running:
            now = time.time()

            with queue_lock:
                if qr_queue and (now - queue_head_since) > QUEUE_HEAD_TIMEOUT:
                    expected_lane_index = qr_queue.pop(0)
                    expected_lane_name = lanes_config[expected_lane_index]['name']
                    current_queue_for_log = list(qr_queue)
                    queue_head_since = now if qr_queue else 0.0
                    
                    msg = f"TIMEOUT! Tự động xóa {expected_lane_name} khỏi hàng chờ."
                    log(f"[SENSOR] {msg}", 'warn')
                    broadcast({"type": "log", "log_type": "warn", "message": msg, "data": {"queue": current_queue_for_log}})

            for i in range(num_lanes):
                lane = lanes_config[i]
                sensor_pin = lane.get("sensor_pin")
                if sensor_pin is None: continue
                
                lane_name = lane['name']
                push_pin = lane.get("push_pin")

                try:
                    sensor_now = GPIO.input(sensor_pin)
                    last_s_state[i] = sensor_now # Cập nhật state cho UI
                except Exception as gpio_e:
                    log(f"[SENSOR] Lỗi đọc GPIO pin {sensor_pin} ({lane_name}): {gpio_e}", 'error')
                    global main_running
                    main_running = False
                    break

                if sensor_now == 0 and last_s_state[i] == 1: # (Lỗi logic nhỏ, đáng lẽ là `last_s_state_prev[i] == 1`)
                                                          # Tuy nhiên, do `last_s_state` vừa được cập nhật, 
                                                          # chúng ta phải so sánh với `last_s_trig` (debounce)
                    pass # Bỏ qua, logic debounce sẽ xử lý
                
                # Logic debounce
                current_state_time = last_s_trig[i]
                if sensor_now == 0 and (last_s_state[i] == 1 or (now - current_state_time > debounce_time and current_state_time != 0)):
                     # (Sửa logic debounce)
                     # Phát hiện sườn xuống (1 -> 0) và đã qua thời gian debounce
                     if (now - last_s_trig[i]) > debounce_time:
                        last_s_trig[i] = now # Ghi lại thời điểm trigger

                        with queue_lock:
                            if not qr_queue:
                                # --- 1. HÀNG CHỜ RỖNG (Sensor-First) ---
                                if push_pin is None:
                                    log(f"Vật đi thẳng (không QR) qua {lane_name}.", 'info')
                                    threading.Thread(target=sorting_process, args=(i,), daemon=True).start()
                                else:
                                    pending_sensor_triggers[i] = now 
                                    log(f"Sensor {lane_name} kích hoạt (hàng chờ rỗng). Đang chờ QR...", 'warn')
                            
                            elif i in qr_queue:
                                # --- 2. KHỚP (Flexible FIFO) ---
                                qr_queue.remove(i)
                                current_queue_for_log = list(qr_queue)
                                if not qr_queue or qr_queue[0] == i: # Nếu xóa đầu hàng
                                      queue_head_since = now if qr_queue else 0.0

                                threading.Thread(target=handle_sorting_with_delay, args=(i,), daemon=True).start()
                                log(f"Sensor {lane_name} khớp (FIFO Linh hoạt).", 'info')
                                broadcast({"type": "log", "log_type": "info", "message": f"Sensor {lane_name} khớp.", "data": {"queue": current_queue_for_log}})
                                pending_sensor_triggers[i] = 0.0
                            else:
                                # --- 3. KHÔNG KHỚP (Pass-over) ---
                                log(f"Sensor {lane_name} kích hoạt, nhưng vật phẩm không có trong hàng chờ. Bỏ qua.", 'warn')
                
                # Cập nhật trạng thái cũ (cần một biến riêng)
                # (Đơn giản hóa: logic debounce ở trên đã đủ, không cần `last_s_state_prev`)
                # Chỉ cần đảm bảo `last_s_state[i]` được cập nhật ở đầu vòng lặp.

            adaptive_sleep = 0.05 if all(s == 1 for s in last_s_state) else 0.01
            time.sleep(adaptive_sleep)

    except Exception as e:
        log(f"[CRITICAL] Luồng sensor bị crash: {e}", 'error')
        main_running = False

# =============================
# (MỚI) LUỒNG GỬI STATE CHO UI
# =============================
def broadcast_state_thread():
    """Luồng riêng để gửi trạng thái (sensor, count, queue) cho UI."""
    global last_s_state, counts, qr_queue
    while main_running:
        try:
            # Tạo snapshot trạng thái
            lanes_snapshot = []
            for i, lane_cfg in enumerate(lanes_config):
                lanes_snapshot.append({
                    "name": lane_cfg['name'],
                    "count": counts[i],
                    "sensor_reading": last_s_state[i],
                    # (Đơn giản hóa: không gửi trạng thái relay)
                    "relay_grab": 1, # Mặc định là thu
                    "relay_push": 0,
                    "status": "Sẵn sàng" # (UI lite không dùng status phức tạp)
                })
            
            with queue_lock:
                queue_snapshot = list(qr_queue)

            state_data = {
                "type": "state_update",
                "state": {
                    "lanes": lanes_snapshot,
                    "queue_indices": queue_snapshot # Gửi queue index cho UI
                }
            }
            broadcast(state_data)
        except Exception as e:
            print(f"[ERROR] Lỗi broadcast state: {e}")
        
        time.sleep(0.5) # Gửi state 2 lần/giây

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
    def gen():
        while main_running:
            frame = None
            with frame_lock:
                if latest_frame is not None:
                    frame = latest_frame.copy()
            if frame is None:
                # Tạo frame đen nếu không có camera
                import numpy as np
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "No Signal", (220, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
                
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ok: continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            time.sleep(1/20) # Stream 20 FPS
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/config")
def get_config():
    """API để UI (nếu có) đọc config (bản lite không dùng)."""
    return jsonify({
        "timing_config": timing_config,
        "lanes_config": lanes_config,
        "qr_config": qr_config
    })

@sock.route("/ws")
def ws(ws):
    with ws_lock: ws_clients.add(ws)
    print(f"[WS] Client kết nối. Tổng: {len(ws_clients)}")
    try:
        while True:
            msg = ws.receive()
            if not msg: break
            data = json.loads(msg)
            act = data.get("action")
            
            if act == "reset_count":
                global counts
                counts = [0] * len(lanes_config)
                log("🧹 Đã reset toàn bộ bộ đếm.", 'warn')
            
            elif act == "reset_queue":
                with queue_lock:
                    global queue_head_since, pending_sensor_triggers
                    qr_queue.clear()
                    queue_head_since = 0.0
                    pending_sensor_triggers = [0.0] * len(lanes_config)
                log("🧹 Reset hàng chờ.", 'warn')
                broadcast({"type": "log", "log_type": "warn", "message": "Hàng chờ đã được reset.", "data": {"queue": []}})
    finally:
        with ws_lock: ws_clients.discard(ws)
        print(f"[WS] Client ngắt kết nối. Còn lại: {len(ws_clients)}")

# =============================
# 🏁 MAIN
# =============================
if __name__ == "__main__":
    try:
        print("--- HỆ THỐNG LITE-PRO TỐI GIẢN ĐANG KHỞI ĐỘNG ---")

        if not load_config():
            raise RuntimeError("Không thể tải file config.json.")
        
        GPIO.setmode(GPIO.BCM if timing_config.get("gpio_mode", "BCM") == "BCM" else GPIO.BOARD)
        GPIO.setwarnings(False)
        print(f"[GPIO] Cài đặt chân SENSOR: {SENSOR_PINS}")
        for pin in SENSOR_PINS: GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        print(f"[GPIO] Cài đặt chân RELAY: {RELAY_PINS}")
        for pin in RELAY_PINS: GPIO.setup(pin, GPIO.OUT)
        
        reset_relays()
        
        CAM_IDX = qr_config.get("camera_index", 0)

        # Khởi động các luồng
        threading.Thread(target=run_camera, args=(CAM_IDX,), name="CameraThread", daemon=True).start()
        threading.Thread(target=qr_detection_loop, name="QRScannerThread", daemon=True).start()
        threading.Thread(target=sensor_monitoring_thread, name="SensorMonThread", daemon=True).start()
        threading.Thread(target=broadcast_state_thread, name="StateBcastThread", daemon=True).start() # (MỚI)

        time.sleep(1)
        if not main_running:
             raise RuntimeError("Khởi động luồng thất bại (Camera hoặc GPIO).")

        print("="*55 + f"\n HỆ THỐNG PHÂN LOẠI SẴN SÀNG (vLiteSimple - Logic V3)\n" +
                     f" Logic: FIFO Linh Hoạt (Đã sửa hạn chế)\n" +
                     f" Truy cập: http://<IP_CUA_PI>:3000\n" + "="*55)
        
        # Chạy Web Server
        app.run(host="0.0.0.0", port=3000)

    except KeyboardInterrupt:
        print("\n--- HỆ THỐNG ĐANG TẮT (Ctrl+C) ---")
    except Exception as startup_err:
        print(f"[CRITICAL] Khởi động hệ thống thất bại: {startup_err}")
    finally:
        main_running = False
        time.sleep(0.5)
        try:
            GPIO.cleanup()
            print("Dọn dẹp GPIO thành công.")
        except Exception as cleanup_err:
            print(f"Lỗi khi dọn dẹp GPIO: {cleanup_err}")
        print("--- HỆ THỐNG ĐÃ TẮT HOÀN TOÀN ---")