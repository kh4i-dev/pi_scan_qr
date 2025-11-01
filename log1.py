# -*- coding: utf-8 -*-
"""
BẢN LOG TỐI GIẢN (CHỈ CHẠY PI THẬT)
- Đã loại bỏ toàn bộ MockGPIO, ErrorManager, Pyzbar, ThreadPoolExecutor.
- Chỉ chạy trên Pi thật (import RPi.GPIO trực tiếp).
- Sửa hạn chế FIFO: Dùng logic 'if i in qr_queue:' (linh hoạt) thay vì 'if i == qr_queue[0]:' (cứng nhắc).
- Tải các giá trị timeout từ config.json.
- Yêu cầu file config.json để chạy.
"""
import cv2
import time
import json
import threading
import logging
import os
import unicodedata
import re
import RPi.GPIO as GPIO # Import trực tiếp, chỉ chạy trên Pi

# =============================
#      CẤU HÌNH & KHỞI TẠO TOÀN CỤC
# =============================
CONFIG_FILE = 'config.json'
LOG_FILE = 'system.log'
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

qr_queue = []
queue_lock = threading.Lock()
queue_head_since = 0.0
pending_sensor_triggers = []

last_s_state, last_s_trig = [], []

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
    try:
        GPIO.output(pin, GPIO.LOW if ACTIVE_LOW else GPIO.HIGH)
    except Exception as e:
        logging.error(f"[GPIO] Lỗi kích hoạt relay pin {pin}: {e}")
        # Lỗi nghiêm trọng, dừng hệ thống
        global main_running
        main_running = False

def RELAY_OFF(pin):
    if pin is None: return
    try:
        GPIO.output(pin, GPIO.HIGH if ACTIVE_LOW else GPIO.LOW)
    except Exception as e:
        logging.error(f"[GPIO] Lỗi tắt relay pin {pin}: {e}")
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
    global last_s_state, last_s_trig, pending_sensor_triggers

    if not os.path.exists(CONFIG_FILE):
        logging.critical(f"[CRITICAL] Không tìm thấy file {CONFIG_FILE}. Không thể khởi động.")
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

        last_s_state = [1] * num_lanes; last_s_trig = [0.0] * num_lanes
        pending_sensor_triggers = [0.0] * num_lanes
        
        logging.info(f"[CONFIG] Đã tải cấu hình cho {num_lanes} lanes.")
        return True
    except Exception as e:
        logging.critical(f"[CRITICAL] Lỗi đọc file {CONFIG_FILE}: {e}. Không thể khởi động.")
        return False

def reset_relays():
    logging.info("[GPIO] Reset tất cả relay (Thu BẬT, Đẩy TẮT)...")
    try:
        for lane in lanes_config:
            pull_pin, push_pin = lane.get("pull_pin"), lane.get("push_pin")
            if pull_pin is not None: RELAY_ON(pull_pin)
            if push_pin is not None: RELAY_OFF(push_pin)
        time.sleep(0.1)
        logging.info("[GPIO] Reset relay hoàn tất.")
    except Exception as e:
        logging.error(f"[GPIO] Lỗi khi reset relay: {e}")
        global main_running
        main_running = False

# =============================
#         LUỒNG CAMERA
# =============================
def run_camera(camera_index):
    global latest_frame, main_running
    camera = None
    try:
        logging.info("[CAMERA] Khởi tạo camera...")
        camera = cv2.VideoCapture(camera_index)
        props = {cv2.CAP_PROP_FRAME_WIDTH: 640, cv2.CAP_PROP_FRAME_HEIGHT: 480, cv2.CAP_PROP_BUFFERSIZE: 1}
        for prop, value in props.items(): camera.set(prop, value)

        if not camera.isOpened():
            logging.critical("[CRITICAL] Không thể mở camera. Dừng hệ thống.")
            main_running = False
            return

        logging.info("[CAMERA] Camera sẵn sàng.")
        while main_running:
            ret, frame = camera.read()
            if not ret:
                logging.critical("[CRITICAL] Mất kết nối camera. Dừng hệ thống.")
                main_running = False # Lỗi nghiêm trọng
                break

            with frame_lock:
                latest_frame = frame.copy()
            time.sleep(1 / 60) # Chụp 60 FPS

    except Exception as e:
        logging.critical(f"[CRITICAL] Luồng camera bị crash: {e}")
        main_running = False
    finally:
        if camera: camera.release()
        logging.info("[CAMERA] Đã giải phóng camera.")

# =============================
#       LOGIC CHU TRÌNH PHÂN LOẠI
# =============================
def sorting_process(lane_index):
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
            logging.info(f"Vật phẩm đi thẳng qua {lane_name}")
        else:
            logging.info(f"Bắt đầu chu trình đẩy {lane_name}")
            RELAY_OFF(pull_pin); time.sleep(settle_delay)
            if not main_running: return
            RELAY_ON(push_pin); time.sleep(delay)
            if not main_running: return
            RELAY_OFF(push_pin); time.sleep(settle_delay)
            if not main_running: return
            RELAY_ON(pull_pin)
        
        # Log đếm
        # (Trong bản log-only, chúng ta không duy trì state `count`)
        logging.info(f"Hoàn tất: {lane_name}")

    except Exception as e:
        logging.error(f"[SORT] Lỗi trong sorting_process (lane {lane_name}): {e}")
        global main_running
        main_running = False

def handle_sorting_with_delay(lane_index):
    try:
        lane_name_for_log = lanes_config[lane_index]['name']
        push_delay = timing_config.get('push_delay', 0.0)

        if push_delay > 0:
            logging.info(f"Đã thấy vật {lane_name_for_log}, chờ {push_delay}s...")
            time.sleep(push_delay)
        if not main_running: return
        
        sorting_process(lane_index)

    except Exception as e:
        logging.error(f"[ERROR] Lỗi trong luồng sorting_delay (lane {lane_index}): {e}")
        global main_running
        main_running = False

# =============================
#       QUÉT MÃ QR (Đã tối giản)
# =============================
def qr_detection_loop():
    global pending_sensor_triggers, queue_head_since
    
    # Chỉ dùng cv2.QRCodeDetector
    detector = cv2.QRCodeDetector()
    last_qr, last_time = "", 0.0
    logging.info("[QR] Luồng QR bắt đầu (Sử dụng: cv2.QRCodeDetector).")
    
    # Tải timeout từ config
    PENDING_TRIGGER_TIMEOUT = timing_config.get("pending_trigger_timeout", 1.0) # An toàn hơn

    while main_running:
        try:
            # Tạo LANE_MAP động (chuẩn hóa ID)
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
                            pending_sensor_triggers[idx] = 0.0 # Xóa cờ chờ
                        
                    if is_pending_match:
                        lane_name = lanes_config[idx]['name']
                        logging.info(f"[QR] '{data_raw}' (key: '{data_key}') -> lane {idx} (Khớp pending sensor {lane_name})")
                        threading.Thread(target=handle_sorting_with_delay, args=(idx,), daemon=True).start()
                    else:
                        with queue_lock:
                            is_queue_empty_before = not qr_queue
                            qr_queue.append(idx) # Thêm index
                            queue_log = [lanes_config[i]['name'] for i in qr_queue]
                            if is_queue_empty_before: queue_head_since = time.time()
                        
                        logging.info(f"[QR] '{data_raw}' (key: '{data_key}') -> lane {idx}. Hàng chờ: {queue_log}")
                            
                elif data_key == "NG":
                    logging.warning(f"[QR] Mã NG: {data_raw}")
                else:
                    logging.warning(f"[QR] Không rõ mã QR: raw='{data_raw}', key='{data_key}'")
            
            time.sleep(0.01)

        except Exception as e:
            logging.error(f"[QR] Lỗi trong luồng QR: {e}")
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

            # --- LOGIC CHỐNG KẸT HÀNG CHỜ ---
            with queue_lock:
                if qr_queue and (now - queue_head_since) > QUEUE_HEAD_TIMEOUT:
                    # Chỉ timeout vật phẩm đầu tiên
                    expected_lane_index = qr_queue.pop(0)
                    expected_lane_name = lanes_config[expected_lane_index]['name']
                    queue_log = [lanes_config[i]['name'] for i in qr_queue]
                    queue_head_since = now if qr_queue else 0.0
                    logging.warning(f"TIMEOUT! Tự động xóa {expected_lane_name} khỏi hàng chờ. Hàng chờ mới: {queue_log}")

            # --- ĐỌC SENSOR TỪNG LANE ---
            for i in range(num_lanes):
                lane = lanes_config[i]
                sensor_pin = lane.get("sensor_pin")
                if sensor_pin is None: continue
                
                lane_name = lane['name']
                push_pin = lane.get("push_pin")

                try:
                    sensor_now = GPIO.input(sensor_pin)
                except Exception as gpio_e:
                    logging.error(f"[SENSOR] Lỗi đọc GPIO pin {sensor_pin} ({lane_name}): {gpio_e}")
                    global main_running
                    main_running = False # Lỗi nghiêm trọng
                    break

                # --- PHÁT HIỆN SƯỜN XUỐNG (1 -> 0) ---
                if sensor_now == 0 and last_s_state[i] == 1:
                    if (now - last_s_trig[i]) > debounce_time:
                        last_s_trig[i] = now

                        with queue_lock:
                            if not qr_queue:
                                # --- 1. HÀNG CHỜ RỖNG (Sensor-First) ---
                                if push_pin is None: # Lane đi thẳng
                                    logging.info(f"Vật đi thẳng (không QR) qua {lane_name}.")
                                    threading.Thread(target=sorting_process, args=(i,), daemon=True).start()
                                else: # Lane đẩy
                                    pending_sensor_triggers[i] = now 
                                    logging.info(f"Sensor {lane_name} kích hoạt (hàng chờ rỗng). Đang chờ QR...")
                            
                            # (*** SỬA LOGIC FIFO ***)
                            # Thay vì 'if i == qr_queue[0]:'
                            elif i in qr_queue:
                                # --- 2. KHỚP (Flexible FIFO) ---
                                # Vật phẩm này có trong hàng chờ!
                                qr_queue.remove(i) # Xóa vật phẩm này (dù nó ở đâu)
                                queue_log = [lanes_config[j]['name'] for j in qr_queue]

                                # Nếu vừa xóa vật phẩm đầu tiên, reset timeout
                                if i == qr_queue[0] if qr_queue else False:
                                    queue_head_since = now if qr_queue else 0.0

                                threading.Thread(target=handle_sorting_with_delay, args=(i,), daemon=True).start()
                                logging.info(f"Sensor {lane_name} khớp (FIFO Linh hoạt). Hàng chờ mới: {queue_log}")
                                pending_sensor_triggers[i] = 0.0 # Xóa cờ chờ (nếu có)

                            else:
                                # --- 3. KHÔNG KHỚP (Pass-over) ---
                                # Vật phẩm lạ (không QR) đi qua sensor
                                logging.warning(f"Sensor {lane_name} kích hoạt, nhưng vật phẩm không có trong hàng chờ. Bỏ qua.")
                        
                last_s_state[i] = sensor_now

            adaptive_sleep = 0.05 if all(s == 1 for s in last_s_state) else 0.01
            time.sleep(adaptive_sleep)

    except Exception as e:
        logging.critical(f"[CRITICAL] Luồng sensor bị crash: {e}")
        main_running = False

# =============================
#         MAIN EXECUTION
# =============================
if __name__ == "__main__":
    try:
        log_format = '%(asctime)s [%(levelname)s] (%(threadName)s) %(message)s'
        logging.basicConfig(level=logging.INFO, format=log_format,
                            handlers=[logging.FileHandler(LOG_FILE, encoding='utf-8'),
                                      logging.StreamHandler()])
        logging.info("--- HỆ THỐNG LOG-ONLY TỐI GIẢN ĐANG KHỞI ĐỘNG ---")

        if not load_config():
            raise RuntimeError("Không thể tải file config.json.")
        
        GPIO.setmode(GPIO.BCM if timing_config.get("gpio_mode", "BCM") == "BCM" else GPIO.BOARD)
        GPIO.setwarnings(False)
        logging.info(f"[GPIO] Cài đặt chân SENSOR: {SENSOR_PINS}")
        for pin in SENSOR_PINS: GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        logging.info(f"[GPIO] Cài đặt chân RELAY: {RELAY_PINS}")
        for pin in RELAY_PINS: GPIO.setup(pin, GPIO.OUT)
        
        reset_relays()

        # Lấy camera index từ config, nếu không có thì dùng 0
        CAM_IDX = qr_config.get("camera_index", 0)

        # Khởi động các luồng
        threading.Thread(target=run_camera, args=(CAM_IDX,), name="CameraThread", daemon=True).start()
        threading.Thread(target=qr_detection_loop, name="QRScannerThread", daemon=True).start()
        threading.Thread(target=sensor_monitoring_thread, name="SensorMonThread", daemon=True).start()

        time.sleep(1) # Chờ các luồng khởi động
        if not main_running:
             raise RuntimeError("Khởi động luồng thất bại (Camera hoặc GPIO).")

        logging.info("="*55 + "\n HỆ THỐNG PHÂN LOẠI SẴN SÀNG (vSimple - Log Only)\n" +
                     f" Logic: FIFO Linh Hoạt (Đã sửa hạn chế)\n" +
                     " (Đang chạy không có Web Server. Nhấn Ctrl+C để dừng)\n" + "="*55)

        while main_running:
            time.sleep(1)

    except KeyboardInterrupt:
        logging.info("\n--- HỆ THỐNG ĐANG TẮT (Ctrl+C) ---")
    except Exception as startup_err:
        logging.critical(f"[CRITICAL] Khởi động hệ thống thất bại: {startup_err}")
    finally:
        main_running = False
        logging.info("Đang dừng các luồng nền...")
        time.sleep(0.5) # Chờ các luồng nhận cờ main_running
        try:
            GPIO.cleanup()
            logging.info("Dọn dẹp GPIO thành công.")
        except Exception as cleanup_err:
            logging.warning(f"Lỗi khi dọn dẹp GPIO: {cleanup_err}")
        logging.info("--- HỆ THỐNG ĐÃ TẮT HOÀN TOÀN ---")