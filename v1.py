
import cv2
import time
import json
import threading
import logging
import sqlite3 
import copy     # Fix #8
import uuid     # Fix #9
from datetime import datetime 

try:
    from ultralytics import YOLO
    YOLO_ENABLED = True
except ImportError:
    YOLO_ENABLED = False
    logging.warning("[AI] Thư viện 'ultralytics' chưa được cài đặt (pip install ultralytics). Tính năng AI sẽ bị tắt.")
import os
import functools
try:
    from pyzbar import pyzbar
    PYZBAR_ENABLED = True
except ImportError:
    PYZBAR_ENABLED = False
    logging.warning("[QR] Thư viện pyzbar chưa được cài đặt (pip install pyzbar). Sẽ chỉ dùng cv2.QRCodeDetector().")
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, Response, jsonify, request
from flask_sock import Sock
import unicodedata, re

# =============================
#        CỜ ĐIỀU KHIỂN HỆ THỐNG
# =============================
hot_reload_enabled = False

# =============================
#       CÁC HÀM TIỆN ÍCH CHUẨN HOÁ
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
#       LỚP TRỪU TƯỢNG GPIO
# =============================
try:
    import RPi.GPIO as RPiGPIO
except (ImportError, RuntimeError):
    RPiGPIO = None

class GPIOProvider:
    def setup(self, pin, mode, pull_up_down=None): raise NotImplementedError
    def output(self, pin, value): raise NotImplementedError
    def input(self, pin): raise NotImplementedError
    def cleanup(self): raise NotImplementedError
    def setmode(self, mode): raise NotImplementedError
    def setwarnings(self, value): raise NotImplementedError

class RealGPIO(GPIOProvider):
    def __init__(self):
        if RPiGPIO is None: raise ImportError("Không thể tải RPi.GPIO.")
        self.gpio = RPiGPIO
        for attr in ['BOARD', 'BCM', 'OUT', 'IN', 'HIGH', 'LOW', 'PUD_UP']:
            setattr(self, attr, getattr(self.gpio, attr))
    def setmode(self, mode): self.gpio.setmode(mode)
    def setwarnings(self, value): self.gpio.setwarnings(value)
    def setup(self, pin, mode, pull_up_down=None):
        if pin is not None:
            if pull_up_down: self.gpio.setup(pin, mode, pull_up_down=pull_up_down)
            else: self.gpio.setup(pin, mode)
    def output(self, pin, value): 
        if pin is not None: self.gpio.output(pin, value)
    def input(self, pin): 
        if pin is not None: return self.gpio.input(pin)
        return self.gpio.HIGH
    def cleanup(self): self.gpio.cleanup()

class MockGPIO(GPIOProvider):
    def __init__(self):
        self.BOARD = "mock_BOARD"; self.BCM = "mock_BCM"; self.OUT = "mock_OUT"
        self.IN = "mock_IN"; self.HIGH = 1; self.LOW = 0
        self.input_pins = set(); self.PUD_UP = "mock_PUD_UP"; self.pin_states = {}
        logging.warning("="*50 + "\nĐANG CHẠY Ở CHẾ ĐỘ GIẢ LẬP (MOCK GPIO).\n" + "="*50)
    def setmode(self, mode): logging.info(f"[MOCK] setmode={mode}")
    def setwarnings(self, value): logging.info(f"[MOCK] setwarnings={value}")
    def setup(self, pin, mode, pull_up_down=None):
        if pin is not None:
            logging.info(f"[MOCK] setup pin {pin} mode={mode} pull_up_down={pull_up_down}")
            if mode == self.OUT: self.pin_states[pin] = self.LOW
            else: self.pin_states[pin] = self.HIGH; self.input_pins.add(pin)
    def output(self, pin, value):
        if pin is not None:
            logging.info(f"[MOCK] output pin {pin}={value}")
            self.pin_states[pin] = value
    def input(self, pin):
        if pin is not None: return self.pin_states.get(pin, self.HIGH)
        return self.HIGH
    def set_input_state(self, pin, logical_state):
        if pin not in self.input_pins: self.input_pins.add(pin)
        state = self.HIGH if logical_state else self.LOW
        self.pin_states[pin] = state
        logging.info(f"[MOCK] set_input_state pin {pin} -> {state}")
        return state
    def toggle_input_state(self, pin):
        if pin not in self.input_pins: self.input_pins.add(pin)
        new_state = self.LOW if self.input(pin) == self.HIGH else self.HIGH
        self.pin_states[pin] = new_state
        logging.info(f"[MOCK] toggle_input_state pin {pin} -> {new_state}")
        return 0 if new_state == self.LOW else 1
    def cleanup(self): logging.info("[MOCK] cleanup GPIO")

def get_gpio_provider():
    if RPiGPIO: return RealGPIO()
    return MockGPIO()

# =============================
#   QUẢN LÝ LỖI (Error Manager)
# =============================
class ErrorManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.maintenance_mode = False
        self.last_error = None
    def trigger_maintenance(self, message):
        with self.lock:
            if self.maintenance_mode: return
            self.maintenance_mode = True
            self.last_error = message
            logging.critical("="*50 + f"\n[MAINTENANCE MODE] Lỗi nghiêm trọng: {message}\n" + "="*50)
            broadcast_log({"log_type": "error", "message": f"MAINTENANCE MODE: {message}"})
    def reset(self):
        with self.lock:
            self.maintenance_mode = False
            self.last_error = None
            logging.info("[MAINTENANCE MODE] Đã reset chế độ bảo trì.")
            with state_lock:
                for lane in system_state["lanes"]:
                    lane["status"] = "Sẵn sàng"
    def is_maintenance(self):
        return self.maintenance_mode

# =============================
#       CẤU HÌNH CHUNG
# =============================
CAMERA_INDEX = 0
CONFIG_FILE = 'config.json'
LOG_FILE = 'system.log'
DATABASE_FILE = 'sort_log.db'
QUEUE_STATE_FILE = 'queue_state.json' # Fix #7
ACTIVE_LOW = True
AUTH_ENABLED = os.environ.get("APP_AUTH_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
USERNAME = os.environ.get("APP_USERNAME", "admin")
PASSWORD = os.environ.get("APP_PASSWORD", "123")
# (SỬA) Vô hiệu hóa Sensor Gác Cổng
# SENSOR_ENTRY_PIN = 6
# SENSOR_ENTRY_MOCK_PIN = 99
SENSOR_ENTRY_PIN = None
SENSOR_ENTRY_MOCK_PIN = None

# =============================
#     KHỞI TẠO CÁC ĐỐI TƯỢNG
# =============================
GPIO = get_gpio_provider()
error_manager = ErrorManager()
executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="TestWorker")
database_lock = threading.Lock()
config_file_lock = threading.Lock()

# =============================
#       KHAI BÁO CHÂN GPIO
# =============================
DEFAULT_LANES_CONFIG = [
    {"id": "SP001", "name": "Phân loại A", "sensor_pin": 3, "push_pin": 17, "pull_pin": 18},
    {"id": "SP002", "name": "Phân loại B", "sensor_pin": 23, "push_pin": 27, "pull_pin": 14},
    {"id": "SP003", "name": "Phân loại C", "sensor_pin": 24, "push_pin": 22, "pull_pin": 4},
    {"id": "NG", "name": "Sản Phẩm NG(Bỏ)", "sensor_pin": None, "push_pin": None, "pull_pin": None},
]
lanes_config = DEFAULT_LANES_CONFIG
RELAY_PINS = []
SENSOR_PINS = []
RELAY_CONVEYOR_PIN = 7

# =============================
#     HÀM ĐIỀU KHIỂN RELAY
# =============================
def RELAY_ON(pin):
    if pin is not None:
        try: GPIO.output(pin, GPIO.LOW if ACTIVE_LOW else GPIO.HIGH)
        except Exception as e:
            logging.error(f"[GPIO] Lỗi RELAY_ON pin {pin}: {e}")
            error_manager.trigger_maintenance(f"Lỗi GPIO pin {pin}: {e}")
def RELAY_OFF(pin):
    if pin is not None:
        try: GPIO.output(pin, GPIO.HIGH if ACTIVE_LOW else GPIO.LOW)
        except Exception as e:
            logging.error(f"[GPIO] Lỗi RELAY_OFF pin {pin}: {e}")
            error_manager.trigger_maintenance(f"Lỗi GPIO pin {pin}: {e}")

def CONVEYOR_RUN():
    logging.info("[CONVEYOR] Băng chuyền: RUN")
    RELAY_ON(RELAY_CONVEYOR_PIN)

def CONVEYOR_STOP():
    logging.info("[CONVEYOR] Băng chuyền: STOP")
    RELAY_OFF(RELAY_CONVEYOR_PIN)

# =============================
#       TRẠNG THÁI HỆ THỐNG
# =============================
system_state = {
    "lanes": [],
    "timing_config": {
        "cycle_delay": 0.3, "settle_delay": 0.2, "sensor_debounce": 0.1,
        "push_delay": 0.0, "gpio_mode": "BCM",
        "queue_head_timeout": 15.0,
        "pending_trigger_timeout": 0.5,
        "RELAY_CONVEYOR_PIN": None,
        "stop_conveyor_on_entry": False,
        "stability_delay": 0.25,
        "stop_conveyor_on_qr": False,      # <-- (MỚI) Thêm StopOnQR
        "conveyor_stop_delay_qr": 2.0      # <-- (MỚI) Thêm StopOnQR
    },
    "is_mock": isinstance(GPIO, MockGPIO), "maintenance_mode": False,
    "auth_enabled": AUTH_ENABLED, "gpio_mode": "BCM", "last_error": None,
    "queue_indices": [],
    # (SỬA) Vô hiệu hóa
    # "sensor_entry_reading": 1,
    "entry_queue_size": 0,
    "ai_config": {},
    "camera_settings": {}
}

state_lock = threading.Lock()
main_loop_running = True
latest_frame = None
frame_lock = threading.Lock()
fps_value = 0.0

AI_MODEL = None
AI_ENABLED = False
AI_LANE_MAP = {}
AI_MIN_CONFIDENCE = 0.6

qr_queue = [] 
qr_queue_lock = threading.Lock()

processing_queue = [] 
processing_queue_lock = threading.Lock()

QUEUE_HEAD_TIMEOUT = 15.0
queue_head_since = 0.0

last_sensor_state = []
last_sensor_trigger_time = []
AUTO_TEST_ENABLED = False
auto_test_last_state = []
auto_test_last_trigger = []

# (SỬA) Vô hiệu hóa
# last_entry_sensor_state = 1
# last_entry_sensor_trigger_time = 0.0

# =============================
# KHỞI TẠO CƠ SỞ DỮ LIỆU
# =============================
def init_database():
    with database_lock:
        try:
            conn = sqlite3.connect(DATABASE_FILE)
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS sort_log (
                date TEXT,
                lane_name TEXT,
                count INTEGER DEFAULT 0,
                PRIMARY KEY (date, lane_name)
            )
            """)
            conn.commit()
            conn.close()
            logging.info(f"[DB] Đã khởi tạo CSDL SQLite tại '{DATABASE_FILE}' thành công.")
        except Exception as e:
            logging.critical(f"[CRITICAL] Không thể khởi tạo CSDL SQLite: {e}")
            error_manager.trigger_maintenance(f"Lỗi khởi tạo CSDL SQLite: {e}")

# =============================
#     (MỚI) FIX #7: LƯU/TẢI HÀNG CHỜ
# =============================
def save_queues_on_shutdown():
    logging.info("[SHUTDOWN] Đang lưu trạng thái hàng chờ...")
    try:
        queue_data = {}
        with qr_queue_lock:
            queue_data['qr_queue'] = list(qr_queue)
        with processing_queue_lock:
            queue_data['processing_queue'] = list(processing_queue)
        
        if not queue_data['qr_queue'] and not queue_data['processing_queue']:
            logging.info("[SHUTDOWN] Hàng chờ trống, không cần lưu.")
            if os.path.exists(QUEUE_STATE_FILE):
                os.remove(QUEUE_STATE_FILE)
            return

        with open(QUEUE_STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(queue_data, f)
        logging.info(f"[SHUTDOWN] Đã lưu {len(queue_data['qr_queue'])} QR, {len(queue_data['processing_queue'])} Job vào {QUEUE_STATE_FILE}")
    except Exception as e:
        logging.error(f"[SHUTDOWN] Lỗi không thể lưu hàng chờ: {e}")

def load_queues_on_startup():
    global qr_queue, processing_queue, queue_head_since
    if os.path.exists(QUEUE_STATE_FILE):
        logging.warning(f"[STARTUP] Phát hiện file {QUEUE_STATE_FILE}. Đang khôi phục hàng chờ...")
        try:
            with open(QUEUE_STATE_FILE, 'r', encoding='utf-8') as f:
                queue_data = json.load(f)
            
            with qr_queue_lock:
                qr_queue = queue_data.get('qr_queue', [])
            with processing_queue_lock:
                processing_queue = queue_data.get('processing_queue', [])
                if processing_queue:
                    queue_head_since = time.time() # Reset timeout
            
            logging.info(f"[STARTUP] Đã khôi phục {len(qr_queue)} QR, {len(processing_queue)} Job.")
            
            with state_lock:
                    system_state["queue_indices"] = [j["lane_index"] for j in processing_queue]
                    system_state["entry_queue_size"] = len(processing_queue)
                    for job in processing_queue:
                        lane_idx = job.get('lane_index')
                        if 0 <= lane_idx < len(system_state['lanes']):
                             system_state['lanes'][lane_idx]['status'] = "Đang chờ vật (Tải lại)"

        except Exception as e:
            logging.error(f"[STARTUP] Lỗi khôi phục hàng chờ: {e}. Bắt đầu với hàng chờ trống.")
            qr_queue = []; processing_queue = []
        
        try:
            os.remove(QUEUE_STATE_FILE)
            logging.info(f"[STARTUP] Đã xử lý và xóa file {QUEUE_STATE_FILE}.")
        except Exception as e:
            logging.error(f"[STARTUP] Lỗi xóa file {QUEUE_STATE_FILE}: {e}")
    else:
        logging.info("[STARTUP] Không có file trạng thái hàng chờ. Bắt đầu mới.")

# =============================
#     HÀM KHỞI ĐỘNG & CONFIG
# =============================
def load_local_config():
    global lanes_config, RELAY_PINS, SENSOR_PINS, last_sensor_state, last_sensor_trigger_time
    global auto_test_last_state, auto_test_last_trigger
    global QUEUE_HEAD_TIMEOUT, RELAY_CONVEYOR_PIN
    global AI_MODEL, AI_ENABLED, AI_LANE_MAP, AI_MIN_CONFIDENCE

    default_ai_config = {
        "enable_ai": False,
        "ai_priority": False,
        "model_path": "best.pt",
        "min_confidence": 0.6,
        "ai_class_to_id_map": {
            "APPLE": "SP001",
            "ORANGE": "SP002"
        }
    }
    # (MỚI) Gộp config StopOnQR vào đây
    default_timing_config = {
        "cycle_delay": 0.3, "settle_delay": 0.2, "sensor_debounce": 0.1,
        "push_delay": 0.0, "gpio_mode": "BCM",
        "queue_head_timeout": 15.0, "pending_trigger_timeout": 0.5,
        "RELAY_CONVEYOR_PIN": None,
        "stop_conveyor_on_entry": False,
        "stability_delay": 0.25,
        "stop_conveyor_on_qr": False,
        "conveyor_stop_delay_qr": 2.0
    }
    default_camera_settings = {
        "auto_exposure": True,
        "brightness": 128,
        "contrast": 32
    }
    default_config_full = {
        "timing_config": default_timing_config,
        "lanes_config": DEFAULT_LANES_CONFIG,
        "ai_config": default_ai_config,
        "camera_settings": default_camera_settings
    }    
    loaded_config = default_config_full
    
    with config_file_lock:
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f: file_content = f.read()
                if not file_content:
                    logging.warning("[CONFIG] File config rỗng, dùng mặc định.")
                else:
                    loaded_config_from_file = json.loads(file_content)
                    
                    timing_cfg = default_timing_config.copy()
                    timing_cfg.update(loaded_config_from_file.get('timing_config', {}))
                    loaded_config['timing_config'] = timing_cfg
                    
                    ai_cfg = default_ai_config.copy()
                    ai_cfg.update(loaded_config_from_file.get('ai_config', {}))
                    loaded_config['ai_config'] = ai_cfg

                    cam_cfg = default_camera_settings.copy()
                    cam_cfg.update(loaded_config_from_file.get('camera_settings', {}))
                    loaded_config['camera_settings'] = cam_cfg

                    lanes_from_file = loaded_config_from_file.get('lanes_config', DEFAULT_LANES_CONFIG)
                    loaded_config['lanes_config'] = ensure_lane_ids(lanes_from_file)
            
            except Exception as e:
                logging.error(f"[CONFIG] Lỗi đọc/parse file config ({e}), dùng mặc định.")
                error_manager.trigger_maintenance(f"Lỗi JSON file config.json: {e}")
                loaded_config = default_config_full
        else:
            logging.warning("[CONFIG] Không có file config, dùng mặc định và tạo mới.")
            try:
                with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                    json.dump(loaded_config, f, indent=4)
            except Exception as e:
                logging.error(f"[CONFIG] Không thể tạo file config mới: {e}")

    lanes_config = loaded_config['lanes_config']
    num_lanes = len(lanes_config)
    new_system_lanes = []
    RELAY_PINS = []; SENSOR_PINS = []
    
    # (SỬA) Vô hiệu hóa
    # if SENSOR_ENTRY_PIN: SENSOR_PINS.append(SENSOR_ENTRY_PIN)
    # if isinstance(GPIO, MockGPIO) and SENSOR_ENTRY_MOCK_PIN:
    #     SENSOR_PINS.append(SENSOR_ENTRY_MOCK_PIN)
        
    RELAY_CONVEYOR_PIN = loaded_config['timing_config'].get('RELAY_CONVEYOR_PIN')
    if RELAY_CONVEYOR_PIN:
        RELAY_PINS.append(RELAY_CONVEYOR_PIN)
        logging.info(f"[CONFIG] Đã cấu hình Relay Băng chuyền tại pin: {RELAY_CONVEYOR_PIN}")

    for i, lane_cfg in enumerate(lanes_config):
        lane_name = lane_cfg.get("name", f"Lane {i+1}"); lane_id = lane_cfg.get("id", f"LANE_{i+1}")
        new_system_lanes.append({
            "name": lane_name, "id": lane_id, "status": "Sẵn sàng", "count": 0,
            "sensor_pin": lane_cfg.get("sensor_pin"), "push_pin": lane_cfg.get("push_pin"),
            "pull_pin": lane_cfg.get("pull_pin"), "sensor_reading": 1,
            "relay_grab": 0, "relay_push": 0
        })
        if lane_cfg.get("sensor_pin") is not None: SENSOR_PINS.append(lane_cfg["sensor_pin"])
        if lane_cfg.get("push_pin") is not None: RELAY_PINS.append(lane_cfg["push_pin"])
        if lane_cfg.get("pull_pin") is not None: RELAY_PINS.append(lane_cfg["pull_pin"])

    last_sensor_state = [1] * num_lanes; last_sensor_trigger_time = [0.0] * num_lanes
    auto_test_last_state = [1] * num_lanes; auto_test_last_trigger = [0.0] * num_lanes

    with state_lock:
        system_state['timing_config'] = loaded_config['timing_config']
        system_state['gpio_mode'] = loaded_config['timing_config'].get("gpio_mode", "BCM")
        system_state['lanes'] = new_system_lanes
        system_state['auth_enabled'] = AUTH_ENABLED
        system_state['is_mock'] = isinstance(GPIO, MockGPIO)
        # (SỬA) Vô hiệu hóa
        # system_state['sensor_entry_reading'] = 1
        system_state['entry_queue_size'] = 0
        system_state['ai_config'] = loaded_config['ai_config']
        system_state['camera_settings'] = loaded_config['camera_settings']
    
    QUEUE_HEAD_TIMEOUT = loaded_config['timing_config'].get('queue_head_timeout', 15.0)

    AI_MIN_CONFIDENCE = loaded_config['ai_config'].get('min_confidence', 0.6)
    
    if loaded_config['ai_config'].get('enable_ai', False):
        if not YOLO_ENABLED:
            logging.error("[AI] Config bật AI, nhưng thư viện 'ultralytics' chưa được cài đặt. AI đã bị tắt.")
        else:
            model_path = loaded_config['ai_config'].get('model_path', 'best.pt')
            if not os.path.exists(model_path):
                logging.error(f"[AI] Lỗi: Không tìm thấy file model tại '{model_path}'. AI đã bị tắt.")
            else:
                try:
                    AI_MODEL = YOLO(model_path)
                    AI_ENABLED = True
                    logging.info(f"[AI] Đã tải thành công model từ '{model_path}'.")
                    
                    lane_id_to_index_map = {canon_id(lane['id']): i for i, lane in enumerate(lanes_config) if lane.get('id')}
                    
                    ai_class_map_config = loaded_config['ai_config'].get('ai_class_to_id_map', {})
                    for class_name, lane_id in ai_class_map_config.items():
                        canon_lane_id = canon_id(lane_id)
                        if canon_lane_id in lane_id_to_index_map:
                            lane_index = lane_id_to_index_map[canon_lane_id]
                            AI_LANE_MAP[class_name.upper()] = lane_index
                            logging.info(f"[AI] Đã map Class '{class_name.upper()}' -> Lane ID '{lane_id}' (index {lane_index})")
                        else:
                            logging.warning(f"[AI] Lỗi map: Lane ID '{lane_id}' (cho class '{class_name}') không tồn tại trong lanes_config.")
                    
                except Exception as e:
                    logging.error(f"[AI] Lỗi nghiêm trọng khi tải model YOLO: {e}. AI đã bị tắt.", exc_info=True)
                    AI_MODEL = None
                    AI_ENABLED = False
    
    if not AI_ENABLED:
        logging.warning("[AI] Tính năng AI hiện đang TẮT (do config hoặc lỗi).")
    logging.info(f"[CONFIG] Loaded {num_lanes} lanes config.")
    logging.info(f"[CONFIG] Queue Timeout: {QUEUE_HEAD_TIMEOUT}s")
    # (SỬA) Vô hiệu hóa
    # logging.info(f"[CONFIG] Sensor Entry Pin (Real/Mock): {SENSOR_ENTRY_PIN} / {SENSOR_ENTRY_MOCK_PIN}")

def ensure_lane_ids(lanes_list):
    default_ids = ['SP001', 'SP002', 'SP003', 'SP004', 'SP005', 'SP006', 'SP007', 'SP008', 'SP009', 'SP010']
    for i, lane in enumerate(lanes_list):
        if 'id' not in lane or not lane['id']:
            if i < len(default_ids): lane['id'] = default_ids[i]
            else: lane['id'] = f"LANE_{i+1}"
            logging.warning(f"[CONFIG] Lane {i+1} thiếu ID. Đã gán ID: {lane['id']}")
    return lanes_list

def reset_all_relays_to_default():
    logging.info("[GPIO] Reset tất cả relay về trạng thái mặc định (THU BẬT, BĂNG CHUYỀN CHẠY).")
    with state_lock:
        for lane in system_state["lanes"]:
            pull_pin = lane.get("pull_pin")
            push_pin = lane.get("push_pin")
            if pull_pin is not None: RELAY_ON(pull_pin)
            if push_pin is not None: RELAY_OFF(push_pin)
            lane["relay_grab"] = 1 if pull_pin is not None else 0
            lane["relay_push"] = 0
            lane["status"] = "Sẵn sàng"
    CONVEYOR_RUN() 
    time.sleep(0.1)
    logging.info("[GPIO] Reset hoàn tất.")

def periodic_config_save():
    while main_loop_running:
        time.sleep(60)
        if error_manager.is_maintenance(): continue
        
        config_to_save = {}
        
        try:
            with state_lock:
                config_to_save['timing_config'] = system_state['timing_config'].copy()
                config_to_save['ai_config'] = system_state['ai_config'].copy()
                config_to_save['camera_settings'] = system_state['camera_settings'].copy()
                
                current_lanes_config = []
                for lane_state in system_state['lanes']:
                    current_lanes_config.append({
                        "id": lane_state['id'], "name": lane_state['name'],
                        "sensor_pin": lane_state.get('sensor_pin'), 
                        "push_pin": lane_state.get('push_pin'), 
                        "pull_pin": lane_state.get('pull_pin')
                    })
                config_to_save['lanes_config'] = current_lanes_config
            
            with config_file_lock:
                with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                    json.dump(config_to_save, f, indent=4)
            logging.info("[CONFIG] Đã tự động lưu config (timing, ai, lanes, camera).")

        except Exception as e:
            logging.error(f"[CONFIG] Lỗi tự động lưu config: {e}") 

# =============================
#       LUỒNG CAMERA (ĐÃ SỬA LAG)
# =============================
def camera_capture_thread():
    global latest_frame, fps_value

    frame_count = 0
    start_time = time.time()
    
    cam_settings = {}
    with state_lock:
        cam_settings = system_state.get('camera_settings', {})
    
    camera = cv2.VideoCapture(CAMERA_INDEX)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640); 
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
    
    try:
        auto_exposure_cfg = cam_settings.get('auto_exposure', True)
        auto_exposure_val = 1 if auto_exposure_cfg else 0
        camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_exposure_val)
        logging.info(f"[CAMERA] Đã đặt Auto Exposure: {'BẬT' if auto_exposure_val == 1 else 'TẮT'}.")

        if auto_exposure_val == 0:
            brightness_val = int(cam_settings.get('brightness', 128))
            contrast_val = int(cam_settings.get('contrast', 32))
            
            camera.set(cv2.CAP_PROP_BRIGHTNESS, brightness_val)
            camera.set(cv2.CAP_PROP_CONTRAST, contrast_val)
            
            logging.info(f"[CAMERA] Đã đặt Brightness thủ công: {brightness_val}")
            logging.info(f"[CAMERA] Đã đặt Contrast thủ công: {contrast_val}")
    except Exception as cam_e:
        logging.error(f"[CAMERA] Lỗi khi cài đặt thông số camera: {cam_e}")
    
    if not camera.isOpened():
        logging.error("[ERROR] Không mở được camera.")
        error_manager.trigger_maintenance("Không thể mở camera.")
        return
    
    logging.info("[CAMERA] Camera đã khởi động.")
    
    retries = 0; 
    max_retries = 5
    while main_loop_running:
        if error_manager.is_maintenance():
            time.sleep(0.5); continue
            
        ret, frame = camera.read()
        
        if not ret:
            retries += 1
            logging.warning(f"[WARN] Mất camera (lần {retries}/{max_retries}), thử khởi động lại...")
            broadcast_log({"log_type":"error","message":f"Mất camera (lần {retries}), đang thử lại..."})
            if retries > max_retries:
                logging.critical("[ERROR] Camera lỗi vĩnh viễn. Chuyển sang chế độ bảo trì.")
                error_manager.trigger_maintenance("Camera lỗi vĩnh viễn (mất kết nối).")
                break
            camera.release(); time.sleep(1); camera = cv2.VideoCapture(CAMERA_INDEX)
            continue
        retries = 0

        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        if elapsed_time >= 1.0:
            fps_value = frame_count / elapsed_time
            frame_count = 0
            start_time = current_time
        
        with frame_lock:
            latest_frame = frame.copy()
            
        time.sleep(1 / 60) # Cung cấp 60 FPS
        
    camera.release()

# =============================
#     LƯU LOG ĐẾM SẢN PHẨM (DÙNG SQLITE)
# =============================
def log_sort_count(lane_index, lane_name):
    with database_lock:
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            conn = sqlite3.connect(DATABASE_FILE)
            cursor = conn.cursor()
            
            cursor.execute("""
            INSERT INTO sort_log (date, lane_name, count)
            VALUES (?, ?, 1)
            ON CONFLICT(date, lane_name) DO UPDATE SET
                count = count + 1
            """, (today, lane_name))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logging.error(f"[DB] Lỗi khi ghi log đếm vào SQLite: {e}")

# =============================
#     CHU TRÌNH PHÂN LOẠI (FIX #9)
# =============================
def sorting_process(lane_index, job_id="N/A"): # Thêm job_id
    job_id_log_prefix = f"[JobID {job_id}]"
    
    lane_name = ""; push_pin, pull_pin = None, None
    is_sorting_lane = False
    try:
        with state_lock:
            if not (0 <= lane_index < len(system_state["lanes"])):
                logging.error(f"[SORT] {job_id_log_prefix} Lane index {lane_index} không hợp lệ.")
                return
            cfg = system_state['timing_config']
            delay = cfg['cycle_delay']; settle_delay = cfg['settle_delay']
            lane = system_state["lanes"][lane_index]
            lane_name = lane["name"]; push_pin = lane.get("push_pin"); pull_pin = lane.get("pull_pin")
            is_sorting_lane = not (push_pin is None and pull_pin is None)
            if is_sorting_lane and (push_pin is None or pull_pin is None):
                logging.error(f"[SORT] {job_id_log_prefix} Lane {lane_name} (index {lane_index}) chưa được cấu hình đủ chân relay.")
                lane["status"] = "Lỗi Config"
                broadcast_log({"log_type": "error", "message": f"{job_id_log_prefix} Lane {lane_name} thiếu cấu hình chân relay."})
                return
            lane["status"] = "Đang phân loại..." if is_sorting_lane else "Đang đi thẳng..."
        
        if not is_sorting_lane:
            broadcast_log({"log_type": "info", "message": f"{job_id_log_prefix} Vật phẩm đi thẳng qua {lane_name}"})
        if is_sorting_lane:
            broadcast_log({"log_type": "info", "message": f"{job_id_log_prefix} Bắt đầu chu trình đẩy {lane_name}"})
            RELAY_OFF(pull_pin)
            with state_lock: system_state["lanes"][lane_index]["relay_grab"] = 0
            time.sleep(settle_delay);
            if not main_loop_running: return
            RELAY_ON(push_pin)
            with state_lock: system_state["lanes"][lane_index]["relay_push"] = 1
            time.sleep(delay);
            if not main_loop_running: return
            RELAY_OFF(push_pin)
            with state_lock: system_state["lanes"][lane_index]["relay_push"] = 0
            time.sleep(settle_delay);
            if not main_loop_running: return
            RELAY_ON(pull_pin)
            with state_lock: system_state["lanes"][lane_index]["relay_grab"] = 1

    except Exception as e:
        logging.error(f"[SORT] {job_id_log_prefix} Lỗi trong sorting_process (lane {lane_name}): {e}")
        error_manager.trigger_maintenance(f"Lỗi sorting_process (Lane {lane_name}): {e}")
    finally:
        with state_lock:
            if 0 <= lane_index < len(system_state["lanes"]):
                lane = system_state["lanes"][lane_index]
                if lane_name and lane["status"] != "Lỗi Config":
                    lane["count"] += 1
                    log_type = "sort" if is_sorting_lane else "pass"
                    broadcast_log({"log_type": log_type, "name": lane_name, "count": lane['count']})
                    log_sort_count(lane_index, lane_name)
                    if lane["status"] != "Lỗi Config":
                        lane["status"] = "Sẵn sàng"
        if lane_name:
            msg = f"Hoàn tất chu trình cho {lane_name}" if is_sorting_lane else f"Hoàn tất đếm vật phẩm đi thẳng qua {lane_name}"
            broadcast_log({"log_type": "info", "message": f"{job_id_log_prefix} {msg}"})
        
        stop_conveyor = False
        with state_lock:
            stop_conveyor = system_state['timing_config'].get('stop_conveyor_on_entry', False)
        
        if stop_conveyor:
            qr_count = 0
            entry_count = 0
            with qr_queue_lock:
                qr_count = len(qr_queue)
            with processing_queue_lock:
                entry_count = len(processing_queue)
                
            if qr_count == 0 and entry_count == 0:
                 logging.info(f"[CONVEYOR] {job_id_log_prefix} Hoàn tất xử lý, không còn vật. Khởi động lại băng chuyền.")
                 CONVEYOR_RUN()
            else:
                 logging.info(f"[CONVEYOR] {job_id_log_prefix} Hoàn tất xử lý. Băng chuyền VẪN DỪNG (còn {qr_count} QR, {entry_count} vật).")

# =============================
# CÁC HÀM TEST RELAY
# =============================
test_seq_running = False
test_seq_lock = threading.Lock()

def _run_test_relay(lane_index, relay_action):
    push_pin, pull_pin, lane_name = None, None, f"Lane {lane_index + 1}"
    try:
        with state_lock:
            if not (0 <= lane_index < len(system_state["lanes"])):
                return broadcast_log({"log_type": "error", "message": f"Test thất bại: Lane index {lane_index} không hợp lệ."})
            lane_state = system_state["lanes"][lane_index]
            lane_name = lane_state['name']
            push_pin = lane_state.get("push_pin"); pull_pin = lane_state.get("pull_pin")
            if push_pin is None and pull_pin is None:
                return broadcast_log({"log_type": "warn", "message": f"Lane '{lane_name}' là lane đi thẳng, không có relay."})
            if (push_pin is None or pull_pin is None):
                 return broadcast_log({"log_type": "error", "message": f"Test thất bại: Lane '{lane_name}' thiếu pin PUSH hoặc PULL."})

        if relay_action == "push":
            broadcast_log({"log_type": "info", "message": f"Test: Kích hoạt ĐẨY (PUSH) cho '{lane_name}'."})
            RELAY_OFF(pull_pin); RELAY_ON(push_pin)
            with state_lock:
                if 0 <= lane_index < len(system_state["lanes"]):
                    system_state["lanes"][lane_index]["relay_grab"] = 0
                    system_state["lanes"][lane_index]["relay_push"] = 1
        
        elif relay_action == "grab":
            broadcast_log({"log_type": "info", "message": f"Test: Kích hoạt THU (PULL/GRAB) cho '{lane_name}'."})
            RELAY_OFF(push_pin); RELAY_ON(pull_pin)
            with state_lock:
                if 0 <= lane_index < len(system_state["lanes"]):
                    system_state["lanes"][lane_index]["relay_grab"] = 1
                    system_state["lanes"][lane_index]["relay_push"] = 0
    except Exception as e:
        logging.error(f"[TEST] Lỗi test relay '{relay_action}' cho '{lane_name}': {e}", exc_info=True)
        broadcast_log({"log_type": "error", "message": f"Lỗi test '{relay_action}' trên '{lane_name}': {e}"})
        reset_all_relays_to_default()

def _run_test_all_relays():
    global test_seq_running
    with test_seq_lock:
        if test_seq_running:
            return broadcast_log({"log_type": "warn", "message": "Test tuần tự đang chạy."})
        test_seq_running = True

    logging.info("[TEST] Bắt đầu test tuần tự (Cycle) relay...")
    broadcast_log({"log_type": "info", "message": "Bắt đầu test tuần tự (Cycle) relay..."})
    stopped_early = False

    try:
        num_lanes = 0
        cycle_delay, settle_delay = 0.3, 0.2
        with state_lock:
            num_lanes = len(system_state['lanes'])
            cfg = system_state['timing_config']
            cycle_delay = cfg.get('cycle_delay', 0.3)
            settle_delay = cfg.get('settle_delay', 0.2)

        for i in range(num_lanes):
            with test_seq_lock: stop_requested = not main_loop_running or not test_seq_running
            if stop_requested: stopped_early = True; break

            lane_name, push_pin, pull_pin = f"Lane {i+1}", None, None
            with state_lock:
                if 0 <= i < len(system_state['lanes']):
                    lane_state = system_state['lanes'][i]
                    lane_name = lane_state['name']
                    push_pin = lane_state.get("push_pin"); pull_pin = lane_state.get("pull_pin")
            
            if push_pin is None or pull_pin is None:
                broadcast_log({"log_type": "info", "message": f"Bỏ qua '{lane_name}' (lane đi thẳng)."})
                continue

            broadcast_log({"log_type": "info", "message": f"Testing Cycle cho '{lane_name}'..."})
            
            RELAY_OFF(pull_pin);
            with state_lock: system_state["lanes"][i]["relay_grab"] = 0
            time.sleep(settle_delay)
            if not main_loop_running or not test_seq_running: stopped_early = True; break

            RELAY_ON(push_pin);
            with state_lock: system_state["lanes"][i]["relay_push"] = 1
            time.sleep(cycle_delay)
            if not main_loop_running or not test_seq_running: stopped_early = True; break

            RELAY_OFF(push_pin);
            with state_lock: system_state["lanes"][i]["relay_push"] = 0
            time.sleep(settle_delay)
            if not main_loop_running or not test_seq_running: stopped_early = True; break

            RELAY_ON(pull_pin)
            with state_lock: system_state["lanes"][i]["relay_grab"] = 1
            
            time.sleep(0.5)

        if stopped_early: broadcast_log({"log_type": "warn", "message": "Test tuần tự đã dừng."})
        else: broadcast_log({"log_type": "info", "message": "Test tuần tự hoàn tất."})
    finally:
        with test_seq_lock: test_seq_running = False
        reset_all_relays_to_default()

# =============================
#       QUÉT MÃ QR (GỘP TÍNH NĂNG)
# =============================
def qr_detection_loop():
    cv2_detector = cv2.QRCodeDetector()
    last_qr, last_time = "", 0.0
    data = None 
    
    if PYZBAR_ENABLED:
        logging.info("[QR] Thread QR Detection started (Ưu tiên Pyzbar, fallback CV2).")
    else:
        logging.info("[QR] Thread QR Detection started (Chỉ dùng CV2).")

    while main_loop_running:
        try:
            if AUTO_TEST_ENABLED or error_manager.is_maintenance():
                time.sleep(0.2); continue
            
            # (MỚI) Đọc config dừng băng chuyền
            LANE_MAP = {}
            stop_on_qr = False
            stop_delay_qr = 2.0
            # (MỚI) Lấy NG_LANE_INDEX và config AI (cho logic trigger mới)
            NG_LANE_INDEX = -1
            NG_LANE_NAME = "Hàng NG"
            ai_cfg = {}
            
            with state_lock:
                LANE_MAP = {canon_id(lane.get("id")): idx 
                            for idx, lane in enumerate(system_state["lanes"]) if lane.get("id")}
                # Đọc config
                cfg_timing = system_state.get('timing_config', {})
                stop_on_qr = cfg_timing.get('stop_conveyor_on_qr', False)
                stop_delay_qr = cfg_timing.get('conveyor_stop_delay_qr', 2.0)
                
                # (MỚI) Lấy NG_LANE_INDEX và config AI
                ai_cfg = system_state.get('ai_config', {})
                for i_ng, lane_ng in enumerate(system_state["lanes"]):
                    if canon_id(lane_ng.get("id")) == "NG":
                        NG_LANE_INDEX = i_ng
                        NG_LANE_NAME = lane_ng.get("name", "Hàng NG")
                        break

            if not LANE_MAP: # Chưa load config xong
                time.sleep(0.5)
                continue

            frame_copy = None
            with frame_lock:
                if latest_frame is not None: frame_copy = latest_frame.copy()
            if frame_copy is None:
                time.sleep(0.1); continue

            gray_frame = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
            if gray_frame.mean() < 10:
                time.sleep(0.1); continue

            data = None 
            qr_source = None # Log chi tiết
            
            if PYZBAR_ENABLED:
                try:
                    decoded_objects = pyzbar.decode(gray_frame)
                    if decoded_objects:
                        raw_bytes = decoded_objects[0].data
                        try:
                            data = raw_bytes.decode('utf-8', errors='ignore').strip().strip('\x00')
                            if not data: 
                                data = None
                            else:
                                qr_source = "Pyzbar" 
                        except Exception as e:
                            logging.warning(f"[QR] Lỗi giải mã dữ liệu QR: {e}")
                            data = None
                except Exception as pz_e:
                    logging.warning(f"[QR] Lỗi pyzbar.decode: {pz_e}")

            if data is None: 
                try:
                    data_cv2, _, _ = cv2_detector.detectAndDecode(gray_frame)
                    if data_cv2:
                        data = data_cv2.strip().strip('\x00') 
                        if not data:
                            data = None
                        else:
                            qr_source = "CV2" 
                except cv2.error:
                    data = None; time.sleep(0.1); continue
            
            if data and (data != last_qr or time.time() - last_time > 3.0):
                last_qr, last_time = data, time.time()
                data_key = canon_id(data); data_raw = data; now = time.time()

                if data_key in LANE_MAP:
                    idx = LANE_MAP[data_key]
                    current_queue_for_log = []
                    
                    # (SỬA) LOGIC MỚI: Bỏ Sensor Gác Cổng. Kích hoạt Job ngay khi quét QR.
                    # Logic này được chuyển từ 'entry_sensor_monitoring_thread' sang đây.
                    
                    ai_is_on = ai_cfg.get('enable_ai', False) and AI_ENABLED
                    ai_has_priority = ai_cfg.get('ai_priority', False)
                    
                    job_lane_index = NG_LANE_INDEX
                    job_lane_name = NG_LANE_NAME
                    job_status = "PENDING"
                    
                    qr_lane_index = idx # Đây là lane từ QR
                    
                    ai_lane_index = NG_LANE_INDEX
                    ai_class_name = None
                    if ai_is_on:
                        ai_lane_index, ai_class_name = run_ai_detection(NG_LANE_INDEX)

                    if ai_has_priority and ai_is_on:
                        if ai_lane_index != NG_LANE_INDEX:
                            job_lane_index = ai_lane_index
                            job_status = f"AI_MATCHED ({ai_class_name})"
                        elif qr_lane_index is not None:
                            job_lane_index = qr_lane_index
                            job_status = "QR_MATCHED (AI_Fallback)"
                        else:
                            job_status = "ALL_FAILED" # Sẽ không xảy ra vì qr_lane_index luôn có
                    else:
                        if qr_lane_index is not None:
                            job_lane_index = qr_lane_index
                            job_status = "QR_MATCHED"
                        elif ai_is_on and ai_lane_index != NG_LANE_INDEX:
                            job_lane_index = ai_lane_index
                            job_status = f"AI_MATCHED ({ai_class_name}) (QR_Fallback)"
                        else:
                            job_status = "ALL_FAILED"
                    
                    job_id = str(uuid.uuid4())[:8] 
                    job_id_log_prefix = f"[JobID {job_id}]"

                    job = {
                        "job_id": job_id, 
                        "lane_index": job_lane_index,
                        "status": job_status,
                        "entry_time": now
                    }

                    if job_lane_index != NG_LANE_INDEX:
                        with state_lock:
                            if 0 <= job_lane_index < len(system_state["lanes"]):
                                job_lane_name = system_state["lanes"][job_lane_index]["name"]
                                system_state["lanes"][job_lane_index]["status"] = "Đang chờ vật..."
                    else:
                        job_lane_name = NG_LANE_NAME
                    
                    current_queue_indices = []
                    with processing_queue_lock:
                        processing_queue.append(job)
                        if len(processing_queue) == 1:
                            queue_head_since = now
                        current_queue_len = len(processing_queue)
                        current_queue_indices = [j["lane_index"] for j in processing_queue]
                    
                    with state_lock:
                        system_state["queue_indices"] = current_queue_indices
                        system_state["entry_queue_size"] = current_queue_len
                    
                    # Ghi log gốc của QR
                    broadcast_log({
                        "log_type": "qr", 
                        "data": {"data_raw": data_raw, "data_key": data_key, "source": qr_source}
                    })
                    # Ghi log của Job mới
                    broadcast_log({"log_type": "info", "message": f"{job_id_log_prefix} Vật vào (Camera Trigger). Ghép cặp: {job_status} -> Lane '{job_lane_name}'."})
                    logging.info(f"[QR/ENTRY] {job_id_log_prefix} (CAM_ENTRY) Ghép cặp: {job_status} -> Lane '{job_lane_name}'. Queue chính: {current_queue_len}")

                    
                    # === (MỚI) LOGIC DỪNG BĂNG CHUYỀN ===
                    if stop_on_qr:
                        logging.info(f"[QR] Đã quét mã. DỪNG băng chuyền trong {stop_delay_qr}s.")
                        CONVEYOR_STOP()
                        executor.submit(restart_conveyor_after_delay, stop_delay_qr)
                    # === KẾT THÚC LOGIC MỚI ===
                            
                elif data_key == "NG":
                    broadcast_log({"log_type": "qr_ng", "data": data_raw, "source": qr_source})
                else:
                    broadcast_log({
                        "log_type": "unknown_qr", 
                        "data": {"data_raw": data_raw, "data_key": data_key, "source": qr_source}
                    }) 
                    logging.warning(f"[QR] ({qr_source}) Không rõ mã QR: raw='{data_raw}', canon='{data_key}', keys={list(LANE_MAP.keys())}")
            
            time.sleep(0.1) 

        except Exception as e:
            logging.error(f"[QR] Lỗi trong luồng QR: {e}", exc_info=True)
            time.sleep(0.5)

# =============================
#       HÀM THỰC THI AI (v3)
# =============================
def run_ai_detection(ng_lane_index):
    if not AI_ENABLED or AI_MODEL is None:
        return ng_lane_index, None

    frame_copy = None
    with frame_lock:
        if latest_frame is not None:
            frame_copy = latest_frame.copy()
            
    if frame_copy is None:
        logging.warning("[AI] Không có frame camera để nhận diện.")
        return ng_lane_index, None

    try:
        results = AI_MODEL.predict(frame_copy, verbose=False, max_det=1) 
        
        if not results or len(results) == 0:
            return ng_lane_index, None

        result = results[0]
        if len(result.boxes) == 0:
            return ng_lane_index, None

        box = result.boxes[0]
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = result.names[class_id].upper()

        if confidence < AI_MIN_CONFIDENCE:
            logging.info(f"[AI] Phát hiện '{class_name}' nhưng độ tin cậy thấp ({confidence:.2f} < {AI_MIN_CONFIDENCE}). Bỏ qua.")
            return ng_lane_index, None

        if class_name in AI_LANE_MAP:
            lane_index = AI_LANE_MAP[class_name]
            logging.info(f"[AI] Phát hiện: '{class_name}' (Conf: {confidence:.2f}) -> Map sang lane index {lane_index}")
            return lane_index, class_name
        else:
            logging.warning(f"[AI] Phát hiện '{class_name}' nhưng class này chưa được map trong ai_config.")
            return ng_lane_index, None

    except Exception as e:
        logging.error(f"[AI] Lỗi trong lúc chạy model.predict: {e}", exc_info=True)
        return ng_lane_index, None

# =============================
#       HÀM HỖ TRỢ BĂNG CHUYỀN
# =============================
def restart_conveyor_after_delay(delay_seconds):
    try:
        time.sleep(delay_seconds)
        logging.info(f"[CONVEYOR] Hết thời gian {delay_seconds}s. Tự động KHỞI ĐỘNG băng chuyền.")
        CONVEYOR_RUN()
    except Exception as e:
        logging.error(f"[CONVEYOR] Lỗi trong luồng tự khởi động lại: {e}")

# =============================
#       LUỒNG SENSOR GÁC CỔNG (FIX #9)
# =============================
# (SỬA) VÔ HIỆU HÓA TOÀN BỘ LUỒNG SENSOR GÁC CỔNG
# def entry_sensor_monitoring_thread():
#     global last_entry_sensor_state, last_entry_sensor_trigger_time, queue_head_since
# 
#     sensor_pin_to_read = SENSOR_ENTRY_PIN
#     if isinstance(GPIO, MockGPIO):
#         sensor_pin_to_read = SENSOR_ENTRY_MOCK_PIN
#         
#     logging.info(f"[ENTRY] Thread Entry Sensor (Pin: {sensor_pin_to_read}) bắt đầu (Đã tích hợp Stability Delay).")
# ... (TOÀN BỘ HÀM ĐÃ BỊ COMMENT OUT) ...
#         last_entry_sensor_state = sensor_now
#         time.sleep(0.05)

# =============================
#       LUỒNG SENSOR LÀN (FIX #9)
# =============================
def lane_sensor_monitoring_thread():
    global last_sensor_state, last_sensor_trigger_time
    global queue_head_since
    
    last_sensor_state_prev = list(last_sensor_state) 

    NG_LANE_INDEX = -1
    NG_LANE_NAME = "Hàng NG"
    try:
        with state_lock:
            for i_ng, lane_ng in enumerate(system_state["lanes"]):
                if canon_id(lane_ng.get("id")) == "NG":
                    NG_LANE_INDEX = i_ng
                    NG_LANE_NAME = lane_ng.get("name", "Hàng NG")
                    break
        logging.info(f"[LANE_S] Đã cấu hình hàng NG tại index: {NG_LANE_INDEX}")
    except Exception as e:
        logging.error(f"[LANE_S] Lỗi khi tìm NG Lane Index: {e}")

    try:
        while main_loop_running:
            if AUTO_TEST_ENABLED or error_manager.is_maintenance():
                time.sleep(0.1); continue

            debounce_time, current_queue_timeout, num_lanes = 0.1, 15.0, 0
            with state_lock:
                cfg_timing = system_state['timing_config']
                debounce_time = cfg_timing.get('sensor_debounce', 0.1)
                current_queue_timeout = cfg_timing.get('queue_head_timeout', 15.0)
                num_lanes = len(system_state['lanes'])
            now = time.time()
            
            if len(last_sensor_state_prev) != num_lanes:
                with state_lock:
                    reference_state = [lane['sensor_reading'] for lane in system_state['lanes']]
                
                if len(reference_state) < num_lanes:
                    reference_state.extend([1] * (num_lanes - len(reference_state)))
                elif len(reference_state) > num_lanes:
                    reference_state = reference_state[:num_lanes]
                
                last_sensor_state_prev = reference_state
                logging.warning(f"[SENSOR] Đã phát hiện thay đổi config. Đồng bộ last_sensor_state_prev (size {num_lanes}).")


            current_queue_indices = []
            with processing_queue_lock:
                current_queue_indices = [j["lane_index"] for j in processing_queue]
                if processing_queue and queue_head_since > 0.0:
                    if (now - queue_head_since) > current_queue_timeout:
                        job_timeout = processing_queue.pop(0) 
                        job_id_timeout = job_timeout.get('job_id', '???') # Fix #9
                        expected_lane_index = job_timeout['lane_index']
                        expected_lane_name = "UNKNOWN"
                        current_queue_indices = [j["lane_index"] for j in processing_queue]
                        
                        with state_lock:
                            if 0 <= expected_lane_index < len(system_state["lanes"]):
                                expected_lane_name = system_state['lanes'][expected_lane_index]['name']
                                if system_state["lanes"][expected_lane_index]["status"].startswith("Đang chờ vật"):
                                    system_state["lanes"][expected_lane_index]["status"] = "Sẵn sàng"
                            system_state["queue_indices"] = current_queue_indices
                            system_state["entry_queue_size"] = len(current_queue_indices)

                        queue_head_since = now if processing_queue else 0.0

                        broadcast_log({
                            "log_type": "warn",
                            "message": f"[JobID {job_id_timeout}] TIMEOUT! Đã tự động xóa Job cho {expected_lane_name} (>{current_queue_timeout}s).",
                            "queue": current_queue_indices
                        })
                        logging.warning(f"[SENSOR] [JobID {job_id_timeout}] TIMEOUT! Xóa Job cho {expected_lane_name}.")
                        
            for i in range(num_lanes):
                sensor_pin, push_pin, lane_name_for_log = None, None, "UNKNOWN"
                with state_lock:
                    if not (0 <= i < len(system_state["lanes"])): continue
                    lane_for_read = system_state["lanes"][i]
                    sensor_pin = lane_for_read.get("sensor_pin")
                    push_pin = lane_for_read.get("push_pin")
                    lane_name_for_log = lane_for_read['name']

                if sensor_pin is None: continue
                # (SỬA) Vô hiệu hóa kiểm tra sensor gác cổng
                # if (sensor_pin == SENSOR_ENTRY_PIN) or (isinstance(GPIO, MockGPIO) and sensor_pin == SENSOR_ENTRY_MOCK_PIN):
                #     continue

                try:
                    sensor_now = GPIO.input(sensor_pin)
                except Exception as gpio_e:
                    logging.error(f"[SENSOR] Lỗi đọc GPIO pin {sensor_pin} ({lane_name_for_log}): {gpio_e}")
                    error_manager.trigger_maintenance(f"Lỗi đọc sensor pin {sensor_pin} ({lane_name_for_log}): {gpio_e}")
                    continue

                with state_lock:
                    if 0 <= i < len(system_state["lanes"]):
                        system_state["lanes"][i]["sensor_reading"] = sensor_now

                prev_state = last_sensor_state_prev[i] if i < len(last_sensor_state_prev) else 1

                if sensor_now == 0 and prev_state == 1:
                    if (now - last_sensor_trigger_time[i]) > debounce_time:
                        last_sensor_trigger_time[i] = now

                        job_to_run = None
                        is_head_match = False
                        
                        with processing_queue_lock:
                            current_queue_indices_for_log = [j["lane_index"] for j in processing_queue]

                            while processing_queue:
                                job_head = processing_queue[0]
                                job_id_head = job_head.get('job_id', '???')

                                if job_head["lane_index"] == i:
                                    is_head_match = True
                                    job_to_run = processing_queue.pop(0)
                                    queue_head_since = now if processing_queue else 0.0
                                    break 

                                elif job_head["lane_index"] == NG_LANE_INDEX:
                                    job_ng_removed = processing_queue.pop(0)
                                    queue_head_since = now if processing_queue else 0.0
                                    current_queue_indices_for_log = [j["lane_index"] for j in processing_queue]
                                    
                                    logging.info(f"[SENSOR] [JobID {job_id_head}] {lane_name_for_log} kích hoạt. Tự động 'tiêu thụ' 1 Job NG khỏi hàng chờ.")
                                    broadcast_log({"log_type": "info", "message": f"[JobID {job_id_head}] Vật NG đã đi thẳng (pass-through). Xóa Job NG.", "queue": current_queue_indices_for_log})
                                    continue 

                                else:
                                    is_head_match = False
                                    job_to_run = None
                                    logging.warning(f"[SENSOR] ⚠️ [JobID {job_id_head}] {lane_name_for_log} kích hoạt nhưng KHÔNG KHỚP Job đầu hàng chờ (Lane {job_head['lane_index']}). Bỏ qua.")
                                    broadcast_log({"log_type": "warn", "message": f"Sensor {lane_name_for_log} kích hoạt (lỗi đồng bộ). Bỏ qua.", "queue": current_queue_indices_for_log})
                                    break
                        
                        if is_head_match and job_to_run:
                            job_id_for_log = job_to_run.get('job_id', 'N/A') # Fix #9
                            
                            with processing_queue_lock:
                                current_queue_indices = [j["lane_index"] for j in processing_queue]

                            with state_lock:
                                system_state["queue_indices"] = current_queue_indices
                                system_state["entry_queue_size"] = len(current_queue_indices)
                                if 0 <= i < len(system_state["lanes"]):
                                    lane_ref = system_state["lanes"][i]
                                    if push_pin is None: lane_ref["status"] = "Đang đi thẳng..."
                                    else: lane_ref["status"] = "Đang chờ đẩy"
                            
                            # Fix #9: Truyền job_id vào
                            threading.Thread(target=sorting_process, args=(i, job_id_for_log), daemon=True).start()
                            
                            broadcast_log({"log_type": "info", "message": f"[JobID {job_id_for_log}] Sensor {lane_name_for_log} khớp Job. Bắt đầu xử lý.", "queue": current_queue_indices})
                            logging.info(f"[LANE_S] [JobID {job_id_for_log}] {lane_name_for_log} kích hoạt. KHỚP Job. Queue chính: {len(current_queue_indices)}")
                        
                        else:
                            pass
                
                if i < len(last_sensor_state_prev):
                    last_sensor_state_prev[i] = sensor_now

            adaptive_sleep = 0.05 if all(s == 1 for s in last_sensor_state_prev) else 0.01 # (SỬA) Bỏ check last_entry_sensor_state
            time.sleep(adaptive_sleep)

    except Exception as e:
        logging.error(f"[ERROR] Luồng lane_sensor_monitoring_thread bị crash: {e}", exc_info=True)
        error_manager.trigger_maintenance(f"Lỗi luồng Lane Sensor: {e}")

# =============================
#     FLASK + WEBSOCKET
# =============================
app = Flask(__name__)
sock = Sock(app)
connected_clients = set()
clients_lock = threading.Lock()

def _add_client(ws):
    with clients_lock: connected_clients.add(ws)
def _remove_client(ws):
    with clients_lock: connected_clients.discard(ws)
def _list_clients():
    with clients_lock: return list(connected_clients)

def broadcast_log(log_data):
    log_data['timestamp'] = time.strftime('%H:%M:%S')
    msg = json.dumps({"type": "log", **log_data})
    for client in _list_clients():
        try: client.send(msg)
        except Exception: _remove_client(client)

# =============================
#     CÁC HÀM CỦA FLASK
# =============================
def check_auth(username, password):
    if not AUTH_ENABLED: return True
    return username == USERNAME and password == PASSWORD
def authenticate():
    return Response('Yêu cầu đăng nhập.', 401, {'WWW-Authenticate': 'Basic realm="Login Required"'})
def requires_auth(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if not AUTH_ENABLED:
            return f(*args, **kwargs)
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

# --- Các hàm broadcast (FIX #8) ---
def broadcast_state():
    last_state_str = ""
    while main_loop_running:
        state_copy_for_json = None # Sẽ chứa bản deepcopy
        
        queue_len = 0
        current_queue_indices = []
        with processing_queue_lock:
            queue_len = len(processing_queue)
            current_queue_indices = [j["lane_index"] for j in processing_queue]
            
        with state_lock:
            # === BẢN VÁ #8: Giảm thời gian lock ===
            # 1. Cập nhật state (nhanh)
            system_state["maintenance_mode"] = error_manager.is_maintenance()
            system_state["last_error"] = error_manager.last_error
            system_state["is_mock"] = isinstance(GPIO, MockGPIO)
            system_state["auth_enabled"] = AUTH_ENABLED
            system_state["gpio_mode"] = system_state['timing_config'].get('gpio_mode', 'BCM')
            system_state["entry_queue_size"] = queue_len
            system_state["queue_indices"] = current_queue_indices
            # (SỬA) Vô hiệu hóa
            # system_state['sensor_entry_reading'] = 1 
            
            # 2. Tạo một bản deepcopy để thả lock
            try:
                state_copy_for_json = copy.deepcopy(system_state)
            except Exception as e:
                logging.warning(f"[BROADCAST] Lỗi khi deepcopy state: {e}")
                time.sleep(0.5)
                continue
            # 3. Thả lock NGAY LẬP TỨC
        
        # 4. Thực hiện các thao tác chậm (JSON encode) BÊN NGOÀI lock
        current_msg = ""
        try:
            current_msg = json.dumps({"type": "state_update", "state": state_copy_for_json})
        except Exception as e:
                logging.warning(f"[BROADCAST] Lỗi JSON encode state: {e}")
                time.sleep(0.5)
                continue

        if current_msg != last_state_str:
            for client in _list_clients():
                try: client.send(current_msg)
                except Exception: _remove_client(client)
            last_state_str = current_msg
        
        time.sleep(0.5)

def generate_frames():
    while main_loop_running:
        frame = None
        if not error_manager.is_maintenance():
            with frame_lock:
                if latest_frame is not None:
                    frame = latest_frame.copy()
        
        if frame is None:
            import numpy as np
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            time.sleep(0.1)
        
        try:
            fps_text = f"FPS: {fps_value:.2f}"
            color = (0, 255, 255) if error_manager.is_maintenance() else (0, 128, 0)
            cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        except Exception as e:
            logging.warning(f"[FRAME] Lỗi khi vẽ FPS: {e}")

        try:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception as encode_e:
            logging.error(f"[CAMERA] Lỗi encode frame: {encode_e}")
            import numpy as np
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 10])
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
        time.sleep(1 / 30) # Stream 30 FPS

# --- Các routes (endpoints) ---

@app.route('/')
@requires_auth
def index():
    return render_template('index.html')

@app.route('/video_feed')
@requires_auth
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/config')
@requires_auth
def get_config():
    with state_lock:
        config_data = {
            "timing_config": system_state.get('timing_config', {}).copy(),
            "ai_config": system_state.get('ai_config', {}).copy(),
            "camera_settings": system_state.get('camera_settings', {}).copy(),
            "lanes_config": [{
                "id": ln.get('id'), "name": ln.get('name'),
                "sensor_pin": ln.get('sensor_pin'), "push_pin": ln.get('push_pin'),
                "pull_pin": ln.get('pull_pin')
             } for ln in system_state.get('lanes', [])]
        }
    return jsonify(config_data)

@app.route('/api/sort_log')
@requires_auth
def get_sort_log():
    output_data = {} 
    
    with database_lock:
        try:
            conn = sqlite3.connect(DATABASE_FILE)
            cursor = conn.cursor()
            
            cursor.execute("SELECT date, lane_name, count FROM sort_log ORDER BY date ASC")
            rows = cursor.fetchall()
            conn.close()
            
            for date, lane_name, count in rows:
                if date not in output_data:
                    output_data[date] = {}
                output_data[date][lane_name] = count
                
        except Exception as e:
            logging.error(f"[API] Lỗi khi đọc /api/sort_log từ SQLite: {e}")
            return jsonify({"error": str(e)}), 500
            
    return jsonify(output_data)


@app.route('/update_config', methods=['POST'])
@requires_auth
def update_config():
    global lanes_config, RELAY_PINS, SENSOR_PINS, RELAY_CONVEYOR_PIN
    global QUEUE_HEAD_TIMEOUT
    global last_sensor_state, last_sensor_trigger_time, auto_test_last_state, auto_test_last_trigger

    new_config_data = request.json
    if not new_config_data:
        return jsonify({"error": "Thiếu dữ liệu JSON"}), 400
    logging.info(f"[CONFIG] Nhận config mới từ API (POST): {new_config_data}")

    new_timing_config = new_config_data.get('timing_config', {})
    new_lanes_config = new_config_data.get('lanes_config')
    new_ai_config = new_config_data.get('ai_config')
    new_camera_settings = new_config_data.get('camera_settings')

    config_to_save = {}
    restart_required = False

    with state_lock:
        current_ai_config = system_state.get('ai_config', {})
        if new_ai_config is not None and new_ai_config != current_ai_config:
            logging.warning("[CONFIG] Cài đặt AI đã thay đổi. Cần khởi động lại.")
            broadcast_log({"log_type": "warn", "message": "Cài đặt AI đã đổi. Cần khởi động lại!"})
            current_ai_config.update(new_ai_config)
            system_state['ai_config'] = current_ai_config
            restart_required = True
        config_to_save['ai_config'] = current_ai_config.copy()
        
        current_camera_settings = system_state.get('camera_settings', {})
        if new_camera_settings is not None and new_camera_settings != current_camera_settings:
            logging.warning("[CONFIG] Cài đặt Camera đã thay đổi. Cần khởi động lại.")
            broadcast_log({"log_type": "warn", "message": "Cài đặt Camera đã đổi. Cần khởi động lại!"})
            current_camera_settings.update(new_camera_settings)
            system_state['camera_settings'] = current_camera_settings
            restart_required = True
        config_to_save['camera_settings'] = current_camera_settings.copy()

        current_timing = system_state['timing_config']
        current_gpio_mode = current_timing.get('gpio_mode', 'BCM')
        
        # (MỚI) Thêm default cho StopOnQR
        default_timing = { 
            "cycle_delay": 0.3, "settle_delay": 0.2, "sensor_debounce": 0.1,
            "push_delay": 0.0, "gpio_mode": "BCM",
            "queue_head_timeout": 15.0, "pending_trigger_timeout": 0.5,
            "RELAY_CONVEYOR_PIN": None, "stop_conveyor_on_entry": False,
            "stability_delay": 0.25,
            "stop_conveyor_on_qr": False,
            "conveyor_stop_delay_qr": 2.0
        }
        temp_timing = default_timing.copy()
        temp_timing.update(current_timing); temp_timing.update(new_timing_config)
        current_timing = temp_timing
        system_state['timing_config'] = current_timing
        
        QUEUE_HEAD_TIMEOUT = current_timing.get('queue_head_timeout', 15.0)
        
        new_conveyor_pin = current_timing.get('RELAY_CONVEYOR_PIN')
        if new_conveyor_pin != RELAY_CONVEYOR_PIN:
            logging.warning("[CONFIG] Chân Relay Băng chuyền đã thay đổi. Cần khởi động lại.")
            RELAY_CONVEYOR_PIN = new_conveyor_pin
            restart_required = True
        
        logging.info(f"[CONFIG] Đã cập nhật động: Queue Timeout={QUEUE_HEAD_TIMEOUT}s")
        logging.info(f"[CONFIG] Đã cập nhật động: Stop Conveyor (Entry)={current_timing.get('stop_conveyor_on_entry')}")
        logging.info(f"[CONFIG] Đã cập nhật động: Stop Conveyor (QR)={current_timing.get('stop_conveyor_on_qr')}")
        logging.info(f"[CONFIG] Đã cập nhật động: Stability Delay={current_timing.get('stability_delay')}")

        
        new_gpio_mode = new_timing_config.get('gpio_mode', current_gpio_mode)
        if new_gpio_mode != current_gpio_mode:
            logging.warning("[CONFIG] Chế độ GPIO đã thay đổi. Cần khởi động lại ứng dụng.")
            broadcast_log({"log_type": "warn", "message": "GPIO Mode đã đổi. Cần khởi động lại!"})
            restart_required = True
        config_to_save['timing_config'] = current_timing.copy()

        if new_lanes_config is not None:
            logging.info("[CONFIG] Cập nhật cấu hình lanes...")
            lanes_config = ensure_lane_ids(new_lanes_config)
            num_lanes = len(lanes_config)
            new_system_lanes = []; new_relay_pins = []; new_sensor_pins = []

            # (SỬA) Vô hiệu hóa
            # if SENSOR_ENTRY_PIN: new_sensor_pins.append(SENSOR_ENTRY_PIN)
            # if isinstance(GPIO, MockGPIO) and SENSOR_ENTRY_MOCK_PIN: new_sensor_pins.append(SENSOR_ENTRY_MOCK_PIN)
            if RELAY_CONVEYOR_PIN: new_relay_pins.append(RELAY_CONVEYOR_PIN)

            for i, lane_cfg in enumerate(lanes_config):
                new_system_lanes.append({
                    "name": lane_cfg.get("name", f"Lane {i+1}"), "id": lane_cfg.get("id"),
                    "status": "Sẵn sàng", "count": 0, 
                    "sensor_pin": lane_cfg.get("sensor_pin"), "push_pin": lane_cfg.get("push_pin"),
                    "pull_pin": lane_cfg.get("pull_pin"), "sensor_reading": 1,
                    "relay_grab": 0, "relay_push": 0
                })
                if lane_cfg.get("sensor_pin") is not None: new_sensor_pins.append(lane_cfg["sensor_pin"])
                if lane_cfg.get("push_pin") is not None: new_relay_pins.append(lane_cfg["push_pin"])
                if lane_cfg.get("pull_pin") is not None: new_relay_pins.append(lane_cfg["pull_pin"])
            
            system_state['lanes'] = new_system_lanes
            
            last_sensor_state = [1] * num_lanes; last_sensor_trigger_time = [0.0] * num_lanes
            auto_test_last_state = [1] * num_lanes; auto_test_last_trigger = [0.0] * num_lanes
            
            RELAY_PINS, SENSOR_PINS = new_relay_pins, new_sensor_pins
            config_to_save['lanes_config'] = lanes_config
            restart_required = True
            logging.warning("[CONFIG] Cấu hình lanes đã thay đổi. Cần khởi động lại ứng dụng.")
            broadcast_log({"log_type": "warn", "message": "Cấu hình Lanes đã đổi. Cần khởi động lại!"})
        else:
            config_to_save['lanes_config'] = [
                {"id": l.get('id'), "name": l['name'], "sensor_pin": l.get('sensor_pin'),
                 "push_pin": l.get('push_pin'), "pull_pin": l.get('pull_pin')}
                for l in system_state['lanes']
            ]

    try:
        with config_file_lock:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config_to_save, f, indent=4)
        
        msg = "Đã lưu config. "
        if restart_required: msg += "Vui lòng khởi động lại hệ thống để áp dụng thay đổi."
        else: msg += "Các thay đổi về timing đã được áp dụng."
        logging.info(f"[CONFIG] {msg}")
        broadcast_log({"log_type": "info", "message": msg})
        
        return jsonify({"message": msg, "config": config_to_save, "restart_required": restart_required})

    except Exception as e:
        logging.error(f"[ERROR] Không thể lưu config (POST): {e}")
        broadcast_log({"log_type": "error", "message": f"Lỗi khi lưu config (POST): {e}"})
        return jsonify({"error": str(e)}), 500

@app.route('/api/reset_maintenance', methods=['POST'])
@requires_auth
def reset_maintenance():
    global queue_head_since
    # (SỬA) Vô hiệu hóa
    # global last_entry_sensor_state, last_entry_sensor_trigger_time

    if error_manager.is_maintenance():
        error_manager.reset()
        
        with qr_queue_lock:
            qr_queue.clear()
        with processing_queue_lock:
            processing_queue.clear()
            queue_head_since = 0.0
        
        # (SỬA) Vô hiệu hóa
        # last_entry_sensor_state = 1
        # last_entry_sensor_trigger_time = 0.0

        with state_lock:
            system_state["queue_indices"] = []
            system_state["entry_queue_size"] = 0
            # (SỬA) Vô hiệu hóa
            # system_state["sensor_entry_reading"] = 1
            
        broadcast_log({"log_type": "success", "message": "Chế độ bảo trì đã được reset. Hàng chờ đã được xóa."})
        return jsonify({"message": "Maintenance mode reset thành công."})
    else:
        return jsonify({"message": "Hệ thống không ở chế độ bảo trì."})

@app.route('/api/queue/reset', methods=['POST'])
@requires_auth
def api_queue_reset():
    global queue_head_since

    if error_manager.is_maintenance():
        return jsonify({"error": "Hệ thống đang bảo trì, không thể reset hàng chờ."}), 403
    try:
        with qr_queue_lock:
            qr_queue.clear()
        with processing_queue_lock:
            processing_queue.clear()
            queue_head_since = 0.0
            current_queue_for_log = []

        with state_lock:
            for lane in system_state["lanes"]:
                lane["status"] = "Sẵn sàng"
            system_state["queue_indices"] = current_queue_for_log
            system_state["entry_queue_size"] = 0
            # (SỬA) Vô hiệu hóa
            # system_state['sensor_entry_reading'] = 1
            
        broadcast_log({"log_type": "warn", "message": "Tất cả hàng chờ (Tạm & Chính) đã được reset thủ công.", "queue": current_queue_for_log})
        logging.info("[API] Tất cả hàng chờ đã được reset thủ công.")
        return jsonify({"message": "Hàng chờ đã được reset."})
    except Exception as e:
        logging.error(f"[API] Lỗi khi reset hàng chờ: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/mock_gpio', methods=['POST'])
@requires_auth
def api_mock_gpio():
    if not isinstance(GPIO, MockGPIO):
        return jsonify({"error": "Chức năng chỉ khả dụng ở chế độ mô phỏng."}), 400
    
    payload = request.get_json(silent=True) or {}; lane_index = payload.get('lane_index')
    pin = payload.get('pin'); requested_state = payload.get('state')
    
    pin_to_mock = None
    lane_name = "N/A"

    if lane_index is not None and pin is None:
        try: lane_index = int(lane_index)
        except (TypeError, ValueError): return jsonify({"error": "lane_index không hợp lệ."}), 400
        with state_lock:
            if 0 <= lane_index < len(system_state['lanes']):
                pin_to_mock = system_state['lanes'][lane_index].get('sensor_pin')
                lane_name = system_state['lanes'][lane_index].get('name', lane_name)
            else: return jsonify({"error": "lane_index vượt ngoài phạm vi."}), 400
    
    elif pin is not None:
        try: pin_to_mock = int(pin)
        except (TypeError, ValueError): return jsonify({"error": "Giá trị pin không hợp lệ."}), 400
        
        # (SỬA) Vô hiệu hóa
        # if pin_to_mock == SENSOR_ENTRY_PIN:
        #     pin_to_mock = SENSOR_ENTRY_MOCK_PIN
        #     lane_name = "SENSOR_ENTRY (Real Pin)"
        # elif pin_to_mock == SENSOR_ENTRY_MOCK_PIN:
        #     lane_name = "SENSOR_ENTRY (Mock Pin)"
        # else:
        #     with state_lock:
        #          for lane in system_state['lanes']:
        #             if lane.get('sensor_pin') == pin_to_mock:
        #                 lane_name = lane.get('name', f"Pin {pin_to_mock}")
        #                 break
        with state_lock:
             for lane in system_state['lanes']:
                if lane.get('sensor_pin') == pin_to_mock:
                    lane_name = lane.get('name', f"Pin {pin_to_mock}")
                    break


    if pin_to_mock is None: return jsonify({"error": "Thiếu thông tin chân sensor."}), 400

    if requested_state is None: logical_state = GPIO.toggle_input_state(pin_to_mock)
    else:
        logical_state = 1 if str(requested_state).strip().lower() in {"1", "true", "high", "inactive"} else 0
        GPIO.set_input_state(pin_to_mock, logical_state)

    with state_lock:
        # (SỬA) Vô hiệu hóa
        # if pin_to_mock == SENSOR_ENTRY_MOCK_PIN:
        #     system_state['sensor_entry_reading'] = 0 if logical_state == 0 else 1
        # else:
        #     for lane in system_state['lanes']:
        #         if lane.get('sensor_pin') == pin_to_mock:
        #             lane['sensor_reading'] = 0 if logical_state == 0 else 1
        #             break
        for lane in system_state['lanes']:
            if lane.get('sensor_pin') == pin_to_mock:
                lane['sensor_reading'] = 0 if logical_state == 0 else 1
                break
                    
    state_label = 'ACTIVE (LOW)' if logical_state == 0 else 'INACTIVE (HIGH)'
    message = f"[MOCK] Sensor pin {pin_to_mock} -> {state_label} ({lane_name})";
    
    broadcast_log({"log_type": "info", "message": message})
    return jsonify({"pin": pin_to_mock, "state": logical_state, "lane": lane_name})

# =============================
#     WEBSOCKET
# =============================
@sock.route('/ws')
@requires_auth
def ws_route(ws):
    global AUTO_TEST_ENABLED, test_seq_running, queue_head_since
    # (SỬA) Vô hiệu hóa
    # global last_entry_sensor_state, last_entry_sensor_trigger_time
    
    auth_user = "guest";
    if AUTH_ENABLED:
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            logging.warning("[WS] Unauthorized connection attempt.")
            ws.close(code=1008, reason="Unauthorized"); return
        auth_user = auth.username
    client_label = f"{auth_user}-{id(ws):x}"
    _add_client(ws)
    logging.info(f"[WS] Client {client_label} connected. Total: {len(_list_clients())}")
    
    queue_len = 0
    current_queue_indices = []
    with processing_queue_lock:
        queue_len = len(processing_queue)
        current_queue_indices = [j["lane_index"] for j in processing_queue]
    try:
        with state_lock:
            system_state["maintenance_mode"] = error_manager.is_maintenance()
            system_state["last_error"] = error_manager.last_error
            system_state["auth_enabled"] = AUTH_ENABLED
            system_state["entry_queue_size"] = queue_len
            # (SỬA) Vô hiệu hóa
            # system_state["sensor_entry_reading"] = last_entry_sensor_state 
            system_state["queue_indices"] = current_queue_indices
            initial_state_msg = json.dumps({"type": "state_update", "state": system_state})
        ws.send(initial_state_msg)
    except Exception as e:
        logging.warning(f"[WS] Lỗi gửi state ban đầu: {e}")
        _remove_client(ws); return

    try:
        while True:
            message = ws.receive()
            if message:
                try:
                    data = json.loads(message)
                    action = data.get('action')
                    if error_manager.is_maintenance() and action != "reset_maintenance":
                        broadcast_log({"log_type": "error", "message": "Hệ thống đang bảo trì, không thể thao tác."})
                        continue

                    if action == 'reset_count':
                        lane_idx = data.get('lane_index')
                        with state_lock:
                            if lane_idx == 'all':
                                for i in range(len(system_state['lanes'])): system_state['lanes'][i]['count'] = 0
                                broadcast_log({"log_type": "info", "message": f"{client_label} đã reset đếm toàn bộ."})
                            elif lane_idx is not None and 0 <= lane_idx < len(system_state['lanes']):
                                lane_name = system_state['lanes'][lane_idx]['name']
                                system_state['lanes'][lane_idx]['count'] = 0
                                broadcast_log({"log_type": "info", "message": f"{client_label} đã reset đếm {lane_name}."})

                    elif action == "test_relay":
                        lane_index = data.get("lane_index"); relay_action = data.get("relay_action")
                        if lane_index is not None and relay_action:
                            executor.submit(_run_test_relay, lane_index, relay_action)
                    elif action == "test_all_relays":
                        executor.submit(_run_test_all_relays)
                    elif action == "toggle_auto_test":
                        AUTO_TEST_ENABLED = data.get("enabled", False)
                        logging.info(f"[TEST] Auto-Test (Sensor->Relay) set by {client_label} to: {AUTO_TEST_ENABLED}")
                        broadcast_log({"log_type": "warn", "message": f"Chế độ Auto-Test đã { 'BẬT' if AUTO_TEST_ENABLED else 'TẮT' } bởi {client_label}."})
                        if not AUTO_TEST_ENABLED: reset_all_relays_to_default()
                    
                    elif action == "reset_maintenance":
                        if error_manager.is_maintenance():
                            error_manager.reset()
                            with qr_queue_lock:
                                qr_queue.clear()
                            with processing_queue_lock:
                                processing_queue.clear()
                                queue_head_since = 0.0
                            
                            # (SỬA) Vô hiệu hóa
                            # last_entry_sensor_state = 1
                            # last_entry_sensor_trigger_time = 0.0
                                
                            with state_lock:
                                system_state["queue_indices"] = []
                                system_state["entry_queue_size"] = 0
                                # (SỬA) Vô hiệu hóa
                                # system_state["sensor_entry_reading"] = 1
                                
                            broadcast_log({"log_type": "success", "message": f"Chế độ bảo trì đã được reset bởi {client_label}. Hàng chờ đã được xóa."})
                        else:
                            broadcast_log({"log_type": "info", "message": "Hệ thống không ở chế độ bảo trì."})

                except json.JSONDecodeError: pass
                except Exception as ws_loop_e: logging.error(f"[WS] Lỗi xử lý message: {ws_loop_e}")
    except Exception as ws_conn_e:
        logging.warning(f"[WS] Kết nối WebSocket bị đóng hoặc lỗi: {ws_conn_e}")
    finally:
        _remove_client(ws)
        logging.info(f"[WS] Client {client_label} disconnected. Total: {len(_list_clients())}")
        
# =============================
#             MAIN (FIX #7)
# =============================
if __name__ == "__main__":
    try:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] (%(threadName)s) %(message)s',
                            handlers=[logging.FileHandler(LOG_FILE, encoding='utf-8'), logging.StreamHandler()])
        
        init_database()
        
        load_local_config()
        
        load_queues_on_startup() # Fix #7: Tải hàng chờ
        
        loaded_gpio_mode = ""
        with state_lock:
            loaded_gpio_mode = system_state.get("gpio_mode", "BCM")

        if isinstance(GPIO, RealGPIO):
            mode_to_set = GPIO.BCM if loaded_gpio_mode == "BCM" else GPIO.BOARD
            GPIO.setmode(mode_to_set); GPIO.setwarnings(False)
            logging.info(f"[GPIO] Đã đặt chế độ chân cắm là: {loaded_gpio_mode}")
            
            active_sensor_pins = list(set([pin for pin in SENSOR_PINS if pin is not None]))
            active_relay_pins = list(set([pin for pin in RELAY_PINS if pin is not None]))
            
            logging.info(f"[GPIO] Setup SENSOR pins: {active_sensor_pins}")
            for pin in active_sensor_pins:
                try: GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
                except Exception as e:
                    logging.critical(f"[CRITICAL] Lỗi cấu hình chân SENSOR {pin}: {e}.")
                    error_manager.trigger_maintenance(f"Lỗi cấu hình chân SENSOR {pin}: {e}")
                    raise
            logging.info(f"[GPIO] Setup RELAY pins: {active_relay_pins}")
            for pin in active_relay_pins:
                try: GPIO.setup(pin, GPIO.OUT)
                except Exception as e:
                    logging.critical(f"[CRITICAL] Lỗi cấu hình chân RELAY {pin}: {e}.")
                    error_manager.trigger_maintenance(f"Lỗi cấu hình chân RELAY {pin}: {e}")
                    raise
        else:
            logging.info("[GPIO] Chạy ở chế độ Mock, bỏ qua setup vật lý.")

        reset_all_relays_to_default()

        # Khởi động các luồng
        threading.Thread(target=camera_capture_thread, name="CameraThread", daemon=True).start()
        threading.Thread(target=qr_detection_loop, name="QRThread", daemon=True).start()
        
        logging.info("[MAIN] Đang khởi động ở chế độ (Gated Job Queue + AI).")
        # (SỬA) Vô hiệu hóa
        # threading.Thread(target=entry_sensor_monitoring_thread, name="EntrySensorThread", daemon=True).start()
        threading.Thread(target=lane_sensor_monitoring_thread, name="LaneSensorThread", daemon=True).start()
            
        threading.Thread(target=broadcast_state, name="BroadcastThread", daemon=True).start()
        threading.Thread(target=periodic_config_save, name="ConfigSaveThread", daemon=True).start()

        logging.info("=========================================")
        logging.info("  HỆ THỐNG PHÂN LOẠI SẴN SÀNG HOẠT ĐỘNG")
        # (SỬA) Cập nhật log
        logging.info(f"  Logic: Camera Trigger (Đã kích hoạt)")
        logging.info(f"  GPIO Mode: {'REAL' if isinstance(GPIO, RealGPIO) else 'MOCK'} (Config: {loaded_gpio_mode})")
        logging.info(f"  API State: http://<IP_CUA_PI>:3000")
        if AUTH_ENABLED:
            logging.info(f"  Truy cập: http://<IP_CUA_PI>:3000 (User: {USERNAME} / Pass: {PASSWORD})")
        else:
            logging.info("  Truy cập: http://<IP_CUA_PI>:3000 (KHÔNG yêu cầu đăng nhập)")
        logging.info("=========================================")
        
        app.run(host='0.0.0.0', port=3000, debug=False)

    except KeyboardInterrupt:
        logging.info("\n🛑 Dừng hệ thống (Ctrl+C)...")
    except Exception as main_e:
        logging.critical(f"[CRITICAL] Lỗi khởi động hệ thống: {main_e}", exc_info=True)
        try:
            if isinstance(GPIO, RealGPIO): GPIO.cleanup()
        except Exception: pass
    finally:
        main_loop_running = False
        
        save_queues_on_shutdown() # Fix #7: Lưu hàng chờ
        
        logging.info("Đang tắt ThreadPoolExecutor...")
        executor.shutdown(wait=False)
        logging.info("Đang cleanup GPIO...")
        try:
            GPIO.cleanup()
            logging.info("✅ GPIO cleaned up.")
        except Exception as clean_e:
            logging.warning(f"Lỗi khi cleanup GPIO: {clean_e}")
        logging.info("👋 Tạm biệt!")

