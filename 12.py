# -*- coding: utf-8 -*-
# (SỬA LỖI 1) Thêm eventlet và monkey_patch ngay dòng đầu tiên
# -*- coding: utf-8 -*-
import os
os.environ.setdefault("EVENTLET_NO_GREENDNS", "yes")  # tránh xung đột DNS trên Windows

import eventlet
eventlet.monkey_patch()  # PHẢI đặt trước: cv2, flask, threading, socket,...

import cv2
import time
import json
import threading
import logging
import functools
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, Response, jsonify, request
from flask_sock import Sock
import unicodedata, re  # ✅ (SỬA LỖI) Đã sửa đúng module 'unicodedata'
import numpy as np # (MỚI) Thêm numpy để tạo frame trắng

# --- (MỚI) Import Pyzbar ---
try:
    import pyzbar.pyzbar as pyzbar
    PYZBAR_AVAILABLE = True
    logging.info("Đã tải thư viện Pyzbar (quét QR nhanh).")
except ImportError:
    PYZBAR_AVAILABLE = False
    logging.warning("="*50)
    logging.warning("LỖI: Không tìm thấy Pyzbar. Sẽ dùng cv2.QRCodeDetector (CHẬM).")
    logging.warning("Vui lòng cài đặt: pip install pyzbar pyzbar-static")
    logging.warning("="*50)

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
    s = re.sub(r"\bQR_?\b", "", s)
    s = re.sub(r"[^A-Z0-9]", "", s)     
    s = re.sub(r"^(LOAI|LO)+", "", s)
    return s

# =============================
#       LỚP TRỪU TƯỢNG GPIO (Giữ nguyên)
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
SORT_LOG_FILE = 'sort_log.json'
ACTIVE_LOW = True
AUTH_ENABLED = os.environ.get("APP_AUTH_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
USERNAME = os.environ.get("APP_USERNAME", "admin")
PASSWORD = os.environ.get("APP_PASSWORD", "123")
# (MỚI) Thêm SENSOR ENTRY
SENSOR_ENTRY_PIN = 6 
SENSOR_ENTRY_MOCK_PIN = 99 

# =============================
#     KHỞI TẠO CÁC ĐỐI TƯỢNG
# =============================
GPIO = get_gpio_provider()
error_manager = ErrorManager()
executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="TestWorker")
sort_log_lock = threading.Lock()
# (MỚI) Thêm config_file_lock
config_file_lock = threading.Lock()

# =============================
# (MỚI) LỚP SMART QUEUE
# =============================
MAX_RETRIES = 3
QUEUE_WARNING_THRESHOLD = 5
QUEUE_AGE_WARNING = 30.0 
MONITOR_INTERVAL = 5.0 

class SmartQueue:
    def __init__(self, name, max_size=100):
        self.name = name
        self.items = []
        self.max_size = max_size
        self.lock = threading.Lock()
        self.history = []  
        logging.info(f"[QUEUE] Khởi tạo SmartQueue: {name} (Max size: {max_size})")
        
    def push(self, item, priority=0, attempts=0, timestamp=None):
        with self.lock:
            if len(self.items) >= self.max_size:
                self._cleanup_old_items()
            
            queue_item = {
                **item,
                "priority": priority,
                "attempts": attempts,
                "timestamp": timestamp if timestamp is not None else time.time()
            }
            
            insert_idx = self._find_insert_position(priority)
            self.items.insert(insert_idx, queue_item)
            
            self.history.append({
                "item_key": item.get('qr_key', 'N/A'),
                "action": "push",
                "priority": priority,
                "size": len(self.items),
                "time": time.time()
            })
            
    def pop(self, lane_index_to_match=None, qr_key_to_match=None):
        with self.lock:
            if not self.items:
                return None, -1
            
            pop_index = -1
            
            if lane_index_to_match is not None:
                pop_index = next((i for i, item in enumerate(self.items) 
                                if item.get('lane_index') == lane_index_to_match), -1)
            elif qr_key_to_match is not None:
                 pop_index = next((i for i, item in enumerate(self.items) 
                                if item.get('qr_key') == qr_key_to_match), -1)
            else:
                pop_index = 0 
                
            if pop_index == -1: return None, -1
            
            item = self.items.pop(pop_index)
            self.history.append({
                "item_key": item.get('qr_key', 'N/A'),
                "action": "pop",
                "priority": item['priority'],
                "size": len(self.items),
                "time": time.time()
            })
            return item, pop_index

    def remove_expired_item(self, timeout):
        with self.lock:
            if not self.items: return None
            now = time.time()
            if not self.items: 
                return None
            head_item = self.items[0]
            if (now - head_item["timestamp"]) > timeout:
                return self.items.pop(0)
            return None
            
    def get_first_item_index(self, lane_index):
        with self.lock:
            return next((i for i, item in enumerate(self.items) 
                        if item.get('lane_index') == lane_index), -1)

    def is_key_in_queue(self, qr_key):
        with self.lock:
            return any(item.get('qr_key') == qr_key for item in self.items)

    def count(self):
        with self.lock: return len(self.items)

    def get_indices(self):
        with self.lock: return [item.get('lane_index') for item in self.items]

    def _cleanup_old_items(self):
        now = time.time()
        self.items = [item for item in self.items 
                     if (time.time() - item["timestamp"]) <= QUEUE_AGE_WARNING 
                     or item['priority'] > 0]
                     
    def _find_insert_position(self, priority):
        for i, item in enumerate(self.items):
            if priority > item["priority"]:
                return i
        return len(self.items)
        
    def get_status(self):
        with self.lock:
            now = time.time()
            oldest_item = min(self.items, key=lambda x: x["timestamp"]) if self.items else None
            return {
                "size": len(self.items),
                "oldest_age": (now - oldest_item["timestamp"]) if oldest_item else None,
                "indices": self.get_indices(),
                "qr_keys": [item.get('qr_key') for item in self.items],
                "priorities": [item.get('priority') for item in self.items],
                "attempts": [item.get('attempts') for item in self.items],
                "is_locked": self.lock.locked()
            }
            
    def get_history(self):
        with self.lock:
            return self.history[-100:] 

# =============================
#       KHAI BÁO CHÂN GPIO
# =============================
DEFAULT_LANES_CONFIG = [
    {"id": "A", "name": "Phân loại A (Đẩy)", "sensor_pin": 3, "push_pin": 17, "pull_pin": 18},
    {"id": "B", "name": "Phân loại B (Đẩy)", "sensor_pin": 23, "push_pin": 27, "pull_pin": 14},
    {"id": "C", "name": "Phân loại C (Đẩy)", "sensor_pin": 24, "push_pin": 22, "pull_pin": 4},
    {"id": "D", "name": "Lane D (Đi thẳng/Thoát)", "sensor_pin": 5, "push_pin": None, "pull_pin": None}, # (MỚI) Lane D dùng Pin 5
]
lanes_config = DEFAULT_LANES_CONFIG 
RELAY_PINS = []
SENSOR_PINS = []

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

# =============================
#       TRẠNG THÁI HỆ THỐNG
# =============================
system_state = {
    "lanes": [],
    "timing_config": {
        "cycle_delay": 0.3, "settle_delay": 0.2, "sensor_debounce": 0.1,
        "push_delay": 0.0, "gpio_mode": "BCM",
        # (MỚI) Thêm 2 cấu hình timeout
        "queue_head_timeout": 15.0, 
        "pending_trigger_timeout": 0.5, # (MỚI) Thêm từ 1.py
    },
    "is_mock": isinstance(GPIO, MockGPIO), "maintenance_mode": False,
    "auth_enabled": AUTH_ENABLED, "gpio_mode": "BCM", "last_error": None,
    "queue_indices": [],
    # (MỚI) Thêm 2 state cho Gated FIFO
    "entry_queue_size": 0, 
    "sensor_entry_reading": 1, 
}

state_lock = threading.Lock()
main_loop_running = True
latest_frame = None
frame_lock = threading.Lock()

# (MỚI) Hàng chờ Gated FIFO
qr_queue = SmartQueue(name="QR_Token_Queue") # Hàng chờ QR (token)
entry_queue = [] # Hàng chờ vật lý (token)
entry_queue_lock = threading.Lock()
qr_log_lock = threading.Lock() # Lock riêng cho file log QR

QUEUE_HEAD_TIMEOUT = 15.0 # Sẽ được load_config() ghi đè
PENDING_TRIGGER_TIMEOUT = 0.5 # Sẽ được load_config() ghi đè

last_sensor_state = []
last_sensor_trigger_time = []
AUTO_TEST_ENABLED = False
auto_test_last_state = []
auto_test_last_trigger = []

# (XÓA) Xóa `pending_sensor_triggers` (logic cũ từ app.py)

# =============================
# (MỚI) HÀM LƯU FILE ATOMIC (An toàn)
# =============================
def atomic_save_json(data, filepath, lock):
    with lock:
        tmp_file = f"{filepath}.tmp"
        try:
            with open(tmp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            os.replace(tmp_file, filepath)
            return True
        except Exception as e:
            logging.error(f"[SAVE] Lỗi atomic save file {filepath}: {e}")
            if os.path.exists(tmp_file):
                try: os.remove(tmp_file)
                except Exception: pass
            return False

# =============================
#     HÀM KHỞI ĐỘNG & CONFIG
# =============================
def load_local_config():
    global lanes_config, RELAY_PINS, SENSOR_PINS, last_sensor_state, last_sensor_trigger_time
    global auto_test_last_state, auto_test_last_trigger
    global QUEUE_HEAD_TIMEOUT, PENDING_TRIGGER_TIMEOUT

    # (MỚI) Cập nhật default timing (từ 1.py)
    default_timing_config = {
        "cycle_delay": 0.3, "settle_delay": 0.2, "sensor_debounce": 0.1,
        "push_delay": 0.0, "gpio_mode": "BCM",
        "queue_head_timeout": 15.0, "pending_trigger_timeout": 0.5
    }
    default_config_full = {"timing_config": default_timing_config, "lanes_config": DEFAULT_LANES_CONFIG}
    loaded_config = default_config_full

    with config_file_lock: # (MỚI) Dùng lock
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
    
    # (XÓA) Xóa logic pending_sensor_triggers (Logic cũ từ app.py)

    with state_lock:
        system_state['timing_config'] = loaded_config['timing_config']
        system_state['gpio_mode'] = loaded_config['timing_config'].get("gpio_mode", "BCM")
        system_state['lanes'] = new_system_lanes
        system_state['auth_enabled'] = AUTH_ENABLED
        system_state['is_mock'] = isinstance(GPIO, MockGPIO)
    
    # (MỚI) Cập nhật timeout từ config
    QUEUE_HEAD_TIMEOUT = loaded_config['timing_config'].get('queue_head_timeout', 15.0)
    PENDING_TRIGGER_TIMEOUT = loaded_config['timing_config'].get('pending_trigger_timeout', 0.5)

    logging.info(f"[CONFIG] Loaded {num_lanes} lanes config.")
    logging.info(f"[CONFIG] Queue Timeout: {QUEUE_HEAD_TIMEOUT}s")
    
    # (MỚI) Thêm SENSOR_ENTRY_PIN vào danh sách các sensor để setup
    if SENSOR_ENTRY_PIN not in SENSOR_PINS: SENSOR_PINS.append(SENSOR_ENTRY_PIN)
    if SENSOR_ENTRY_MOCK_PIN not in SENSOR_PINS and isinstance(GPIO, MockGPIO): 
        SENSOR_PINS.append(SENSOR_ENTRY_MOCK_PIN)

def ensure_lane_ids(lanes_list):
    default_ids = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    for i, lane in enumerate(lanes_list):
        if 'id' not in lane or not lane['id']:
            if i < len(default_ids): lane['id'] = default_ids[i]
            else: lane['id'] = f"LANE_{i+1}"
            logging.warning(f"[CONFIG] Lane {i+1} thiếu ID. Đã gán ID: {lane['id']}")
    return lanes_list

def reset_all_relays_to_default():
    logging.info("[GPIO] Reset tất cả relay về trạng thái mặc định (THU BẬT).")
    with state_lock:
        for lane in system_state["lanes"]:
            pull_pin = lane.get("pull_pin"); push_pin = lane.get("push_pin")
            if pull_pin is not None: RELAY_ON(pull_pin)
            if push_pin is not None: RELAY_OFF(push_pin)
            lane["relay_grab"] = 1 if pull_pin is not None else 0
            lane["relay_push"] = 0
            lane["status"] = "Sẵn sàng"
    time.sleep(0.1)
    logging.info("[GPIO] Reset hoàn tất.")

# (MỚI) Tách logic lưu config/log ra (từ 1.py)
def periodic_config_save():
    while main_loop_running:
        time.sleep(60)
        if error_manager.is_maintenance(): continue
        
        config_to_save = {}; counts_snapshot = {}; today = time.strftime('%Y-%m-%d')
        
        try:
            with state_lock:
                config_to_save['timing_config'] = system_state['timing_config'].copy()
                current_lanes_config = []
                for lane_state in system_state['lanes']:
                    current_lanes_config.append({
                        "id": lane_state['id'], "name": lane_state['name'],
                        "sensor_pin": lane_state.get('sensor_pin'), 
                        "push_pin": lane_state.get('push_pin'), 
                        "pull_pin": lane_state.get('pull_pin')
                    })
                    counts_snapshot[lane_state['name']] = lane_state['count']
                config_to_save['lanes_config'] = current_lanes_config
            
            # Dùng hàm lưu atomic
            if atomic_save_json(config_to_save, CONFIG_FILE, config_file_lock):
                logging.info("[CONFIG] Đã tự động lưu config.")
            
            sort_log = {}
            with sort_log_lock:
                if os.path.exists(SORT_LOG_FILE):
                    try:
                        with open(SORT_LOG_FILE, 'r', encoding='utf-8') as f:
                            file_content = f.read()
                            if file_content: sort_log = json.loads(file_content)
                    except Exception as e:
                        logging.error(f"[SORT_LOG] Lỗi đọc {SORT_LOG_FILE}: {e}")
                        sort_log = {}
            sort_log[today] = counts_snapshot
            
            if atomic_save_json(sort_log, SORT_LOG_FILE, sort_log_lock):
                logging.info("[SORT_LOG] Đã tự động lưu số đếm.")

        except Exception as e:
            logging.error(f"[CONFIG] Lỗi tự động lưu config/log: {e}")

# =============================
#     LƯU LOG ĐẾM SẢN PHẨM
# =============================
def log_sort_count(lane_index, lane_name):
    # Logic đã được chuyển vào periodic_config_save
    pass 

# =============================
#     CHU TRÌNH PHÂN LOẠI
# =============================
def sorting_process(lane_index, qr_key=None, lane_id=None):
    lane_name = ""; push_pin, pull_pin = None, None
    is_sorting_lane = False
    operation_successful = False
    try:
        with state_lock:
            if not (0 <= lane_index < len(system_state["lanes"])):
                logging.error(f"[SORT] Lane index {lane_index} không hợp lệ.")
                return
            cfg = system_state['timing_config']
            delay = cfg['cycle_delay']; settle_delay = cfg['settle_delay']
            lane = system_state["lanes"][lane_index]
            lane_name = lane["name"]; push_pin = lane.get("push_pin"); pull_pin = lane.get("pull_pin")
            is_sorting_lane = not (push_pin is None and pull_pin is None)
            if is_sorting_lane and (push_pin is None or pull_pin is None):
                logging.error(f"[SORT] Lane {lane_name} (index {lane_index}) chưa được cấu hình đủ chân relay.")
                lane["status"] = "Lỗi Config"
                broadcast_log({"log_type": "error", "message": f"Lane {lane_name} thiếu cấu hình chân relay."})
                return
            lane["status"] = "Đang phân loại..." if is_sorting_lane else "Đang đi thẳng..."
        
        if not is_sorting_lane:
            broadcast_log({"log_type": "info", "message": f"Vật phẩm đi thẳng qua {lane_name}"})
        if is_sorting_lane:
            broadcast_log({"log_type": "info", "message": f"Bắt đầu chu trình đẩy {lane_name}"})
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
        
        operation_successful = True
            
    except Exception as e:
        logging.error(f"[SORT] Lỗi trong sorting_process (lane {lane_name}): {e}")
        error_manager.trigger_maintenance(f"Lỗi sorting_process (Lane {lane_name}): {e}")
    finally:
        with state_lock:
            if 0 <= lane_index < len(system_state["lanes"]):
                lane = system_state["lanes"][lane_index]
                if lane_name and lane["status"] != "Lỗi Config":
                    if operation_successful:
                        lane["count"] += 1
                        log_type = "sort" if is_sorting_lane else "pass"
                        broadcast_log({"log_type": log_type, "name": lane_name, "count": lane['count']})
                        
                        # (MỚI) Gọi hàm log_sort_count tách biệt
                        log_sort_count(lane_index, lane_name)
                            
                    if lane["status"] != "Lỗi Config":
                        lane["status"] = "Sẵn sàng"
        if lane_name and operation_successful:
            msg = f"Hoàn tất chu trình cho {lane_name}" if is_sorting_lane else f"Hoàn tất đếm vật phẩm đi thẳng qua {lane_name}"
            broadcast_log({"log_type": "info", "message": msg})


def handle_sorting_with_delay(lane_index, qr_key=None, lane_id=None):
    push_delay = 0.0; lane_name_for_log = f"Lane {lane_index + 1}"
    try:
        with state_lock:
            if not (0 <= lane_index < len(system_state["lanes"])):
                logging.error(f"[DELAY] Lane index {lane_index} không hợp lệ.")
                return
            push_delay = system_state['timing_config'].get('push_delay', 0.0)
            lane_name_for_log = system_state['lanes'][lane_index]['name']

        if push_delay > 0:
            broadcast_log({"log_type": "info", "message": f"Đã thấy vật {lane_name_for_log}, chờ {push_delay}s..."})
            time.sleep(push_delay)
        if not main_loop_running:
            broadcast_log({"log_type": "warn", "message": f"Hủy chu trình {lane_name_for_log} do hệ thống đang tắt."})
            return
        
        current_status = ""
        with state_lock:
            if not (0 <= lane_index < len(system_state["lanes"])): return
            current_status = system_state["lanes"][lane_index]["status"]

        if current_status in ["Đang chờ đẩy", "Sẵn sàng", "Đang đi thẳng..."]: 
            sorting_process(lane_index, qr_key, lane_id)
        else:
            broadcast_log({"log_type": "warn", "message": f"Hủy chu trình {lane_name_for_log} do trạng thái thay đổi ({current_status})."})
    except Exception as e:
        logging.error(f"[ERROR] Lỗi trong luồng handle_sorting_with_delay (lane {lane_name_for_log}): {e}")
        error_manager.trigger_maintenance(f"Lỗi luồng sorting_delay (Lane {lane_name_for_log}): {e}")
        with state_lock:
            if 0 <= lane_index < len(system_state["lanes"]):
                if system_state["lanes"][lane_index]["status"] == "Đang chờ đẩy":
                    system_state["lanes"][lane_index]["status"] = "Sẵn sàng"

# =============================
# (NÂNG CẤP) HÀM TEST RELAY (từ 1.py)
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

        # (SỬA LỖI) Sửa logic test relay (từ 1.py, khớp với 'grab' của index.html)
        if relay_action == "push":
            broadcast_log({"log_type": "info", "message": f"Test: Kích hoạt ĐẨY (PUSH) cho '{lane_name}'."})
            RELAY_OFF(pull_pin); RELAY_ON(push_pin)
            with state_lock:
                if 0 <= lane_index < len(system_state["lanes"]):
                    system_state["lanes"][lane_index]["relay_grab"] = 0
                    system_state["lanes"][lane_index]["relay_push"] = 1
        
        elif relay_action == "grab": # 'grab' là trạng thái nghỉ (Pull ON)
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
                broadcast_log({"log_type": "info", "message": f"Bỏ qua '{lane_name}' (lane đi thẳng/không có relay)."})
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
            
            time.sleep(0.5) # Chờ 0.5s giữa các lane

        if stopped_early: broadcast_log({"log_type": "warn", "message": "Test tuần tự đã dừng."})
        else: broadcast_log({"log_type": "info", "message": "Test tuần tự hoàn tất."})
    finally:
        with test_seq_lock: test_seq_running = False
        reset_all_relays_to_default()
        broadcast({"type": "test_sequence_complete"})


# =============================
#       LUỒNG CAMERA
# =============================
def camera_capture_thread():
    global latest_frame
    camera = cv2.VideoCapture(CAMERA_INDEX)
    camera.set(cv2.CAP_PROP_FPS, 30)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # (MỚI) Thêm cài đặt phơi sáng (từ 1.py)
    camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75) 
    camera.set(cv2.CAP_PROP_EXPOSURE, -4) 
    camera.set(cv2.CAP_PROP_GAIN, 8)

    if not camera.isOpened():
        logging.error("[ERROR] Không mở được camera.")
        error_manager.trigger_maintenance("Không thể mở camera.")
        return
    logging.info("[CAMERA] Camera đã khởi động.")

    retries = 0; max_retries = 5
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
            # Áp dụng lại cài đặt khi mở lại
            camera.set(cv2.CAP_PROP_FPS, 30); camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480); camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75); camera.set(cv2.CAP_PROP_EXPOSURE, -4)
            camera.set(cv2.CAP_PROP_GAIN, 8)
            continue
            
        retries = 0
        with frame_lock:
            latest_frame = frame.copy()
        time.sleep(1 / 60) # Chạy nhanh hơn
    camera.release()
    
# =============================
#     (NÂNG CẤP) QUÉT MÃ QR (PYZBAR + CV2 FALLBACK)
# =============================
def qr_detection_loop():
    detector = cv2.QRCodeDetector()
    last_qr, last_time = "", 0.0
    
    if not PYZBAR_AVAILABLE:
        logging.warning("[QR] Không tìm thấy Pyzbar. CHỈ DÙNG cv2.QRCodeDetector (rất chậm).")
    
    logging.info(f"[QR] Thread QR (Hybrid Pyzbar + cv2) bắt đầu (Pyzbar: {PYZBAR_AVAILABLE}).")

    while main_loop_running:
        try:
            if AUTO_TEST_ENABLED or error_manager.is_maintenance():
                time.sleep(0.2); continue
            
            LANE_MAP_CONFIG = {}; 
            
            with state_lock:
                LANE_MAP_CONFIG = {canon_id(lane.get("id")): idx 
                                   for idx, lane in enumerate(system_state["lanes"]) if lane.get("id")}
                
            frame_copy = None; gray_frame = None 
            with frame_lock:
                if latest_frame is not None: 
                    frame_copy = latest_frame.copy()
            
            if frame_copy is None:
                time.sleep(0.01); continue

            gray_frame = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY) 
            if gray_frame.mean() < 10: time.sleep(0.1); continue

            data_key, data_raw, map_source = None, None, None
            
            # 3.1: Thử Pyzbar (Nhanh)
            if PYZBAR_AVAILABLE:
                barcodes = pyzbar.decode(gray_frame)
                if barcodes:
                    data_raw = barcodes[0].data.decode("utf-8")
                    data_key = canon_id(data_raw)
                    map_source = "Pyzbar"
                        
            # 3.2: Thử cv2.QRCodeDetector (Chậm, Fallback)
            if data_key is None:
                try:
                    data, _, _ = detector.detectAndDecode(gray_frame)
                    if data:
                        data_raw = data
                        data_key = canon_id(data_raw)
                        map_source = "cv2"
                except cv2.error:
                    pass 

            # --- Xử lý kết quả ---
            if data_key and (data_key != last_qr or time.time() - last_time > 3.0):
                last_qr, last_time = data_key, time.time()
                
                mapped_index = LANE_MAP_CONFIG.get(data_key)
                lane_id_to_use = None
                
                if mapped_index is not None:
                     with state_lock:
                        if 0 <= mapped_index < len(system_state['lanes']):
                            lane_id_to_use = system_state['lanes'][mapped_index]['id']

                # --- Logic SmartQueue ---
                if mapped_index is not None and lane_id_to_use is not None:
                    idx = mapped_index
                    lane_name = "UNKNOWN"
                    with state_lock:
                         if 0 <= idx < len(system_state["lanes"]): lane_name = system_state["lanes"][idx]['name']

                    if qr_queue.is_key_in_queue(data_key):
                        logging.info(f"[QR] '{data_raw}' (key: '{data_key}') đã có trong hàng chờ. Bỏ qua.")
                        continue

                    queue_item = {
                        "lane_index": idx,
                        "qr_key": data_key,
                        "lane_id": lane_id_to_use,
                        "map_source": map_source,
                        "data_raw": data_raw
                    }
                    
                    qr_queue.push(queue_item, priority=0)
                    
                    with state_lock:
                        if qr_queue.get_first_item_index(idx) != -1:
                            if system_state["lanes"][idx]["status"] == "Sẵn sàng":
                                system_state["lanes"][idx]["status"] = "Đang chờ vật..."
                        system_state["queue_indices"] = qr_queue.get_indices()
                    
                    broadcast_log({"log_type": "qr", "data": data_raw, "data_key": data_key, "queue": system_state["queue_indices"]})
                    logging.info(f"[QR] '{data_raw}' (key: '{data_key}', src: {map_source}) -> lane {idx} (Thêm vào hàng chờ QR)")

                elif data_key == "NG":
                    broadcast_log({"log_type": "qr_ng", "data": data_raw})
                else:
                    if data_key:
                        broadcast_log({"log_type": "unknown_qr", "data": data_raw, "data_key": data_key}) 
                        logging.warning(f"[QR] Không rõ mã QR: raw='{data_raw}', key='{data_key}' (Nguồn: {map_source})")
            
            time.sleep(0.01) 

        except Exception as e:
            logging.error(f"[QR] Lỗi trong luồng QR: {e}", exc_info=True)
            time.sleep(0.5)

# =============================
# (MỚI) LUỒNG GIÁM SÁT ENTRY SENSOR
# =============================
last_entry_sensor_state = 1
last_entry_sensor_trigger_time = 0.0

def entry_sensor_monitoring_thread():
    global last_entry_sensor_state, last_entry_sensor_trigger_time
    logging.info(f"[ENTRY] Thread Entry Sensor (Pin: {SENSOR_ENTRY_PIN}) bắt đầu.")

    while main_loop_running:
        if AUTO_TEST_ENABLED or error_manager.is_maintenance():
            time.sleep(0.1); continue
        
        debounce_time = 0.1
        with state_lock:
            debounce_time = system_state['timing_config'].get('sensor_debounce', 0.1)
        now = time.time()

        sensor_pin_to_read = SENSOR_ENTRY_MOCK_PIN if isinstance(GPIO, MockGPIO) else SENSOR_ENTRY_PIN
        
        try:
            sensor_now = GPIO.input(sensor_pin_to_read)
        except Exception as gpio_e:
            logging.error(f"[ENTRY] Lỗi đọc GPIO pin {sensor_pin_to_read} (SENSOR_ENTRY): {gpio_e}")
            error_manager.trigger_maintenance(f"Lỗi đọc sensor ENTRY pin {sensor_pin_to_read}: {gpio_e}")
            time.sleep(0.5); continue

        with state_lock:
            system_state["sensor_entry_reading"] = sensor_now

        if sensor_now == 0 and last_entry_sensor_state == 1:
            if (now - last_entry_sensor_trigger_time) > debounce_time:
                last_entry_sensor_trigger_time = now
                
                with entry_queue_lock:
                    entry_queue.append(True) 
                
                broadcast_log({"log_type": "info", "message": f"Vật phẩm đã vào khu vực (SENSOR_ENTRY kích hoạt). Token Entry đã thêm. Queue size: {len(entry_queue)}"})
                logging.info(f"[ENTRY] SENSOR_ENTRY kích hoạt. Token Entry đã thêm. Queue size: {len(entry_queue)}")

        last_entry_sensor_state = sensor_now
        
        time.sleep(0.01)

# =============================
# (MỚI) LUỒNG GIÁM SÁT SENSOR LANE (Logic Gác Cổng)
# =============================
def lane_sensor_monitoring_thread():
    global last_sensor_state, last_sensor_trigger_time

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

            # --- LOGIC XỬ LÝ TIMEOUT & RETRY CHO QR_QUEUE ---
            expired_item = qr_queue.remove_expired_item(current_queue_timeout)
            if expired_item:
                if expired_item["attempts"] < MAX_RETRIES:
                    qr_queue.push(expired_item, 
                                priority=expired_item["priority"] + 1,
                                attempts=expired_item["attempts"] + 1)
                    
                    lane_name = "UNKNOWN"
                    with state_lock:
                        if 0 <= expired_item['lane_index'] < len(system_state["lanes"]):
                            lane_name = system_state['lanes'][expired_item['lane_index']]['name']
                    
                    log_msg = f"TIMEOUT! Item {lane_name} ({expired_item['qr_key']}) hết hạn. Tăng Priority ({expired_item['priority']+1}) và Thử lại #{expired_item['attempts']+1}."
                    broadcast_log({"log_type": "warn", "message": log_msg, "queue": qr_queue.get_indices()})
                    logging.warning(f"[QR_Q] {log_msg}")

                else:
                    lane_name = "UNKNOWN"
                    with state_lock:
                        if 0 <= expired_item['lane_index'] < len(system_state["lanes"]):
                            lane_name = system_state['lanes'][expired_item['lane_index']]['name']
                            if system_state["lanes"][expired_item['lane_index']]["status"] == "Đang chờ vật...":
                                system_state["lanes"][expired_item['lane_index']]["status"] = "Sẵn sàng"

                    log_msg = f"DROPPED! Item {lane_name} ({expired_item['qr_key']}) bị loại bỏ sau {MAX_RETRIES} lần thử."
                    broadcast_log({"log_type": "error", "message": log_msg, "queue": qr_queue.get_indices()})
                    logging.error(f"[QR_Q] {log_msg}")
            # --- HẾT LOGIC TIMEOUT ---

            # --- ĐỌC SENSOR TỪNG LANE ---
            for i in range(num_lanes):
                sensor_pin, push_pin, lane_name_for_log, lane_id = None, None, "UNKNOWN", None
                with state_lock:
                    if not (0 <= i < len(system_state["lanes"])): continue
                    lane_for_read = system_state["lanes"][i]
                    sensor_pin = lane_for_read.get("sensor_pin")
                    lane_name_for_log = lane_for_read['name']
                    lane_id = lane_for_read['id']

                if sensor_pin is None or sensor_pin == SENSOR_ENTRY_PIN: continue 

                try:
                    sensor_now = GPIO.input(sensor_pin)
                except Exception as gpio_e:
                    logging.error(f"[SENSOR] Lỗi đọc GPIO pin {sensor_pin} ({lane_name_for_log}): {gpio_e}")
                    error_manager.trigger_maintenance(f"Lỗi đọc sensor pin {sensor_pin} ({lane_name_for_log}): {gpio_e}")
                    continue

                with state_lock:
                    if 0 <= i < len(system_state["lanes"]):
                        system_state["lanes"][i]["sensor_reading"] = sensor_now

                # --- PHÁT HIỆN SƯỜN XUỐNG (1 -> 0) ---
                if sensor_now == 0 and last_sensor_state[i] == 1:
                    if (now - last_sensor_trigger_time[i]) > debounce_time:
                        last_sensor_trigger_time[i] = now

                        # LOGIC GÁC CỔNG (GATED FIFO)
                        item, qr_index = qr_queue.pop(lane_index_to_match=i)

                        if item:
                            with entry_queue_lock:
                                has_entry_token = len(entry_queue) > 0
                                if has_entry_token: entry_queue.pop(0) 

                            if has_entry_token:
                                # KHỚP HOÀN TOÀN (QR + ENTRY TOKEN)
                                qr_key_to_pass = item['qr_key']
                                lane_id_to_pass = item['lane_id']
                                
                                with state_lock:
                                    if 0 <= i < len(system_state["lanes"]):
                                        lane_ref = system_state["lanes"][i]
                                        if lane_ref.get("push_pin") is None and lane_ref.get("pull_pin") is None: 
                                            lane_ref["status"] = "Đang đi thẳng..."
                                        else: 
                                            lane_ref["status"] = "Đang chờ đẩy"
                                        system_state["queue_indices"] = qr_queue.get_indices()
                                
                                threading.Thread(target=handle_sorting_with_delay, args=(i, qr_key_to_pass, lane_id_to_pass), daemon=True).start()
                                
                                broadcast_log({"log_type": "info", "message": f"Sensor {lane_name_for_log} khớp QR '{qr_key_to_pass}' và Token Entry. BẮT ĐẦU PHÂN LOẠI.", "queue": qr_queue.get_indices()})
                                logging.info(f"[LANE_S] {lane_name_for_log} kích hoạt. KHỚP (QR+ENTRY) -> PUSH. QR_Q: {qr_queue.count()}, ENTRY_Q: {len(entry_queue)}")

                            else:
                                # KHỚP QR NHƯNG THIẾU ENTRY TOKEN
                                qr_queue.push(item, priority=item['priority'], attempts=item['attempts'], timestamp=item['timestamp']) 
                                broadcast_log({"log_type": "warn", "message": f"Sensor {lane_name_for_log} kích hoạt. KHỚP QR ({item['qr_key']}) nhưng thiếu Token Entry. Bỏ qua.", "queue": qr_queue.get_indices()})
                                logging.warning(f"[LANE_S] {lane_name_for_log} kích hoạt. KHỚP QR NHƯNG THIẾU ENTRY TOKEN. Bỏ qua. ENTRY_Q: {len(entry_queue)}")
                                
                        else:
                            # KHÔNG KHỚP QR (Vật lạ)
                            broadcast_log({"log_type": "warn", "message": f"Sensor {lane_name_for_log} kích hoạt nhưng KHÔNG KHỚP QR nào trong hàng chờ. Bỏ qua.", "queue": qr_queue.get_indices()})
                            logging.warning(f"[LANE_S] {lane_name_for_log} kích hoạt. KHÔNG KHỚP QR. Bỏ qua.")
                        
                last_sensor_state[i] = sensor_now

            adaptive_sleep = 0.05 if all(s == 1 for s in last_sensor_state) and last_entry_sensor_state == 1 else 0.01
            time.sleep(adaptive_sleep)

    except Exception as e:
        logging.error(f"[ERROR] Luồng lane_sensor_monitoring_thread bị crash: {e}", exc_info=True)
        error_manager.trigger_maintenance(f"Lỗi luồng Lane Sensor: {e}")

# =============================
# (MỚI) LUỒNG MONITORING QUEUE
# =============================
def queue_monitoring_thread():
    logging.info("[MONITOR] Thread Queue Monitoring bắt đầu.")
    while main_loop_running:
        time.sleep(MONITOR_INTERVAL)
        if error_manager.is_maintenance(): continue
        
        qr_status = qr_queue.get_status()
        
        if qr_status["size"] > QUEUE_WARNING_THRESHOLD:
            logging.warning(f"[MONITOR] Kích thước QR Queue ({qr_status['size']}) vượt ngưỡng cảnh báo ({QUEUE_WARNING_THRESHOLD})!")
            
        if qr_status["oldest_age"] and qr_status["oldest_age"] > QUEUE_AGE_WARNING:
            logging.warning(f"[MONITOR] Item QR cũ nhất ({qr_status['oldest_age']:.1f}s) vượt ngưỡng tuổi ({QUEUE_AGE_WARNING}s)!")
            
        entry_size = 0
        with entry_queue_lock:
            entry_size = len(entry_queue)
            
        if entry_size > QUEUE_WARNING_THRESHOLD:
            logging.warning(f"[MONITOR] Kích thước Entry Queue ({entry_size}) vượt ngưỡng cảnh báo ({QUEUE_WARNING_THRESHOLD})!")


# =============================
#     FLASK + WEBSOCKET
# =============================
app = Flask(__name__)
sock = Sock(app)
connected_clients = set()
clients_lock = threading.Lock()

def _add_client(ws):
    with clients_lock:
        connected_clients.add(ws)

def _remove_client(ws):
    with clients_lock:
        connected_clients.discard(ws)

def _list_clients():
    with clients_lock:
        return list(connected_clients)

def broadcast_log(log_data):
    log_data['timestamp'] = time.strftime('%H:%M:%S')
    msg = json.dumps({"type": "log", **log_data})
    for client in _list_clients():
        try:
            client.send(msg)
        except Exception:
            _remove_client(client)
            
# (MỚI) Hàm broadcast (cho state và test)
def broadcast(data):
    msg = json.dumps(data)
    for client in _list_clients():
        try:
            client.send(msg)
        except Exception:
            _remove_client(client)

# =============================
#     CÁC HÀM CỦA FLASK (TIẾP)
# =============================
def check_auth(username, password):
    if not AUTH_ENABLED: return True
    return username == USERNAME and password == PASSWORD

def authenticate():
    return Response(
        'Yêu cầu đăng nhập.', 401,
        {'WWW-Authenticate': 'Basic realm="Login Required"'})

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

# --- Các hàm broadcast ---
def broadcast_state():
    last_state_str = ""
    while main_loop_running:
        current_msg = ""
        with state_lock:
            system_state["maintenance_mode"] = error_manager.is_maintenance()
            system_state["last_error"] = error_manager.last_error
            system_state["is_mock"] = isinstance(GPIO, MockGPIO)
            system_state["auth_enabled"] = AUTH_ENABLED
            system_state["gpio_mode"] = system_state['timing_config'].get('gpio_mode', 'BCM')
            # (MỚI) Cập nhật các state của Gated FIFO
            system_state["queue_indices"] = qr_queue.get_indices()
            system_state["entry_queue_size"] = len(entry_queue)
            
            current_msg = json.dumps({"type": "state_update", "state": system_state})

        if current_msg != last_state_str:
            # (MỚI) Dùng hàm broadcast thay vì lặp lại
            broadcast({"type": "state_update", "state": system_state})
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
            # (MỚI) Tạo frame đen bằng numpy (từ 1.py)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            time.sleep(0.1)
        try:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception as encode_e:
            logging.error(f"[CAMERA] Lỗi encode frame: {encode_e}")
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 10]) # Chất lượng thấp
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(1 / 20) 

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
            "lanes_config": [{
                "id": ln.get('id'), "name": ln.get('name'),
                "sensor_pin": ln.get('sensor_pin'), "push_pin": ln.get('push_pin'),
                "pull_pin": ln.get('pull_pin')
             } for ln in system_state.get('lanes', [])]
        }
    return jsonify(config_data)

@app.route('/update_config', methods=['POST'])
@requires_auth
def update_config():
    global lanes_config, RELAY_PINS, SENSOR_PINS
    global QUEUE_HEAD_TIMEOUT, PENDING_TRIGGER_TIMEOUT

    new_config_data = request.json
    if not new_config_data:
        return jsonify({"error": "Thiếu dữ liệu JSON"}), 400
    logging.info(f"[CONFIG] Nhận config mới từ API (POST): {new_config_data}")

    new_timing_config = new_config_data.get('timing_config', {})
    new_lanes_config = new_config_data.get('lanes_config')

    config_to_save = {}
    restart_required = False

    with state_lock:
        current_timing = system_state['timing_config']
        current_gpio_mode = current_timing.get('gpio_mode', 'BCM')
        
        # (MỚI) Cập nhật default timing (từ 1.py)
        default_timing = { 
            "queue_head_timeout": 15.0, "pending_trigger_timeout": 0.5,
        }
        temp_timing = default_timing.copy()
        temp_timing.update(current_timing); temp_timing.update(new_timing_config)
        current_timing = temp_timing
        system_state['timing_config'] = current_timing
        
        QUEUE_HEAD_TIMEOUT = current_timing.get('queue_head_timeout', 15.0)
        PENDING_TRIGGER_TIMEOUT = current_timing.get('pending_trigger_timeout', 0.5)
        logging.info(f"[CONFIG] Đã cập nhật động: Queue Timeout={QUEUE_HEAD_TIMEOUT}s")
        
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
            
            # (MỚI) Thêm SENSOR_ENTRY_PIN
            if SENSOR_ENTRY_PIN is not None: new_sensor_pins.append(SENSOR_ENTRY_PIN)
            if SENSOR_ENTRY_MOCK_PIN is not None and isinstance(GPIO, MockGPIO): new_sensor_pins.append(SENSOR_ENTRY_MOCK_PIN)
            
            for i, lane_cfg in enumerate(lanes_config):
                new_system_lanes.append({
                    "name": lane_cfg.get("name", f"Lane {i+1}"), "id": lane_cfg.get("id"),
                    "status": "Sẵn sàng", "count": 0, 
                    "sensor_pin": lane_cfg.get("sensor_pin"), "push_pin": lane_cfg.get("push_pin"),
                    "pull_pin": lane_cfg.get("pull_pin"), "sensor_reading": 1,
                    "relay_grab": 0, "relay_push": 0
                })
                # (MỚI) Đảm bảo không thêm SENSOR_ENTRY_PIN 2 lần
                if lane_cfg.get("sensor_pin") is not None and lane_cfg.get("sensor_pin") != SENSOR_ENTRY_PIN: 
                    new_sensor_pins.append(lane_cfg["sensor_pin"])
                if lane_cfg.get("push_pin") is not None: new_relay_pins.append(lane_cfg["push_pin"])
                if lane_cfg.get("pull_pin") is not None: new_relay_pins.append(lane_cfg["pull_pin"])
            
            system_state['lanes'] = new_system_lanes
            global last_sensor_state, last_sensor_trigger_time, auto_test_last_state, auto_test_last_trigger
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

    # (MỚI) Dùng hàm lưu atomic
    if atomic_save_json(config_to_save, CONFIG_FILE, config_file_lock):
        msg = "Đã lưu config. "
        if restart_required: msg += "Vui lòng KHỞI ĐỘNG LẠI DỊCH VỤ để áp dụng thay đổi."
        else: msg += "Các thay đổi về timing đã được áp dụng."
        logging.info(f"[CONFIG] {msg}")
        broadcast_log({"log_type": "info", "message": msg})
        return jsonify({"message": msg, "config": config_to_save, "restart_required": restart_required})
    else:
        msg = "Lỗi nghiêm trọng khi lưu config. Vui lòng kiểm tra log."
        logging.error(f"[CONFIG] {msg}")
        broadcast_log({"log_type": "error", "message": msg})
        return jsonify({"error": msg}), 500


@app.route('/api/reset_maintenance', methods=['POST'])
@requires_auth
def reset_maintenance():
    # (MỚI) Reset Gated FIFO
    global qr_queue, entry_queue, last_entry_sensor_state, last_entry_sensor_trigger_time 

    if error_manager.is_maintenance():
        error_manager.reset()
        
        with entry_queue_lock: entry_queue.clear()
        qr_queue = SmartQueue(name="QR_Token_Queue") 
        last_entry_sensor_state = 1
        last_entry_sensor_trigger_time = 0.0

        with state_lock:
            system_state["queue_indices"] = []
            system_state["entry_queue_size"] = 0
            system_state["sensor_entry_reading"] = 1
        broadcast_log({"log_type": "success", "message": "Chế độ bảo trì đã được reset. Hàng chờ đã được xóa."})
        return jsonify({"message": "Maintenance mode reset thành công."})
    else:
        return jsonify({"message": "Hệ thống không ở chế độ bảo trì."})

@app.route('/api/queue/reset', methods=['POST'])
@requires_auth
def api_queue_reset():
    if error_manager.is_maintenance():
        return jsonify({"error": "Hệ thống đang bảo trì, không thể reset hàng chờ."}), 403
    try:
        global qr_queue, entry_queue
        qr_queue = SmartQueue(name="QR_Token_Queue")
        with entry_queue_lock: entry_queue.clear()
        
        with state_lock:
            for lane in system_state["lanes"]:
                lane["status"] = "Sẵn sàng"
            system_state["queue_indices"] = []
            system_state["entry_queue_size"] = 0
            
        broadcast_log({"log_type": "warn", "message": "Hàng chờ QR & Entry đã được reset thủ công."})
        logging.info("[API] Hàng chờ QR & Entry đã được reset thủ công.")
        return jsonify({"message": "Hàng chờ đã được reset."})
    except Exception as e:
        logging.error(f"[API] Lỗi khi reset hàng chờ: {e}")
        return jsonify({"error": str(e)}), 500

# (MỚI) Thêm các API debug
@app.route('/api/queue/status')
@requires_auth
def api_queue_status():
    with entry_queue_lock:
        entry_size = len(entry_queue)
    return jsonify({
        "qr_queue": qr_queue.get_status(),
        "entry_queue": {
            "size": entry_size,
            "tokens": [True] * entry_size 
        }
    })

@app.route('/api/queue/history')
@requires_auth
def api_queue_history():
    return jsonify({"history": qr_queue.get_history()})

@app.route('/api/sort_log')
@requires_auth
def get_sort_log():
    sort_log = {}
    with sort_log_lock:
        if os.path.exists(SORT_LOG_FILE):
            try:
                with open(SORT_LOG_FILE, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                    if file_content: sort_log = json.loads(file_content)
            except Exception as e:
                logging.error(f"[SORT_LOG] Lỗi đọc {SORT_LOG_FILE}: {e}")
                sort_log = {}
    return jsonify(sort_log)

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
        try: pin = int(pin)
        except (TypeError, ValueError): return jsonify({"error": "Giá trị pin không hợp lệ."}), 400
        if pin == SENSOR_ENTRY_PIN: pin_to_mock = SENSOR_ENTRY_MOCK_PIN; lane_name = "SENSOR_ENTRY"
        elif pin == SENSOR_ENTRY_MOCK_PIN: pin_to_mock = SENSOR_ENTRY_MOCK_PIN; lane_name = "SENSOR_ENTRY (Mock)"
        else: pin_to_mock = pin 
    
    if pin_to_mock is None: return jsonify({"error": "Không thể mô phỏng sensor cho Lane/Pin không hợp lệ."}), 400
    
    if requested_state is None: logical_state = GPIO.toggle_input_state(pin_to_mock)
    else:
        logical_state = 1 if str(requested_state).strip().lower() in {"1", "true", "high", "inactive"} else 0
        GPIO.set_input_state(pin_to_mock, logical_state)
        
    
    with state_lock:
        if pin_to_mock == SENSOR_ENTRY_MOCK_PIN or pin_to_mock == SENSOR_ENTRY_PIN:
            system_state['sensor_entry_reading'] = 0 if logical_state == 0 else 1
        else:
            for lane in system_state['lanes']:
                if lane.get('sensor_pin') == pin_to_mock:
                    lane['sensor_reading'] = 0 if logical_state == 0 else 1
                    lane_name = lane.get('name', lane_name)
    state_label = 'ACTIVE (LOW)' if logical_state == 0 else 'INACTIVE (HIGH)'
    message = f"[MOCK] Sensor pin {pin_to_mock} -> {state_label} ({lane_name})";
    broadcast_log({"log_type": "info", "message": message})
    return jsonify({"pin": pin_to_mock, "state": logical_state, "lane": lane_name})

@sock.route('/ws')
@requires_auth
def ws_route(ws):
    global AUTO_TEST_ENABLED, test_seq_running, last_entry_sensor_state, last_entry_sensor_trigger_time
    # (MỚI) Thêm global
    global qr_queue, entry_queue

    auth = request.authorization if AUTH_ENABLED else None
    client_label = auth.username if auth else f"guest-{id(ws):x}"
    _add_client(ws)
    logging.info(f"[WS] Client {client_label} connected. Total: {len(_list_clients())}")

    try:
        with state_lock:
            system_state["maintenance_mode"] = error_manager.is_maintenance()
            system_state["last_error"] = error_manager.last_error
            system_state["auth_enabled"] = AUTH_ENABLED
            # (MỚI) Thêm state
            system_state["queue_indices"] = qr_queue.get_indices()
            system_state["entry_queue_size"] = len(entry_queue)
            initial_state_msg = json.dumps({"type": "state_update", "state": system_state})
        ws.send(initial_state_msg)
    except Exception as e:
        logging.warning(f"[WS] Lỗi gửi state ban đầu: {e}")
        _remove_client(ws)
        return

    # (SỬA LỖI WS 1001) Cấu trúc lại vòng lặp while True và try...except
    try:
        # while True PHẢI nằm bên ngoài try...except của timeout
        while True: 
            try:
                # Dùng timeout=1 để không block luồng
                message = ws.receive(timeout=1) 
                
                if message:
                    # Nếu có tin nhắn, xử lý logic như cũ
            try:
                        data = json.loads(message)
                        action = data.get('action')

                        if error_manager.is_maintenance() and action not in ["reset_maintenance", "reset_count"]:
                            broadcast_log({"log_type": "error", "message": "Hệ thống đang bảo trì, không thể thao tác."})
                            continue

                        if action == 'reset_count':
                            lane_idx = data.get('lane_index')

                    if action == 'reset_count':
                        lane_idx = data.get('lane_index')
                        with state_lock:
                            if lane_idx == 'all':
                                for i in range(len(system_state['lanes'])):
                                    system_state['lanes'][i]['count'] = 0
                                broadcast_log({"log_type": "info", "message": f"{client_label} đã reset đếm toàn bộ."})
                            elif lane_idx is not None and 0 <= lane_idx < len(system_state['lanes']):
                                lane_name = system_state['lanes'][lane_idx]['name']
                                system_state['lanes'][lane_idx]['count'] = 0
                                broadcast_log({"log_type": "info", "message": f"{client_label} đã reset đếm {lane_name}."})

                        elif action == "test_relay":
                            lane_index = data.get("lane_index")
                        relay_action = data.get("relay_action")
                        if lane_index is not None and relay_action:
                            executor.submit(_run_test_relay, lane_index, relay_action)

                    elif action == "test_all_relays":
                        executor.submit(_run_test_all_relays)

                    elif action == "toggle_auto_test":
                        AUTO_TEST_ENABLED = data.get("enabled", False)
                        logging.info(f"[TEST] Auto-Test (Sensor->Relay) set by {client_label} to: {AUTO_TEST_ENABLED}")
                        broadcast_log({"log_type": "warn", "message": f"Chế độ Auto-Test đã { 'BẬT' if AUTO_TEST_ENABLED else 'TẮT' } bởi {client_label}."})
                        if not AUTO_TEST_ENABLED:
                            reset_all_relays_to_default()

                    elif action == "reset_maintenance":
                        if error_manager.is_maintenance():
                            error_manager.reset()
                            # (MỚI) Reset Gated FIFO
                            with entry_queue_lock: entry_queue.clear()
                            qr_queue = SmartQueue(name="QR_Token_Queue") 
                            last_entry_sensor_state = 1
                            last_entry_sensor_trigger_time = 0.0

                            with state_lock:
                                system_state["queue_indices"] = []
                                system_state["entry_queue_size"] = 0
                            broadcast_log({"log_type": "success", "message": f"Chế độ bảo trì đã được reset bởi {client_label}. Hàng chờ đã được xóa."})
                        else:
                            broadcast_log({"log_type": "info", "message": "Hệ thống không ở chế độ bảo trì."})
                    
                    # (MỚI) Thêm action reset queue
                    elif action == "api_queue_reset":
                        if error_manager.is_maintenance():
                            broadcast_log({"log_type": "error", "message": "Hệ thống đang bảo trì, không thể reset hàng chờ."})
                            continue
                        
                        qr_queue = SmartQueue(name="QR_Token_Queue")
                        with entry_queue_lock: entry_queue.clear()
                        
                        with state_lock:
                            for lane in system_state["lanes"]:
                                lane["status"] = "Sẵn sàng"
                            system_state["queue_indices"] = []
                            system_state["entry_queue_size"] = 0
                            
                        broadcast_log({"log_type": "warn", "message": f"Hàng chờ QR & Entry đã được reset thủ công bởi {client_label}."})
                        logging.info("[API] Hàng chờ QR & Entry đã được reset thủ công.")


                    except json.JSONDecodeError:
                        logging.warning(f"[WS] Nhận dữ liệu không phải JSON: {message}")
                        continue # Thêm continue để bỏ qua tin nhắn lỗi
                    except Exception as ws_loop_e:
                        logging.error(f"[WS] Lỗi xử lý message: {ws_loop_e}")
            
            except Exception as inner_e:
                # Bắt lỗi timeout từ ws.receive(timeout=1)
                if "timed out" in str(inner_e).lower():
                    # Đây là timeout, không phải lỗi.
                    # Chỉ cần sleep nhẹ để vòng lặp tiếp tục mà không tốn CPU.
                    time.sleep(0.05) 
                    pass # Bỏ qua và tiếp tục vòng lặp while True
                else:
                    # Đây là lỗi thật (ví dụ: Connection closed 1001)
                    logging.warning(f"[WS] Lỗi trong vòng nhận WS: {inner_e}")
                    break # Thoát khỏi vòng lặp while True

    except Exception as ws_conn_e:
        # Lỗi này xảy ra nếu ws.send() ban đầu thất bại
        logging.warning(f"[WS] Kết nối WebSocket bị ngắt (bên ngoài vòng lặp): {ws_conn_e}")
    finally:
        _remove_client(ws)
        logging.info(f"[WS] Client {client_label} disconnected. Total: {len(_list_clients())}")

# =============================
#             MAIN
# =============================
if __name__ == "__main__":
    try:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] (%(threadName)s) %(message)s',
            handlers=[
                logging.FileHandler(LOG_FILE, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )

        load_local_config()

        with state_lock:
            loaded_gpio_mode = system_state.get("gpio_mode", "BCM")

        if isinstance(GPIO, RealGPIO):
            mode_to_set = GPIO.BCM if loaded_gpio_mode == "BCM" else GPIO.BOARD
            GPIO.setmode(mode_to_set)
            GPIO.setwarnings(False)
            logging.info(f"[GPIO] Đã đặt chế độ chân cắm là: {loaded_gpio_mode}")

            # (MỚI) Lấy danh sách pin đã Active (bao gồm cả SENSOR_ENTRY_PIN)
            active_sensor_pins = [pin for pin in SENSOR_PINS if pin is not None]
            active_relay_pins = [pin for pin in RELAY_PINS if pin is not None]

            logging.info(f"[GPIO] Setup SENSOR pins: {active_sensor_pins}")
            for pin in active_sensor_pins:
                try:
                    GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
                except Exception as e:
                    logging.critical(f"[CRITICAL] Lỗi cấu hình chân SENSOR {pin}: {e}. Kiểm tra lại GPIO Mode.")
                    error_manager.trigger_maintenance(f"Lỗi cấu hình chân SENSOR {pin}: {e}")
                    raise
                

            logging.info(f"[GPIO] Setup RELAY pins: {active_relay_pins}")
            for pin in active_relay_pins:
                try:
                    GPIO.setup(pin, GPIO.OUT)
                except Exception as e:
                    logging.critical(f"[CRITICAL] Lỗi cấu hình chân RELAY {pin}: {e}. Kiểm tra lại GPIO Mode.")
                    error_manager.trigger_maintenance(f"Lỗi cấu hình chân RELAY {pin}: {e}")
                    raise
        else:
            logging.info("[GPIO] Chạy ở chế độ Mock, bỏ qua setup vật lý.")


        reset_all_relays_to_default()

        # Khởi tạo các luồng (Thread)
        threading.Thread(target=camera_capture_thread, name="CameraThread", daemon=True).start()
        threading.Thread(target=qr_detection_loop, name="QRThread", daemon=True).start()
        # (MỚI) Thêm 3 luồng
        threading.Thread(target=entry_sensor_monitoring_thread, name="EntrySensorThread", daemon=True).start() 
        threading.Thread(target=lane_sensor_monitoring_thread, name="LaneSensorThread", daemon=True).start()
        threading.Thread(target=queue_monitoring_thread, name="QueueMonitorThread", daemon=True).start()
        
        threading.Thread(target=broadcast_state, name="BroadcastThread", daemon=True).start()
        threading.Thread(target=periodic_config_save, name="ConfigSaveThread", daemon=True).start()


        logging.info("=========================================")
        # (MỚI) Cập nhật tiêu đề
        logging.info("  HỆ THỐNG PHÂN LOẠI SẴN SÀNG (vGATED 1.3 - Flask/Sock)")
        logging.info(f"  Logic: FIFO Gác Cổng (Gated FIFO) + SmartQueue")
        if not PYZBAR_AVAILABLE:
            logging.warning(f"  Scan: CHỈ DÙNG cv2 (Rất chậm - nên cài Pyzbar)")
        else:
            logging.info(f"  Scan: Pyzbar (Nhanh) + cv2 (Dự phòng)")
        logging.info(f"  Sensor Entry Pin: {'MOCK 99' if isinstance(GPIO, MockGPIO) else SENSOR_ENTRY_PIN}")
        logging.info(f"  GPIO Mode: {'REAL' if isinstance(GPIO, RealGPIO) else 'MOCK'} (Config: {loaded_gpio_mode})")
        
        # (SỬA LỖI 2) Thay thế app.run() bằng eventlet.wsgi.server
        logging.info("=========================================")
        logging.info("  KHỞI ĐỘNG SERVER BẰNG EVENTLET (HỖ TRỢ WS)")
        logging.info(f"  Truy cập: http://<IP_CUA_PI>:3000")
        if AUTH_ENABLED:
            logging.info(f"  (User: {USERNAME} / Pass: {PASSWORD})")
        logging.info("=========================================")

        # ✅ (SỬA LỖI) Thêm import eventlet.debug
        import eventlet
        from eventlet import wsgi
        from eventlet import debug # (SỬA LỖI) Thêm import này
        debug.hub_prevent_multiple_readers(False)  # Ngăn lỗi cổng bị giữ
        wsgi.server(eventlet.listen(('0.0.0.0', 3000)), app)

        # (CŨ) Chạy server với threaded=True (GÂY LỖI 1001 CHO WS)
        # app.run(host='0.0.0.0', port=3000, threaded=True)

    except KeyboardInterrupt:
        logging.info("\n🛑 Dừng hệ thống (Ctrl+C)...")
    except Exception as main_e:
        logging.critical(f"[CRITICAL] Lỗi khởi động hệ thống: {main_e}", exc_info=True)
        try:
            if isinstance(GPIO, RealGPIO): GPIO.cleanup()
        except Exception: pass
    finally:
        main_loop_running = False
        logging.info("Đang tắt ThreadPoolExecutor...")
        executor.shutdown(wait=False)

        logging.info("Đang cleanup GPIO...")
        try:
            GPIO.cleanup()
            logging.info("✅ GPIO cleaned up.")
        except Exception as clean_e:
            logging.warning(f"Lỗi khi cleanup GPIO: {clean_e}")

        logging.info("👋 Tạm biệt!")



