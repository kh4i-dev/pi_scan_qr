# -*- coding: utf-8 -*-
import cv2
import time
import json
import threading
import logging
import os
import functools
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, Response, jsonify, request
from flask_sock import Sock
# =============================
#        CỜ ĐIỀU KHIỂN HỆ THỐNG
# =============================
hot_reload_enabled = False  # 🔥 Nếu = True: Bật Hot Reload khi nhấn "Lưu Config"
                           # 🚫 Nếu = False: Chỉ lưu file, KHÔNG khởi động lại camera/sensor

import unicodedata, re # (MỚI) Thêm thư viện xử lý ký tự và regex

# =============================
#       CÁC HÀM TIỆN ÍCH CHUẨN HOÁ
# =============================
def _strip_accents(s: str) -> str:
    # Bỏ dấu tiếng Việt
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))

def canon_id(s: str) -> str:
    """
    Chuẩn hoá ID/QR về dạng so khớp:
    - Bỏ dấu, Uppercase
    - Bỏ prefix QR_/QR, LOAI/LO
    - Bỏ mọi ký tự không phải A-Z/0-9
    """
    if s is None:
        return ""
    s = str(s).strip()
    # Decode kiểu '\u1ea0' nếu lỡ truyền vào dạng escaped
    try:
        s = s.encode("utf-8").decode("unicode_escape")
    except Exception:
        pass
        
    s = _strip_accents(s).upper()
    
    # Bỏ các prefix phổ biến và ký tự đặc biệt
    s = re.sub(r"\bQR_?\b", "", s)      # QR_ hoặc QR
    s = re.sub(r"[^A-Z0-9]", "", s)     # Bỏ mọi ký tự không phải A-Z/0-9
    
    # Bỏ tiền tố LOAI / LO ở ĐẦU chuỗi (ví dụ: LOAI1 -> 1, LOAIB -> B)
    s = re.sub(r"^(LOAI|LO)+", "", s)
    
    return s

# =============================
#       LỚP TRỪU TƯỢNG GPIO
# =============================
try:
    # Thử import thư viện thật của Pi
    import RPi.GPIO as RPiGPIO
except (ImportError, RuntimeError):
    # Nếu thất bại (chạy trên Windows/Mac), dùng None
    RPiGPIO = None

class GPIOProvider:
    """Lớp trừu tượng (Abstract Class) để tương tác GPIO."""
    def setup(self, pin, mode, pull_up_down=None):
        raise NotImplementedError
    def output(self, pin, value):
        raise NotImplementedError
    def input(self, pin):
        raise NotImplementedError
    def cleanup(self):
        raise NotImplementedError
    def setmode(self, mode):
        raise NotImplementedError
    def setwarnings(self, value):
        raise NotImplementedError

class RealGPIO(GPIOProvider):
    """Triển khai GPIO thật (chạy trên Raspberry Pi)."""
    def __init__(self):
        if RPiGPIO is None:
            raise ImportError("Không thể tải thư viện RPi.GPIO. Bạn đang chạy trên Pi?")
        self.gpio = RPiGPIO
        self.BOARD = self.gpio.BOARD
        self.BCM = self.gpio.BCM
        self.OUT = self.gpio.OUT
        self.IN = self.gpio.IN
        self.HIGH = self.gpio.HIGH
        self.LOW = self.gpio.LOW
        self.PUD_UP = self.gpio.PUD_UP

    def setmode(self, mode): self.gpio.setmode(mode)
    def setwarnings(self, value): self.gpio.setwarnings(value)
    def setup(self, pin, mode, pull_up_down=None):
        # Chỉ setup nếu pin không phải là None
        if pin is not None:
            if pull_up_down:
                self.gpio.setup(pin, mode, pull_up_down=pull_up_down)
            else:
                self.gpio.setup(pin, mode)
    def output(self, pin, value): 
        if pin is not None:
            self.gpio.output(pin, value)
    def input(self, pin): 
        if pin is not None:
            return self.gpio.input(pin)
        return self.gpio.HIGH # Mặc định trả về HIGH nếu pin là None
    def cleanup(self): self.gpio.cleanup()

class MockGPIO(GPIOProvider):
    """Triển khai GPIO giả lập (Mock) để test trên PC."""
    def __init__(self):
        self.BOARD = "mock_BOARD"
        self.BCM = "mock_BCM"
        self.OUT = "mock_OUT"
        self.IN = "mock_IN"
        self.HIGH = 1
        self.LOW = 0
        self.input_pins = set()
        self.PUD_UP = "mock_PUD_UP"
        self.pin_states = {} # Giả lập trạng thái pin
        logging.warning("="*50)
        logging.warning("KHÔNG TÌM THẤY RPi.GPIO! ĐANG CHẠY Ở CHẾ ĐỘ GIẢ LẬP (MOCK).")
        logging.warning("="*50)

    def setmode(self, mode): logging.info(f"[MOCK] setmode={mode}")
    def setwarnings(self, value): logging.info(f"[MOCK] setwarnings={value}")
    def setup(self, pin, mode, pull_up_down=None):
        if pin is not None:
            logging.info(f"[MOCK] setup pin {pin} mode={mode} pull_up_down={pull_up_down}")
            if mode == self.OUT:
                self.pin_states[pin] = self.LOW # Mặc định là LOW
            else: # IN
                self.pin_states[pin] = self.HIGH # Mặc định là HIGH (do PUD_UP)
                self.input_pins.add(pin)
    def output(self, pin, value):
        if pin is not None:
            logging.info(f"[MOCK] output pin {pin}={value}")
            self.pin_states[pin] = value
    def input(self, pin):
        if pin is not None:
            # Giả lập sensor luôn ở trạng thái 1 (không có vật)
            val = self.pin_states.get(pin, self.HIGH)
            return val
        return self.HIGH # Pin None (không dùng) luôn trả về HIGH
    def set_input_state(self, pin, logical_state):
        """Đặt trạng thái chân input (0 hoặc 1)."""
        if pin not in self.input_pins:
            logging.info(f"[MOCK] Tự động thêm chân input {pin}.")
            self.input_pins.add(pin)
        state = self.HIGH if logical_state else self.LOW
        self.pin_states[pin] = state
        logging.info(f"[MOCK] set_input_state pin {pin} -> {state}")
        return state
    def toggle_input_state(self, pin):
        """Đảo trạng thái chân input và trả về giá trị mới (0/1)."""
        if pin not in self.input_pins:
            self.input_pins.add(pin)
        current = self.input(pin)
        new_state = self.LOW if current == self.HIGH else self.HIGH
        self.pin_states[pin] = new_state
        logging.info(f"[MOCK] toggle_input_state pin {pin} -> {new_state}")
        return 0 if new_state == self.LOW else 1
    def cleanup(self): logging.info("[MOCK] cleanup GPIO")

def get_gpio_provider():
    """Tự động chọn RealGPIO nếu có thư viện, ngược lại chọn MockGPIO."""
    if RPiGPIO:
        return RealGPIO()
    return MockGPIO()

# =============================
#   QUẢN LÝ LỖI (Error Manager)
# =============================
class ErrorManager:
    """Quản lý trạng thái lỗi/bảo trì của hệ thống."""
    def __init__(self):
        self.lock = threading.Lock()
        self.maintenance_mode = False
        self.last_error = None

    def trigger_maintenance(self, message):
        """Kích hoạt chế độ bảo trì."""
        with self.lock:
            if self.maintenance_mode: # Đã ở chế độ bảo trì rồi
                return
            self.maintenance_mode = True
            self.last_error = message
            logging.critical("="*50)
            logging.critical(f"[MAINTENANCE MODE] Lỗi nghiêm trọng: {message}")
            logging.critical("Hệ thống đã dừng hoạt động. Yêu cầu kiểm tra.")
            logging.critical("="*50)
            # Gửi log cho client ngay lập tức
            broadcast_log({"log_type": "error", "message": f"MAINTENANCE MODE: {message}"})

    def reset(self):
        """Reset lại trạng thái (khi admin yêu cầu)."""
        with self.lock:
            self.maintenance_mode = False
            self.last_error = None
            logging.info("[MAINTENANCE MODE] Đã reset chế độ bảo trì.")

            # Reset trạng thái tất cả các lanes về Sẵn sàng
            with state_lock:
                for lane in system_state["lanes"]:
                    lane["status"] = "Sẵn sàng"


    def is_maintenance(self):
        """Kiểm tra xem hệ thống có đang bảo trì không."""
        return self.maintenance_mode

# =============================
#       CẤU HÌNH CHUNG
# =============================
CAMERA_INDEX = 0
CONFIG_FILE = 'config.json'
LOG_FILE = 'system.log'
SORT_LOG_FILE = 'sort_log.json'
ACTIVE_LOW = True

# (MỚI) Thông tin đăng nhập (có thể tắt hoàn toàn qua biến môi trường)
AUTH_ENABLED = os.environ.get("APP_AUTH_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
USERNAME = os.environ.get("APP_USERNAME", "admin")
PASSWORD = os.environ.get("APP_PASSWORD", "123")

# =============================
#     KHỞI TẠO CÁC ĐỐI TƯỢNG
# =============================
GPIO = get_gpio_provider()
error_manager = ErrorManager()

# Khởi tạo ThreadPoolExecutor (giới hạn 3 luồng test)
executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="TestWorker")
# Thêm lock cho file sort_log.json
sort_log_lock = threading.Lock()

# =============================
#       KHAI BÁO CHÂN GPIO
# =============================
# Cấu hình mặc định cho 4 lanes (3 đẩy, 1 đi thẳng) - Đã thêm ID cố định
DEFAULT_LANES_CONFIG = [
    {"id": "A", "name": "Phân loại A (Đẩy)", "sensor_pin": 3, "push_pin": 17, "pull_pin": 18},
    {"id": "B", "name": "Phân loại B (Đẩy)", "sensor_pin": 23, "push_pin": 27, "pull_pin": 14},
    {"id": "C", "name": "Phân loại C (Đẩy)", "sensor_pin": 24, "push_pin": 22, "pull_pin": 4},
    {"id": "D", "name": "Lane D (Đi thẳng/Thoát)", "sensor_pin": None, "push_pin": None, "pull_pin": None},
]
lanes_config = DEFAULT_LANES_CONFIG # Sẽ được cập nhật từ file config

# Tạo danh sách chân từ config (sau khi load)
RELAY_PINS = []
SENSOR_PINS = []

# =============================
#     HÀM ĐIỀU KHIỂN RELAY
# =============================
def RELAY_ON(pin):
    """Bật relay (kích hoạt)."""
    if pin is not None:
        GPIO.output(pin, GPIO.LOW if ACTIVE_LOW else GPIO.HIGH)

def RELAY_OFF(pin):
    """Tắt relay (ngừng kích hoạt)."""
    if pin is not None:
        GPIO.output(pin, GPIO.HIGH if ACTIVE_LOW else GPIO.LOW)

# =============================
#       TRẠNG THÁI HỆ THỐNG
# =============================
# Cấu trúc system_state ban đầu, lanes sẽ được tạo động từ config
system_state = {
    "lanes": [], # Sẽ được tạo động từ lanes_config
    "timing_config": {
        "cycle_delay": 0.3,
        "settle_delay": 0.2,
        "sensor_debounce": 0.1,
        "push_delay": 0.0,
        "gpio_mode": "BCM"
    },
    "is_mock": isinstance(GPIO, MockGPIO),
    "maintenance_mode": False,
    "auth_enabled": AUTH_ENABLED,
    "gpio_mode": "BCM",
    "last_error": None,
    "queue_indices": [] # (MỚI) Thêm mảng chứa index của hàng chờ
}

# Các biến global cho threading
state_lock = threading.Lock()
main_loop_running = True
latest_frame = None
frame_lock = threading.Lock()

# Hàng chờ QR (Queue)
qr_queue = []
queue_lock = threading.Lock()
# CHỐNG KẸT HÀNG CHỜ
QUEUE_HEAD_TIMEOUT = 15.0 # Timeout (giây) cho vật phẩm đầu hàng chờ
queue_head_since = 0.0 # Thời điểm vật phẩm đầu tiên được thêm vào hàng chờ

# Biến cho việc chống nhiễu (debounce) sensor
last_sensor_state = []
last_sensor_trigger_time = []

# Biến toàn cục cho chức năng Test
AUTO_TEST_ENABLED = False
auto_test_last_state = []
auto_test_last_trigger = []

# =============================
# (NÂNG CẤP) TÍNH NĂNG "SENSOR-FIRST"
# =============================
# Mảng lưu trạng thái sensor đang chờ QR (nếu sensor kích hoạt trước)
pending_sensor_triggers = []
# Thời gian tối đa (giây) mà sensor chờ QR
PENDING_TRIGGER_TIMEOUT = 0.5 


# =============================
#     HÀM KHỞI ĐỘNG & CONFIG
# =============================
def load_local_config():
    """Tải cấu hình từ config.json, bao gồm cả timing và lanes."""
    global lanes_config, RELAY_PINS, SENSOR_PINS
    global last_sensor_state, last_sensor_trigger_time
    global auto_test_last_state, auto_test_last_trigger
    # (NÂNG CẤP) Thêm biến global
    global pending_sensor_triggers

    default_timing_config = {
        "cycle_delay": 0.3,
        "settle_delay": 0.2,
        "sensor_debounce": 0.1,
        "push_delay": 0.0,
        "gpio_mode": "BCM"
    }
    default_config_full = {
        "timing_config": default_timing_config,
        "lanes_config": DEFAULT_LANES_CONFIG
    }

    loaded_config = default_config_full

    if os.path.exists(CONFIG_FILE):
        try:
            # (SỬA LỖI ENCODING) Buộc dùng UTF-8 khi đọc
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                file_content = f.read()
                if not file_content:
                    logging.warning("[CONFIG] File config rỗng, dùng mặc định.")
                else:
                    loaded_config_from_file = json.loads(file_content)
                    
                    # Merge timing config
                    timing_cfg = default_timing_config.copy()
                    timing_cfg.update(loaded_config_from_file.get('timing_config', {}))
                    loaded_config['timing_config'] = timing_cfg
                    
                    # Merge lanes config
                    # (MỚI) Dùng hàm an toàn để thêm 'id' nếu thiếu (từ cấu hình cũ)
                    lanes_from_file = loaded_config_from_file.get('lanes_config', DEFAULT_LANES_CONFIG)
                    loaded_config['lanes_config'] = ensure_lane_ids(lanes_from_file)

        except json.JSONDecodeError as e:
            # Lỗi JSON Decode
            logging.error(f"[CONFIG] Lỗi JSON Decode file config ({e}), dùng mặc định.")
            error_manager.trigger_maintenance(f"Lỗi JSON file config.json: {e}")
            loaded_config = default_config_full
        except UnicodeDecodeError as e:
            # Lỗi Unicode (thường là lỗi charmap)
            logging.error(f"[CONFIG] Lỗi mã hóa (Unicode) file config ({e}), dùng mặc định.")
            error_manager.trigger_maintenance(f"Lỗi mã hóa file config.json: {e}")
            loaded_config = default_config_full
        except Exception as e:
            logging.error(f"[CONFIG] Lỗi đọc file config ({e}), dùng mặc định.")
            error_manager.trigger_maintenance(f"Lỗi đọc file config.json: {e}")
            loaded_config = default_config_full
    else:
        logging.warning("[CONFIG] Không có file config, dùng mặc định và tạo mới.")
        try:
            # (SỬA LỖI ENCODING) Buộc dùng UTF-8 khi ghi
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(loaded_config, f, indent=4) # Lưu config đầy đủ
        except Exception as e:
            logging.error(f"[CONFIG] Không thể tạo file config mới: {e}")

    # Cập nhật global config và state
    lanes_config = loaded_config['lanes_config']
    num_lanes = len(lanes_config)

    # Tạo lại system_state["lanes"] dựa trên config mới
    new_system_lanes = []
    RELAY_PINS = []
    SENSOR_PINS = []
    for i, lane_cfg in enumerate(lanes_config):
        # Lấy ID cố định 
        lane_name = lane_cfg.get("name", f"Lane {i+1}")
        lane_id = lane_cfg.get("id", f"LANE_{i+1}") # Lấy ID nếu có, nếu không dùng fallback an toàn
        
        new_system_lanes.append({
            "name": lane_name,
            "id": lane_id,
            "status": "Sẵn sàng",
            "count": 0,
            "sensor_pin": lane_cfg.get("sensor_pin"),
            "push_pin": lane_cfg.get("push_pin"),
            "pull_pin": lane_cfg.get("pull_pin"),
            "sensor_reading": 1,
            "relay_grab": 0,
            "relay_push": 0
        })
        
        if lane_cfg.get("sensor_pin") is not None: SENSOR_PINS.append(lane_cfg["sensor_pin"])
        if lane_cfg.get("push_pin") is not None: RELAY_PINS.append(lane_cfg["push_pin"])
        if lane_cfg.get("pull_pin") is not None: RELAY_PINS.append(lane_cfg["pull_pin"])

    # Khởi tạo các biến state dựa trên số lanes
    last_sensor_state = [1] * num_lanes
    last_sensor_trigger_time = [0.0] * num_lanes
    auto_test_last_state = [1] * num_lanes
    auto_test_last_trigger = [0.0] * num_lanes
    # (NÂNG CẤP) Khởi tạo mảng pending triggers
    pending_sensor_triggers = [0.0] * num_lanes


    with state_lock:
        system_state['timing_config'] = loaded_config['timing_config']
        system_state['gpio_mode'] = loaded_config['timing_config'].get("gpio_mode", "BCM")
        system_state['lanes'] = new_system_lanes
        system_state['auth_enabled'] = AUTH_ENABLED
        system_state['is_mock'] = isinstance(GPIO, MockGPIO)
    logging.info(f"[CONFIG] Loaded {num_lanes} lanes config.")
    logging.info(f"[CONFIG] Loaded timing config: {system_state['timing_config']}")

# (MỚI) Hàm đảm bảo mỗi lane có một ID cố định
def ensure_lane_ids(lanes_list):
    """Thêm ID mặc định ('A', 'B', 'C', ...) nếu lane bị thiếu trường 'id'."""
    default_ids = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'] # Thêm ID mặc định
    for i, lane in enumerate(lanes_list):
        if 'id' not in lane or not lane['id']:
            if i < len(default_ids):
                lane['id'] = default_ids[i]
                logging.warning(f"[CONFIG] Lane {i+1} thiếu ID. Đã gán ID mặc định: {lane['id']}")
            else:
                lane['id'] = f"LANE_{i+1}"
                logging.warning(f"[CONFIG] Lane {i+1} thiếu ID. Đã gán ID fallback: {lane['id']}")
    return lanes_list


def reset_all_relays_to_default():
    """Reset tất cả relay về trạng thái an toàn (THU BẬT, ĐẨY TẮT)."""
    logging.info("[GPIO] Reset tất cả relay về trạng thái mặc định (THU BẬT).")
    with state_lock:
        # Lặp qua state để đảm bảo dùng đúng pin đã load
        for lane in system_state["lanes"]:
            # Chỉ cố gắng điều khiển nếu pin không phải là None
            if lane.get("pull_pin") is not None: RELAY_ON(lane["pull_pin"])
            if lane.get("push_pin") is not None: RELAY_OFF(lane["push_pin"])
            
            # Cập nhật trạng thái (0 hoặc 1)
            lane["relay_grab"] = 1 if lane.get("pull_pin") is not None else 0
            lane["relay_push"] = 0
            lane["status"] = "Sẵn sàng"
    time.sleep(0.1)
    logging.info("[GPIO] Reset hoàn tất.")

def periodic_config_save():
    """Tự động lưu config mỗi 60s (bao gồm cả timing và lanes)."""
    while main_loop_running:
        time.sleep(60)

        if error_manager.is_maintenance():
            continue

        try:
            config_to_save = {}
            with state_lock:
                config_to_save['timing_config'] = system_state['timing_config'].copy()
                # Lấy lanes_config hiện tại (có thể đã thay đổi)
                current_lanes_config = []
                for lane_state in system_state['lanes']:
                    # Chỉ lưu các trường cần thiết, bao gồm ID
                    current_lanes_config.append({
                        "id": lane_state['id'], # Lưu lại ID cố định
                        "name": lane_state['name'],
                        "sensor_pin": lane_state.get('sensor_pin'), 
                        "push_pin": lane_state.get('push_pin'), 
                        "pull_pin": lane_state.get('pull_pin')
                    })
                config_to_save['lanes_config'] = current_lanes_config

            # (SỬA LỖI ENCODING) Buộc dùng UTF-8 khi ghi
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config_to_save, f, indent=4)
            logging.info("[CONFIG] Đã tự động lưu config (timing + lanes).")
        except Exception as e:
            logging.error(f"[CONFIG] Lỗi tự động lưu config: {e}")

# =============================
#       LUỒNG CAMERA
# =============================
def camera_capture_thread():
    global latest_frame
    camera = cv2.VideoCapture(CAMERA_INDEX)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not camera.isOpened():
        logging.error("[ERROR] Không mở được camera.")
        error_manager.trigger_maintenance("Không thể mở camera.")
        return

    retries = 0
    max_retries = 5

    while main_loop_running:
        if error_manager.is_maintenance():
            time.sleep(0.5)
            continue

        ret, frame = camera.read()
        if not ret:
            retries += 1
            logging.warning(f"[WARN] Mất camera (lần {retries}/{max_retries}), thử khởi động lại...")
            broadcast_log({"log_type":"error","message":f"Mất camera (lần {retries}), đang thử lại..."})

            if retries > max_retries:
                logging.critical("[ERROR] Camera lỗi vĩnh viễn. Chuyển sang chế độ bảo trì.")
                error_manager.trigger_maintenance("Camera lỗi vĩnh viễn (mất kết nối).")
                break

            camera.release()
            time.sleep(1)
            camera = cv2.VideoCapture(CAMERA_INDEX)
            continue

        retries = 0

        with frame_lock:
            latest_frame = frame.copy()
        time.sleep(1 / 60)
    camera.release()

# =============================
#     LƯU LOG ĐẾM SẢN PHẨM
# =============================
def log_sort_count(lane_index, lane_name):
    """Ghi lại số lượng đếm vào file JSON theo ngày (an toàn)."""
    with sort_log_lock:
        try:
            today = time.strftime('%Y-%m-%d')

            sort_log = {}
            if os.path.exists(SORT_LOG_FILE):
                try:
                    # (SỬA LỖI ENCODING) Buộc dùng UTF-8 khi đọc
                    with open(SORT_LOG_FILE, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                        if file_content:
                            sort_log = json.loads(file_content)
                except json.JSONDecodeError:
                    logging.error(f"[SORT_LOG] Lỗi đọc file {SORT_LOG_FILE}, file có thể bị hỏng.")
                    backup_name = f"{SORT_LOG_FILE}.{time.strftime('%Y%m%d_%H%M%S')}.bak"
                    try:
                        os.rename(SORT_LOG_FILE, backup_name)
                        logging.warning(f"[SORT_LOG] Đã backup file lỗi thành {backup_name}")
                    except Exception as re:
                        logging.error(f"[SORT_LOG] Không thể backup file lỗi: {re}")
                    sort_log = {}
                except Exception as e:
                    logging.error(f"[SORT_LOG] Lỗi không xác định khi đọc file: {e}")
                    sort_log = {}

            sort_log.setdefault(today, {})
            sort_log[today].setdefault(lane_name, 0)

            sort_log[today][lane_name] += 1

            # (SỬA LỖI ENCODING) Buộc dùng UTF-8 khi ghi
            with open(SORT_LOG_FILE, 'w', encoding='utf-8') as f:
                json.dump(sort_log, f, indent=4)

        except Exception as e:
            logging.error(f"[ERROR] Lỗi khi ghi sort_log.json: {e}")

# =============================
#     CHU TRÌNH PHÂN LOẠI
# =============================
def sorting_process(lane_index):
    """Quy trình đẩy-thu piston (chạy trên 1 luồng riêng)."""
    lane_name = ""
    push_pin, pull_pin = None, None
    is_sorting_lane = False

    try:
        with state_lock:
            if not (0 <= lane_index < len(system_state["lanes"])):
                logging.error(f"[SORT] Lane index {lane_index} không hợp lệ.")
                return

            cfg = system_state['timing_config']
            delay = cfg['cycle_delay']
            settle_delay = cfg['settle_delay']

            lane = system_state["lanes"][lane_index]
            lane_name = lane["name"]
            push_pin = lane.get("push_pin")
            pull_pin = lane.get("pull_pin")
            
            if push_pin is None and pull_pin is None:
                is_sorting_lane = False
            else:
                is_sorting_lane = True

            if is_sorting_lane and (push_pin is None or pull_pin is None):
                logging.error(f"[SORT] Lane {lane_name} (index {lane_index}) chưa được cấu hình đủ chân relay.")
                lane["status"] = "Lỗi Config"
                broadcast_log({"log_type": "error", "message": f"Lane {lane_name} thiếu cấu hình chân relay."})
                return

            lane["status"] = "Đang phân loại..." if is_sorting_lane else "Đang đi thẳng..."

        # Xử lý Lane đi thẳng
        if not is_sorting_lane:
            broadcast_log({"log_type": "info", "message": f"Vật phẩm đi thẳng qua {lane_name}"})
        
        # Chu trình Phân loại (Chỉ cho Lane có piston)
        if is_sorting_lane:
            broadcast_log({"log_type": "info", "message": f"Bắt đầu chu trình đẩy {lane_name}"})

            # 1. Nhả Grab (Pull OFF)
            RELAY_OFF(pull_pin)
            with state_lock: system_state["lanes"][lane_index]["relay_grab"] = 0
            time.sleep(settle_delay)
            if not main_loop_running: return

            # 2. Kích hoạt Push (Push ON)
            RELAY_ON(push_pin)
            with state_lock: system_state["lanes"][lane_index]["relay_push"] = 1
            time.sleep(delay)
            if not main_loop_running: return

            # 3. Tắt Push (Push OFF)
            RELAY_OFF(push_pin)
            with state_lock: system_state["lanes"][lane_index]["relay_push"] = 0
            time.sleep(settle_delay)
            if not main_loop_running: return

            # 4. Kích hoạt Grab (Pull ON)
            RELAY_ON(pull_pin)
            with state_lock: system_state["lanes"][lane_index]["relay_grab"] = 1

    except Exception as e:
        logging.error(f"[SORT] Lỗi trong sorting_process (lane {lane_name}): {e}")
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
            if is_sorting_lane:
                broadcast_log({"log_type": "info", "message": f"Hoàn tất chu trình cho {lane_name}"})
            else:
                broadcast_log({"log_type": "info", "message": f"Hoàn tất đếm vật phẩm đi thẳng qua {lane_name}"})


def handle_sorting_with_delay(lane_index):
    """Luồng trung gian, chờ push_delay rồi mới gọi sorting_process."""
    push_delay = 0.0
    lane_name_for_log = f"Lane {lane_index + 1}"

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

        with state_lock:
            if not (0 <= lane_index < len(system_state["lanes"])): return
            # (NÂNG CẤP) Trạng thái có thể là "Đang chờ đẩy" (QR-first) 
            # hoặc "Sẵn sàng" (Sensor-first, đã xử lý pending)
            current_status = system_state["lanes"][lane_index]["status"]

        # (NÂNG CẤP) Chấp nhận nhiều trạng thái hơn
        if current_status in ["Đang chờ đẩy", "Sẵn sàng"]:
            sorting_process(lane_index)
        else:
            broadcast_log({"log_type": "warn", "message": f"Hủy chu trình {lane_name_for_log} do trạng thái thay đổi ({current_status})."})

    except Exception as e:
        logging.error(f"[ERROR] Lỗi trong luồng handle_sorting_with_delay (lane {lane_name_for_log}): {e}")
        error_manager.trigger_maintenance(f"Lỗi luồng sorting_delay (Lane {lane_name_for_log}): {e}")
        with state_lock:
            if 0 <= lane_index < len(system_state["lanes"]):
                if system_state["lanes"][lane_index]["status"] == "Đang chờ đẩy":
                    system_state["lanes"][lane_index]["status"] = "Sẵn sàng"
                    broadcast_log({"log_type": "error", "message": f"Lỗi delay, reset {lane_name_for_log}"})


# =============================
# (FIX) CÁC HÀM TEST RELAY
# (Thêm các hàm còn thiếu _run_test_relay và _run_test_all_relays)
# =============================
def _run_test_relay(lane_index, relay_action):
    """Chạy test cho 1 relay cụ thể (chạy trong executor)."""
    try:
        lane_index = int(lane_index)
    except (TypeError, ValueError):
        logging.error(f"[TEST] _run_test_relay: lane_index không hợp lệ: {lane_index}")
        return

    lane_name, push_pin, pull_pin = None, None, None
    
    with state_lock:
        if not (0 <= lane_index < len(system_state["lanes"])):
            logging.error(f"[TEST] _run_test_relay: lane_index ngoài phạm vi: {lane_index}")
            return
        lane = system_state["lanes"][lane_index]
        lane_name = lane.get("name")
        push_pin = lane.get("push_pin")
        pull_pin = lane.get("pull_pin")

    if push_pin is None or pull_pin is None:
        broadcast_log({"log_type": "warn", "message": f"Không thể test '{relay_action}' cho {lane_name} (thiếu pin)."})
        return

    broadcast_log({"log_type": "info", "message": f"Bắt đầu test '{relay_action}' cho {lane_name}..."})
    
    # Lấy thời gian delay từ config
    with state_lock:
         cfg = system_state['timing_config']
         cycle_delay = cfg.get('cycle_delay', 0.3)
         settle_delay = cfg.get('settle_delay', 0.2)

    try:
        if relay_action == "push":
            # 1. Nhả Thu (OFF)
            RELAY_OFF(pull_pin)
            with state_lock: system_state["lanes"][lane_index]["relay_grab"] = 0
            time.sleep(settle_delay)
            # 2. Bật Đẩy (ON)
            RELAY_ON(push_pin)
            with state_lock: system_state["lanes"][lane_index]["relay_push"] = 1
            time.sleep(cycle_delay)
        
        elif relay_action == "pull": # 'pull' thực ra là reset về 'thu'
            # 1. Tắt Đẩy (OFF)
            RELAY_OFF(push_pin)
            with state_lock: system_state["lanes"][lane_index]["relay_push"] = 0
            time.sleep(settle_delay)
            # 2. Bật Thu (ON)
            RELAY_ON(pull_pin)
            with state_lock: system_state["lanes"][lane_index]["relay_grab"] = 1
            time.sleep(cycle_delay)

        elif relay_action == "cycle": # Đẩy ra -> Thu về
            # 1. Nhả Thu (OFF)
            RELAY_OFF(pull_pin)
            with state_lock: system_state["lanes"][lane_index]["relay_grab"] = 0
            time.sleep(settle_delay)
            # 2. Bật Đẩy (ON)
            RELAY_ON(push_pin)
            with state_lock: system_state["lanes"][lane_index]["relay_push"] = 1
            time.sleep(cycle_delay)
            # 3. Tắt Đẩy (OFF)
            RELAY_OFF(push_pin)
            with state_lock: system_state["lanes"][lane_index]["relay_push"] = 0
            time.sleep(settle_delay)
            # 4. Bật Thu (ON)
            RELAY_ON(pull_pin)
            with state_lock: system_state["lanes"][lane_index]["relay_grab"] = 1
            time.sleep(0.1) # Chờ 1 chút
        
        # Reset (Giống 'pull')
        else: 
            RELAY_OFF(push_pin)
            with state_lock: system_state["lanes"][lane_index]["relay_push"] = 0
            time.sleep(settle_delay)
            RELAY_ON(pull_pin)
            with state_lock: system_state["lanes"][lane_index]["relay_grab"] = 1

        broadcast_log({"log_type": "success", "message": f"Test '{relay_action}' cho {lane_name} hoàn tất."})

    except Exception as e:
        logging.error(f"[TEST] Lỗi khi đang test {lane_name}: {e}")
        broadcast_log({"log_type": "error", "message": f"Lỗi khi test {lane_name}: {e}"})
    finally:
        # LUÔN LUÔN trả về trạng thái an toàn
        RELAY_OFF(push_pin)
        RELAY_ON(pull_pin)
        with state_lock:
            if 0 <= lane_index < len(system_state["lanes"]):
                system_state["lanes"][lane_index]["relay_push"] = 0
                system_state["lanes"][lane_index]["relay_grab"] = 1
                system_state["lanes"][lane_index]["status"] = "Sẵn sàng"

def _run_test_all_relays():
    """Chạy test 'cycle' cho tất cả các lane (chạy trong executor)."""
    broadcast_log({"log_type": "warn", "message": f"Bắt đầu test chu trình (cycle) TẤT CẢ các lane..."})
    
    num_lanes = 0
    with state_lock:
        num_lanes = len(system_state["lanes"])
        
    if num_lanes == 0:
        logging.warning("[TEST] _run_test_all_relays: Không có lane nào để test.")
        return

    for i in range(num_lanes):
        if not main_loop_running: return # Dừng nếu app tắt
        
        # Kiểm tra xem lane này có pin không
        has_pins = False
        with state_lock:
            if 0 <= i < len(system_state["lanes"]):
                 lane = system_state["lanes"][i]
                 if lane.get("push_pin") is not None and lane.get("pull_pin") is not None:
                     has_pins = True
        
        if has_pins:
            _run_test_relay(i, "cycle")
            time.sleep(0.1) # Nghỉ 0.1s giữa các lane

    broadcast_log({"log_type": "success", "message": "Test tất cả các lane hoàn tất."})
    reset_all_relays_to_default()


# =============================
#     (NÂNG CẤP) QUÉT MÃ QR 
# =============================
def qr_detection_loop():
    global pending_sensor_triggers, queue_head_since
    detector = cv2.QRCodeDetector()
    last_qr, last_time = "", 0.0
    
    logging.info("[QR] Thread QR Detection started (dynamic lane map enabled).")

    while main_loop_running:
        if AUTO_TEST_ENABLED or error_manager.is_maintenance():
            time.sleep(0.2)
            continue
        
        # 🔄 Tạo lại LANE_MAP động theo config hiện tại
        with state_lock:
            # Map ID cố định đã chuẩn hóa (A, B, C...) sang Index (0, 1, 2...)
            LANE_MAP = {canon_id(lane.get("id", f"LANE_{idx+1}")): idx 
                        for idx, lane in enumerate(system_state["lanes"])}

        frame_copy = None
        with frame_lock:
            if latest_frame is not None:
                frame_copy = latest_frame.copy()

        if frame_copy is None:
            time.sleep(0.1)
            continue

        gray_frame = None # Khởi tạo
        try:
            gray_frame = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
            if gray_frame.mean() < 10:
                time.sleep(0.1)
                continue
        except Exception:
            pass # Bỏ qua nếu frame lỗi

        if gray_frame is None:
            time.sleep(0.01) # Thử lại nhanh
            continue

        try:
            # (TỐI ƯU HÓA TỐC ĐỘ) Chạy detect trên frame XÁM thay vì frame MÀU
            data, _, _ = detector.detectAndDecode(gray_frame)
        except cv2.error:
            data = None
            time.sleep(0.1)
            continue

        if data and (data != last_qr or time.time() - last_time > 3.0):
            last_qr, last_time = data, time.time()
            
            # (MỚI) Chuẩn hóa dữ liệu QR code đầu vào
            data_key = canon_id(data)
            data_raw = data.strip()
            now = time.time()

            if data_key in LANE_MAP:
                idx = LANE_MAP[data_key]
                current_queue_for_log = [] # Khởi tạo
                
                # ==================================================
                # (NÂNG CẤP) Logic "Sensor-First"
                # Kiểm tra xem sensor lane này có đang chờ QR không
                # ==================================================
                is_pending_match = False
                with queue_lock: # Cần lock khi kiểm tra pending_sensor_triggers
                    if 0 <= idx < len(pending_sensor_triggers):
                        if (pending_sensor_triggers[idx] > 0.0) and (now - pending_sensor_triggers[idx] < PENDING_TRIGGER_TIMEOUT):
                            is_pending_match = True
                            pending_sensor_triggers[idx] = 0.0 # Xóa cờ chờ
                    current_queue_for_log = list(qr_queue) # Lấy queue log

                if is_pending_match:
                    # TRƯỜNG HỢP 1: Sensor đã kích hoạt TRƯỚC. Giờ QR mới tới.
                    # Khớp! Xử lý ngay, không cần thêm vào hàng chờ.
                    lane_name_for_log = "UNKNOWN"
                    with state_lock:
                         if 0 <= idx < len(system_state["lanes"]):
                            lane_name_for_log = system_state["lanes"][idx]['name']

                    broadcast_log({
                        "log_type": "info",
                        "message": f"QR '{data_raw}' khớp với sensor {lane_name_for_log} đang chờ.",
                        "queue": current_queue_for_log
                    })
                    logging.info(f"[QR] '{data_raw}' -> canon='{data_key}' -> lane {idx} (Khớp pending sensor)")
                    
                    # (NÂNG CẤP) Bỏ qua hàng chờ, chạy xử lý đẩy ngay
                    threading.Thread(target=handle_sorting_with_delay, args=(idx,), daemon=True).start()
                
                else:
                    # TRƯỜNG HỢP 2: Bình thường. QR tới trước. Thêm vào hàng chờ.
                    with queue_lock:
                        is_queue_empty_before = not qr_queue
                        qr_queue.append(idx)
                        current_queue_for_log = list(qr_queue) # Cập nhật lại

                        if is_queue_empty_before and qr_queue:
                            queue_head_since = time.time()

                    with state_lock:
                        if 0 <= idx < len(system_state["lanes"]):
                            if system_state["lanes"][idx]["status"] == "Sẵn sàng":
                                system_state["lanes"][idx]["status"] = "Đang chờ vật..."
                    
                    system_state["queue_indices"] = current_queue_for_log

                    # Gửi log (vẫn dùng data_raw cho log UI)
                    broadcast_log({
                        "log_type": "qr",
                        "data": data_raw, 
                        "data_key": data_key, # Thêm key đã chuẩn hóa cho debug
                        "queue": current_queue_for_log
                    })
                    logging.info(f"[QR] '{data_raw}' -> canon='{data_key}' -> lane index {idx} (Thêm vào hàng chờ)")

                # (FIX LỖI ĐỒNG BỘ UI) Đẩy state update ngay lập tức qua WebSocket
                # (Đã di chuyển ra ngoài if/else)
                with state_lock:
                    # Đảm bảo state được cập nhật trước khi gửi
                    system_state["maintenance_mode"] = error_manager.is_maintenance()
                    system_state["last_error"] = error_manager.last_error
                    current_state_msg = json.dumps({"type": "state_update", "state": system_state})
                
                for client in _list_clients():
                    try:
                        client.send(current_state_msg)
                    except Exception:
                        _remove_client(client)
                        

            elif data_key == "NG":
                broadcast_log({"log_type": "qr_ng", "data": data_raw})
            else:
                # Gửi data gốc chưa chuẩn hóa (data.strip()) cho log UI, nhưng log console ID đã chuẩn hóa
                broadcast_log({"log_type": "unknown_qr", "data": data_raw, "data_key": data_key}) 
                logging.warning(f"[QR] Không rõ mã QR: raw='{data_raw}', canon='{data_key}', keys={list(LANE_MAP.keys())}")
        
        # (TỐI ƯU HÓA TỐC ĐỘ) Giảm sleep time để tăng tốc độ quét QR
        time.sleep(0.01)

# =============================
# (NÂNG CẤP) LUỒNG GIÁM SÁT SENSOR
# =============================
def sensor_monitoring_thread():
    global last_sensor_state, last_sensor_trigger_time
    global queue_head_since, pending_sensor_triggers

    try:
        while main_loop_running:
            if AUTO_TEST_ENABLED or error_manager.is_maintenance():
                time.sleep(0.1)
                continue

            with state_lock:
                debounce_time = system_state['timing_config']['sensor_debounce']
                num_lanes = len(system_state['lanes'])
            now = time.time()

            # --- LOGIC CHỐNG KẸT HÀNG CHỜ ---
            with queue_lock:
                if qr_queue and queue_head_since > 0.0:
                    if (now - queue_head_since) > QUEUE_HEAD_TIMEOUT:
                        expected_lane_index = qr_queue[0]
                        expected_lane_name = "UNKNOWN"
                        with state_lock:
                            if 0 <= expected_lane_index < len(system_state["lanes"]):
                                expected_lane_name = system_state['lanes'][expected_lane_index]['name']
                                if system_state["lanes"][expected_lane_index]["status"] == "Đang chờ vật...":
                                    system_state["lanes"][expected_lane_index]["status"] = "Sẵn sàng"

                        qr_queue.pop(0) # Dùng pop(0) cho list (hoạt động như popleft)
                        current_queue_for_log = list(qr_queue)
                        queue_head_since = now if qr_queue else 0.0

                        broadcast_log({
                            "log_type": "warn",
                            "message": f"TIMEOUT! Đã tự động xóa vật phẩm {expected_lane_name} khỏi hàng chờ (>{QUEUE_HEAD_TIMEOUT}s).",
                            "queue": current_queue_for_log
                        })
                        with state_lock:
                            system_state["queue_indices"] = current_queue_for_log

            # --- ĐỌC SENSOR TỪNG LANE ---
            for i in range(num_lanes):
                with state_lock:
                    if not (0 <= i < len(system_state["lanes"])):
                        continue
                    lane_for_read = system_state["lanes"][i]
                    sensor_pin = lane_for_read.get("sensor_pin")
                    push_pin = lane_for_read.get("push_pin") # Lấy push_pin để quyết định logic

                if sensor_pin is None:
                    continue

                try:
                    sensor_now = GPIO.input(sensor_pin)
                except Exception as gpio_e:
                    logging.error(f"[SENSOR] Lỗi đọc GPIO pin {sensor_pin} (Lane {lane_for_read.get('name', i+1)}): {gpio_e}")
                    error_manager.trigger_maintenance(f"Lỗi đọc sensor pin {sensor_pin} (Lane {lane_for_read.get('name', i+1)}): {gpio_e}")
                    continue

                with state_lock:
                    if 0 <= i < len(system_state["lanes"]):
                        system_state["lanes"][i]["sensor_reading"] = sensor_now

                # --- PHÁT HIỆN SƯỜN XUỐNG ---
                if sensor_now == 0 and last_sensor_state[i] == 1:
                    if (now - last_sensor_trigger_time[i]) > debounce_time:
                        last_sensor_trigger_time[i] = now

                        lane_name_for_log = "UNKNOWN"
                        with state_lock:
                            if 0 <= i < len(system_state["lanes"]):
                                lane_name_for_log = system_state["lanes"][i]['name']

                        # ============================================
                        # (FIX v3.0) LOGIC FIFO NGHIÊM NGẶT
                        # (Thay thế logic 'if i in qr_queue:' bằng 'if i == qr_queue[0]:')
                        # ============================================
                        with queue_lock:
                            if not qr_queue:
                                # --- 1. HÀNG CHỜ RỖNG ---
                                
                                # (LOGIC CŨ VẪN ĐÚNG) Chỉ xử lý lane đi thẳng
                                if push_pin is None:
                                    broadcast_log({"log_type": "info", "message": f"Vật đi thẳng (không QR) qua {lane_name_for_log}."})
                                    threading.Thread(target=sorting_process, args=(i,), daemon=True).start()
                                else:
                                    # Là lane đẩy, kích hoạt Sensor-First (chờ QR)
                                    if 0 <= i < len(pending_sensor_triggers):
                                        pending_sensor_triggers[i] = now 
                                    broadcast_log({"log_type": "warn", "message": f"Sensor {lane_name_for_log} kích hoạt (hàng chờ rỗng). Đang chờ QR (timeout {PENDING_TRIGGER_TIMEOUT}s)..."})

                            elif i == qr_queue[0]:
                                # --- 2. KHỚP ĐẦU HÀNG CHỜ (FIFO Success) ---
                                # Đây là vật chúng ta đang mong đợi!
                                
                                qr_queue.pop(0) # Xóa vật phẩm khỏi ĐẦU hàng chờ
                                current_queue_for_log = list(qr_queue)
                                queue_head_since = now if qr_queue else 0.0
                                
                                with state_lock:
                                    if 0 <= i < len(system_state["lanes"]):
                                        lane_ref = system_state["lanes"][i]
                                        # Đặt trạng thái chờ đẩy (nếu là lane đẩy)
                                        if push_pin is None:
                                            lane_ref["status"] = "Đang đi thẳng..."
                                        else:
                                            lane_ref["status"] = "Đang chờ đẩy"
                                        system_state["queue_indices"] = current_queue_for_log

                                # Khởi chạy luồng xử lý (đẩy hoặc đếm)
                                threading.Thread(target=handle_sorting_with_delay, args=(i,), daemon=True).start()

                                broadcast_log({
                                    "log_type": "info",
                                    "message": f"Sensor {lane_name_for_log} khớp đầu hàng chờ (FIFO).",
                                    "queue": current_queue_for_log
                                })
                                
                                # Xóa cờ chờ sensor (nếu có)
                                if 0 <= i < len(pending_sensor_triggers):
                                    pending_sensor_triggers[i] = 0.0

                            else:
                                # --- 3. KHÔNG KHỚP ĐẦU HÀNG CHỜ (Pass-over) ---
                                # Hàng chờ KHÔNG rỗng, nhưng sensor (i) không phải là
                                # vật ở đầu hàng chờ (qr_queue[0]).
                                # Đây là trường hợp vật đi NGANG QUA sensor khác.
                                
                                # BỎ QUA HOÀN TOÀN - KHÔNG BÁO LỖI
                                pass
                                
                                # (Optional: Ghi log debug nếu cần, nhưng không báo lỗi cho UI)
                                # current_queue_for_log = list(qr_queue)
                                # logging.info(f"[SENSOR] Bỏ qua trigger {lane_name_for_log} (đang chờ lane {qr_queue[0]}). Queue: {current_queue_for_log}")
                        
                        # --- HẾT LOGIC MỚI ---

                last_sensor_state[i] = sensor_now

            adaptive_sleep = 0.05 if all(s == 1 for s in last_sensor_state) else 0.01
            time.sleep(adaptive_sleep)

    except Exception as e:
        logging.error(f"[ERROR] Luồng sensor_monitoring_thread bị crash: {e}")
        error_manager.trigger_maintenance(f"Lỗi luồng Sensor: {e}")



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
    """Gửi 1 tin nhắn log cụ thể cho client."""
    log_data['timestamp'] = time.strftime('%H:%M:%S')
    msg = json.dumps({"type": "log", **log_data})
    for client in _list_clients():
        try:
            client.send(msg)
        except Exception:
            _remove_client(client)

# =============================
#     CÁC HÀM CỦA FLASK (TIẾP)
# =============================
def check_auth(username, password):
    """Kiểm tra username và password."""
    if not AUTH_ENABLED:
        return True
    return username == USERNAME and password == PASSWORD

def authenticate():
    """Gửi phản hồi 401 (Yêu cầu đăng nhập)."""
    return Response(
        'Yêu cầu đăng nhập.', 401,
        {'WWW-Authenticate': 'Basic realm="Login Required"'})

def requires_auth(f):
    """Decorator để yêu cầu đăng nhập cho một route."""
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
    """Gửi state cho client, chỉ khi state thay đổi."""
    last_state_str = ""

    while main_loop_running:
        current_msg = ""
        with state_lock:
            # Cập nhật trạng thái bảo trì và lỗi cuối
            system_state["maintenance_mode"] = error_manager.is_maintenance()
            system_state["last_error"] = error_manager.last_error
            system_state["is_mock"] = isinstance(GPIO, MockGPIO)
            system_state["auth_enabled"] = AUTH_ENABLED
            # Cập nhật chế độ gpio từ timing_config
            system_state["gpio_mode"] = system_state['timing_config'].get('gpio_mode', 'BCM')
            current_msg = json.dumps({"type": "state_update", "state": system_state})

        if current_msg != last_state_str:
            for client in _list_clients():
                try:
                    client.send(current_msg)
                except Exception:
                    # Gỡ client lỗi ra khỏi danh sách
                    _remove_client(client)
            last_state_str = current_msg

        time.sleep(0.5)

def generate_frames():
    """Stream video từ camera."""
    while main_loop_running:
        frame = None
        # Chỉ stream nếu không bảo trì
        if not error_manager.is_maintenance():
            with frame_lock:
                if latest_frame is not None:
                    frame = latest_frame.copy()

        if frame is None:
            # Nếu đang bảo trì hoặc không có frame, gửi frame đen
            frame_path = 'black_frame.png'
            if os.path.exists(frame_path):
                frame = cv2.imread(frame_path) # Tạo 1 file ảnh đen 640x480
            
            if frame is None: # Nếu đọc file lỗi, tạo frame đen bằng numpy
                import numpy as np
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            time.sleep(0.1)
            # Không cần continue, vẫn gửi frame đen

        try:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception as encode_e:
            logging.error(f"[CAMERA] Lỗi encode frame: {encode_e}")
            # Gửi frame đen nếu encode lỗi
            import numpy as np
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 10]) # Chất lượng thấp
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        time.sleep(1 / 20) # Giữ 20 FPS

# --- Các routes (endpoints) ---

@app.route('/')
@requires_auth
def index():
    """Trang chủ (dashboard)."""
    return render_template('index.html')

@app.route('/video_feed')
@requires_auth
def video_feed():
    """Nguồn cấp video."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/config')
@requires_auth
def get_config():
    """API (GET) để lấy config hiện tại (timing + lanes)."""
    with state_lock:
        config_data = {
            "timing_config": system_state.get('timing_config', {}),
            # (MỚI) Trả về cả ID
            "lanes_config": [{
                "id": ln.get('id'), 
                "name": ln.get('name'),
                "sensor_pin": ln.get('sensor_pin'),
                "push_pin": ln.get('push_pin'),
                "pull_pin": ln.get('pull_pin')
             } for ln in system_state.get('lanes', [])]
        }
    return jsonify(config_data)

@app.route('/update_config', methods=['POST'])
@requires_auth
def update_config():
    """API (POST) để cập nhật config (timing + lanes)."""
    global lanes_config, RELAY_PINS, SENSOR_PINS
    # (NÂNG CẤP) Thêm biến
    global pending_sensor_triggers


    new_config_data = request.json
    if not new_config_data:
        return jsonify({"error": "Thiếu dữ liệu JSON"}), 400

    logging.info(f"[CONFIG] Nhận config mới từ API (POST): {new_config_data}")

    new_timing_config = new_config_data.get('timing_config', {})
    new_lanes_config = new_config_data.get('lanes_config')

    config_to_save = {}
    restart_required = False

    with state_lock:
        # 1. Cập nhật Timing Config
        current_timing = system_state['timing_config']
        current_gpio_mode = current_timing.get('gpio_mode', 'BCM')
        timing_changed = False
        
        temp_timing = current_timing.copy()
        temp_timing.update(new_timing_config)
        if any(temp_timing.get(k) != current_timing.get(k) for k in new_timing_config if k != 'gpio_mode'):
            timing_changed = True
        
        current_timing.update({k:v for k,v in new_timing_config.items() if k != 'gpio_mode'})
        
        new_gpio_mode = new_timing_config.get('gpio_mode', current_gpio_mode)

        if new_gpio_mode != current_gpio_mode:
            logging.warning("[CONFIG] Chế độ GPIO đã thay đổi. Cần khởi động lại ứng dụng để áp dụng.")
            broadcast_log({"log_type": "warn", "message": "GPIO Mode đã đổi. Cần khởi động lại!"})
            restart_required = True
            config_to_save['timing_config'] = current_timing.copy()
            config_to_save['timing_config']['gpio_mode'] = new_gpio_mode
        elif timing_changed:
            config_to_save['timing_config'] = current_timing.copy()
        else:
            config_to_save['timing_config'] = current_timing.copy()


        # 2. Cập nhật Lanes Config (nếu có gửi)
        if new_lanes_config is not None:
            logging.info("[CONFIG] Cập nhật cấu hình lanes...")
            
            # (MỚI) Đảm bảo tất cả các lane có ID trước khi cập nhật
            lanes_config = ensure_lane_ids(new_lanes_config)
            num_lanes = len(lanes_config)
            
            new_system_lanes = []
            new_relay_pins = []
            new_sensor_pins = []
            for i, lane_cfg in enumerate(lanes_config):
                lane_name = lane_cfg.get("name", f"Lane {i+1}")
                lane_id = lane_cfg.get("id") # ID phải tồn tại sau khi gọi ensure_lane_ids

                new_system_lanes.append({
                    "name": lane_name,
                    "id": lane_id,
                    "status": "Sẵn sàng", "count": 0, 
                    "sensor_pin": lane_cfg.get("sensor_pin"),
                    "push_pin": lane_cfg.get("push_pin"),
                    "pull_pin": lane_cfg.get("pull_pin"),
                    "sensor_reading": 1, "relay_grab": 0, "relay_push": 0
                })
                
                if lane_cfg.get("sensor_pin") is not None: new_sensor_pins.append(lane_cfg["sensor_pin"])
                if lane_cfg.get("push_pin") is not None: new_relay_pins.append(lane_cfg["push_pin"])
                if lane_cfg.get("pull_pin") is not None: new_relay_pins.append(lane_cfg["pull_pin"])

            system_state['lanes'] = new_system_lanes
            
            global last_sensor_state, last_sensor_trigger_time, auto_test_last_state, auto_test_last_trigger
            last_sensor_state = [1] * num_lanes
            last_sensor_trigger_time = [0.0] * num_lanes
            auto_test_last_state = [1] * num_lanes
            auto_test_last_trigger = [0.0] * num_lanes
            # (NÂNG CẤP) Reset pending triggers
            pending_sensor_triggers = [0.0] * num_lanes


            RELAY_PINS = new_relay_pins
            SENSOR_PINS = new_sensor_pins

            config_to_save['lanes_config'] = lanes_config
            restart_required = True
            logging.warning("[CONFIG] Cấu hình lanes đã thay đổi. Cần khởi động lại ứng dụng.")
            broadcast_log({"log_type": "warn", "message": "Cấu hình Lanes đã đổi. Cần khởi động lại!"})
        else:
            current_lanes_cfg_for_save = []
            for lane_state in system_state['lanes']:
                # Lưu lại ID cố định khi lưu file
                current_lanes_cfg_for_save.append({
                    "id": lane_state.get('id'),
                    "name": lane_state['name'],
                    "sensor_pin": lane_state.get('sensor_pin'),
                    "push_pin": lane_state.get('push_pin'),
                    "pull_pin": lane_state.get('pull_pin')
                })
            config_to_save['lanes_config'] = current_lanes_cfg_for_save
            
        if 'gpio_mode' in new_timing_config:
            config_to_save['timing_config']['gpio_mode'] = new_timing_config['gpio_mode']


    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            old_config_data = json.load(f)
    except Exception:
        old_config_data = {}
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            old_config_data = json.load(f)
    except Exception:
        old_config_data = {}

    try:
        # (SỬA LỖI ENCODING) Buộc dùng UTF-8 khi ghi
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config_to_save, f, indent=4)

        msg = "Đã lưu config mới. Vui lòng khởi động lại hệ thống để áp dụng thay đổi."
        logging.info("[CONFIG] Cấu hình mới đã được lưu (Hot Reload bị tắt).")
        broadcast_log({
            "log_type": "info",
            "message": "Đã lưu config. Hãy restart hệ thống để áp dụng thay đổi."
        })

        return jsonify({
            "message": msg,
            "config": config_to_save,
            "restart_required": True
        })

    except Exception as e:
        logging.error(f"[ERROR] Không thể lưu config (POST): {e}")
        broadcast_log({"log_type": "error", "message": f"Lỗi khi lưu config (POST): {e}"})
        return jsonify({"error": str(e)}), 500





@app.route('/api/reset_maintenance', methods=['POST'])
@requires_auth
def reset_maintenance():
    """API (POST) để reset chế độ bảo trì."""
    # (NÂNG CẤP) Thêm biến
    global pending_sensor_triggers, queue_head_since

    if error_manager.is_maintenance():
        error_manager.reset()
        with queue_lock:
            qr_queue.clear()
            queue_head_since = 0.0
            # (NÂNG CẤP) Reset pending triggers
            pending_sensor_triggers = [0.0] * len(pending_sensor_triggers)

        broadcast_log({"log_type": "success", "message": "Chế độ bảo trì đã được reset từ UI. Hàng chờ đã được xóa."})
        return jsonify({"message": "Maintenance mode reset thành công."})
    else:
        return jsonify({"message": "Hệ thống không ở chế độ bảo trì."})

@app.route('/api/queue/reset', methods=['POST'])
@requires_auth
def api_queue_reset():
    """API (POST) để xóa hàng chờ QR."""
    # (NÂNG CẤP) Thêm biến
    global pending_sensor_triggers, queue_head_since

    if error_manager.is_maintenance():
        return jsonify({"error": "Hệ thống đang bảo trì, không thể reset hàng chờ."}), 403

    try:
        with queue_lock:
            qr_queue.clear()
            queue_head_since = 0.0
            current_queue_for_log = list(qr_queue)
            # (NÂNG CẤP) Reset pending triggers
            pending_sensor_triggers = [0.0] * len(pending_sensor_triggers)


        with state_lock:
            for lane in system_state["lanes"]:
                lane["status"] = "Sẵn sàng"
            # (FIX LỖI ĐỒNG BỘ) Cập nhật queue indices vào state
            system_state["queue_indices"] = current_queue_for_log

        broadcast_log({
            "log_type": "warn",
            "message": "Hàng chờ QR đã được reset thủ công từ UI.",
            "queue": current_queue_for_log
        })
        logging.info("[API] Hàng chờ QR đã được reset thủ công.")
        return jsonify({"message": "Hàng chờ đã được reset."})
    except Exception as e:
        logging.error(f"[API] Lỗi khi reset hàng chờ: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/mock_gpio', methods=['POST'])
@requires_auth
def api_mock_gpio():
    """API (POST) để điều khiển sensor ở chế độ giả lập."""
    if not isinstance(GPIO, MockGPIO):
        return jsonify({"error": "Chức năng chỉ khả dụng ở chế độ mô phỏng."}), 400

    payload = request.get_json(silent=True) or {}
    lane_index = payload.get('lane_index')
    pin = payload.get('pin')
    requested_state = payload.get('state')

    if lane_index is not None and pin is None:
        try:
            lane_index = int(lane_index)
        except (TypeError, ValueError):
            return jsonify({"error": "lane_index không hợp lệ."}), 400
        with state_lock:
            if 0 <= lane_index < len(system_state['lanes']):
                pin = system_state['lanes'][lane_index].get('sensor_pin')
            else:
                return jsonify({"error": "lane_index vượt ngoài phạm vi."}), 400

    if pin is None:
        return jsonify({"error": "Thiếu thông tin chân sensor."}), 400

    try:
        pin = int(pin)
    except (TypeError, ValueError):
        return jsonify({"error": "Giá trị pin không hợp lệ."}), 400

    if pin is None:
        return jsonify({"error": "Không thể mô phỏng sensor cho Lane không có chân cắm."}), 400


    if requested_state is None:
        logical_state = GPIO.toggle_input_state(pin)
    else:
        logical_state = 1 if str(requested_state).strip().lower() in {"1", "true", "high", "inactive"} else 0
        GPIO.set_input_state(pin, logical_state)

    lane_name = None
    with state_lock:
        for lane in system_state['lanes']:
            if lane.get('sensor_pin') == pin:
                lane['sensor_reading'] = 0 if logical_state == 0 else 1
                lane_name = lane.get('name', lane_name)

    state_label = 'ACTIVE (LOW)' if logical_state == 0 else 'INACTIVE (HIGH)'
    message = f"[MOCK] Sensor pin {pin} -> {state_label}"
    if lane_name:
        message += f" ({lane_name})"
    broadcast_log({
        "log_type": "info",
        "message": message
    })
    return jsonify({"pin": pin, "state": logical_state, "lane": lane_name})
@sock.route('/ws')
@requires_auth
def ws_route(ws):
    """Kết nối WebSocket chính."""
    global AUTO_TEST_ENABLED

    auth = request.authorization if AUTH_ENABLED else None
    if AUTH_ENABLED and (not auth or not check_auth(auth.username, auth.password)):
        logging.warning("[WS] Unauthorized connection attempt.")
        ws.close(code=1008, reason="Unauthorized")
        return

    client_label = auth.username if auth else f"guest-{id(ws):x}"
    _add_client(ws)
    logging.info(f"[WS] Client {client_label} connected. Total: {len(_list_clients())}")

    try:
        with state_lock:
            system_state["maintenance_mode"] = error_manager.is_maintenance()
            system_state["last_error"] = error_manager.last_error
            system_state["auth_enabled"] = AUTH_ENABLED
            initial_state_msg = json.dumps({"type": "state_update", "state": system_state})
        ws.send(initial_state_msg)
    except Exception as e:
        logging.warning(f"[WS] Lỗi gửi state ban đầu: {e}")
        _remove_client(ws)
        return

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
                        # (NÂNG CẤP) Thêm biến
                        global pending_sensor_triggers, queue_head_since
                        if error_manager.is_maintenance():
                            error_manager.reset()
                            with queue_lock:
                                qr_queue.clear()
                                queue_head_since = 0.0
                                # (NÂNG CẤP) Reset pending triggers
                                pending_sensor_triggers = [0.0] * len(pending_sensor_triggers)

                            
                            # (FIX LỖI ĐỒNG BỘ) Cập nhật queue indices vào state
                            with state_lock:
                                system_state["queue_indices"] = []
                                
                            broadcast_log({"log_type": "success", "message": f"Chế độ bảo trì đã được reset bởi {client_label}. Hàng chờ đã được xóa."})
                        else:
                            broadcast_log({"log_type": "info", "message": "Hệ thống không ở chế độ bảo trì."})


                except json.JSONDecodeError:
                    pass
                except Exception as ws_loop_e:
                    logging.error(f"[WS] Lỗi xử lý message: {ws_loop_e}")

    except Exception as ws_conn_e:
        logging.warning(f"[WS] Kết nối WebSocket bị đóng hoặc lỗi: {ws_conn_e}")
    finally:
        _remove_client(ws)
        logging.info(f"[WS] Client {client_label} disconnected. Total: {len(_list_clients())}")
# =============================
#     (MỚI) HÀM KHỞI ĐỘNG THREAD
# =============================
# def start_background_threads():
    #"""Khởi động toàn bộ luồng nền."""
    # threading.Thread(target=camera_capture_thread, name="CameraThread", daemon=True).start()
    #threading.Thread(target=qr_detection_loop, name="QRThread", daemon=True).start()
    #threading.Thread(target=sensor_monitoring_thread, name="SensorThread", daemon=True).start()
    #threading.Thread(target=broadcast_state, name="BroadcastThread", daemon=True).start()
    #threading.Thread(target=periodic_config_save, name="ConfigSaveThread", daemon=True).start()
        
# =============================
#             MAIN
# =============================
# (NÂNG CẤP) Khởi tạo biến global ở top-level
pending_sensor_triggers = []

if __name__ == "__main__":
    try:
        # Setup logging
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
        threading.Thread(target=sensor_monitoring_thread, name="SensorThread", daemon=True).start()
        threading.Thread(target=broadcast_state, name="BroadcastThread", daemon=True).start()
       ## threading.Thread(target=auto_test_loop, name="AutoTestThread", daemon=True).start()
        threading.Thread(target=periodic_config_save, name="ConfigSaveThread", daemon=True).start()


        logging.info("=========================================")
        logging.info("  HỆ THỐNG PHÂN LOẠI SẴN SÀNG (V2.0 - Sensor-First)")
        logging.info(f"  GPIO Mode: {'REAL' if isinstance(GPIO, RealGPIO) else 'MOCK'} (Config: {loaded_gpio_mode})")
        logging.info(f"  Log file: {LOG_FILE}")
        logging.info(f"  Sort log file: {SORT_LOG_FILE}")
        logging.info(f"  API State: http://<IP_CUA_PI>:3000 (Đã đổi port 3000)")
        if AUTH_ENABLED:
            logging.info(f"  Truy cập: http://<IP_CUA_PI>:3000 (User: {USERNAME} / Pass: {PASSWORD})")
        else:
            logging.info("  Truy cập: http://<IP_CUA_PI>:3000 (không yêu cầu đăng nhập)")
        logging.info("=========================================")
        

        # Chạy Flask server (Đổi port 3000 cho an toàn)
        app.run(host='0.0.0.0', port=3000)

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

