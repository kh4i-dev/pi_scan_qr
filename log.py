# -*- coding: utf-8 -*-
import cv2, time, threading, RPi.GPIO as GPIO

# =============================
# CẤU HÌNH HỆ THỐNG
# =============================
CAMERA_INDEX = 0
ACTIVE_LOW = True
CYCLE_DELAY = 0.3      # thời gian piston đẩy
SETTLE_DELAY = 0.2     # chờ giữa 2 hành động
SENSOR_DEBOUNCE = 0.1  # chống nhiễu sensor
PUSH_DELAY = 0.0        # trễ giữa khi phát hiện vật và đẩy
QUEUE_TIMEOUT = 10.0   # timeout hàng chờ

LANES = [
    {"id": "A", "sensor": 3, "push": 17, "pull": 18},
    {"id": "B", "sensor": 23, "push": 27, "pull": 14},
    {"id": "C", "sensor": 24, "push": 22, "pull": 4},
    {"id": "D", "sensor": None, "push": None, "pull": None},  # đi thẳng
]

# =============================
# GPIO SETUP
# =============================
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
for l in LANES:
    if l["push"]: GPIO.setup(l["push"], GPIO.OUT)
    if l["pull"]: GPIO.setup(l["pull"], GPIO.OUT)
    if l["sensor"]: GPIO.setup(l["sensor"], GPIO.IN, pull_up_down=GPIO.PUD_UP)

def RELAY_ON(p): GPIO.output(p, GPIO.LOW if ACTIVE_LOW else GPIO.HIGH)
def RELAY_OFF(p): GPIO.output(p, GPIO.HIGH if ACTIVE_LOW else GPIO.LOW)
def reset_relays():
    for l in LANES:
        if l["pull"]: RELAY_ON(l["pull"])
        if l["push"]: RELAY_OFF(l["push"])
    print("[GPIO] ✅ Reset tất cả relay (THU BẬT).")

# =============================
# HÀNG CHỜ VÀ CỜ TRẠNG THÁI
# =============================
qr_queue, queue_lock = [], threading.Lock()
main_loop, head_time = True, 0.0

# =============================
# CHU TRÌNH ĐẨY
# =============================
def do_push(lane):
    if not lane["push"] and not lane["pull"]:
        print(f"[{lane['id']}] 🚗 Đi thẳng.")
        return
    print(f"[{lane['id']}] 🚀 Bắt đầu đẩy...")
    time.sleep(PUSH_DELAY)
    RELAY_OFF(lane["pull"])
    time.sleep(SETTLE_DELAY)
    RELAY_ON(lane["push"])
    time.sleep(CYCLE_DELAY)
    RELAY_OFF(lane["push"])
    time.sleep(SETTLE_DELAY)
    RELAY_ON(lane["pull"])
    print(f"[{lane['id']}] ✅ Hoàn tất.")

# =============================
# QUÉT QR
# =============================
def qr_loop():
    global head_time
    cap, detector = cv2.VideoCapture(CAMERA_INDEX), cv2.QRCodeDetector()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    last, t0 = "", 0

    if not cap.isOpened():
        print("[CAMERA] ❌ Không mở được camera.")
        return

    while main_loop:
        ret, frame = cap.read()
        if not ret: time.sleep(0.1); continue
        data, _, _ = detector.detectAndDecode(frame)
        if data and (data != last or time.time() - t0 > 2):
            last, t0 = data, time.time()
            lane = next((l for l in LANES if l["id"] == data), None)
            if lane:
                with queue_lock:
                    qr_queue.append(lane)
                    head_time = head_time or time.time()
                    print(f"[QR] 📦 {data} → hàng chờ: {[x['id'] for x in qr_queue]}")
            else:
                print(f"[QR] ⚠️ Không khớp ID: {data}")
        time.sleep(0.05)
    cap.release()

# =============================
# GIÁM SÁT SENSOR
# =============================
def sensor_loop():
    global head_time
    prev = {}
    while main_loop:
        now = time.time()
        with queue_lock:
            if qr_queue and now - head_time > QUEUE_TIMEOUT:
                drop = qr_queue.pop(0)
                print(f"[QUEUE] ⏰ Timeout bỏ {drop['id']}.")
                head_time = time.time() if qr_queue else 0

        for l in LANES:
            sp = l["sensor"]
            if not sp: continue
            val = GPIO.input(sp)
            if val == 0 and prev.get(sp, 1) == 1:
                with queue_lock:
                    if qr_queue:
                        target = qr_queue[0]
                        if target["sensor"] == sp:
                            qr_queue.pop(0)
                            print(f"[SENSOR] ✅ {l['id']} kích đúng hàng chờ.")
                            threading.Thread(target=do_push, args=(l,), daemon=True).start()
                            head_time = time.time() if qr_queue else 0
                        else:
                            print(f"[SENSOR] ⚠️ {l['id']} kích sai hàng chờ.")
                    else:
                        print(f"[SENSOR] ⚠️ {l['id']} kích khi trống.")
            prev[sp] = val
        time.sleep(0.02)

# =============================
# MAIN
# =============================
if __name__ == "__main__":
    try:
        reset_relays()
        print("=================================")
        print(" ⚙️ HỆ THỐNG PHÂN LOẠI HÀNG CHỜ")
        print(" 🚀 Bắt đầu chạy (bản ultra tối giản)")
        print("=================================")
        threading.Thread(target=qr_loop, daemon=True).start()
        threading.Thread(target=sensor_loop, daemon=True).start()
        while True: time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Dừng hệ thống (Ctrl+C)")
    finally:
        main_loop = False
        GPIO.cleanup()
        print("✅ GPIO cleaned up.")
