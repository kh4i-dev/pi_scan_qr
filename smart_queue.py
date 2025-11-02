# =============================
#     LỚP SMART QUEUE MỚI
# =============================
MAX_RETRIES = 3
QUEUE_WARNING_THRESHOLD = 5
QUEUE_AGE_WARNING = 30.0 # Cảnh báo nếu item cũ hơn 30s
MONITOR_INTERVAL = 5.0 # Kiểm tra queue mỗi 5s

class SmartQueue:
    def __init__(self, name, max_size=100):
        self.name = name
        self.items = []
        self.max_size = max_size
        self.lock = threading.Lock()
        self.history = []  # Lưu lịch sử
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
            
            # Chèn theo priority (priority cao hơn nằm ở đầu)
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
            
            # Tìm item khớp (dựa trên lane_index hoặc qr_key)
            # Logic: Tìm item khớp với lane/key đầu tiên trong hàng chờ
            pop_index = -1
            
            if lane_index_to_match is not None:
                pop_index = next((i for i, item in enumerate(self.items) 
                                if item.get('lane_index') == lane_index_to_match), -1)
            elif qr_key_to_match is not None:
                 pop_index = next((i for i, item in enumerate(self.items) 
                                if item.get('qr_key') == qr_key_to_match), -1)
            else:
                pop_index = 0 # Mặc định pop đầu tiên nếu không có điều kiện
                
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
        """Xóa item hết hạn và trả về nó để xử lý retry."""
        with self.lock:
            if not self.items: return None
            now = time.time()
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
        # Giữ lại item quan trọng (priority cao) hoặc item chưa quá cũ
        self.items = [item for item in self.items 
                     if (time.time() - item["timestamp"]) <= QUEUE_AGE_WARNING 
                     or item['priority'] > 0]
                     
    def _find_insert_position(self, priority):
        # Tìm vị trí chèn dựa trên priority (priority cao hơn chèn trước)
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
            return self.history[-100:] # Lấy 100 item gần nhất
