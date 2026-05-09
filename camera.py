"""
摄像头线程管理模块
在独立线程中采集摄像头画面，线程安全地提供最新帧
"""

import cv2
import threading
import time
import os

# 屏蔽 OpenCV DShow 后端在检测不存在摄像头时的警告
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"


class CameraThread:
    """摄像头采集线程 — 仅负责采集，不做任何处理"""

    def __init__(self, camera_id=0, width=480, height=360, fps=30):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.target_fps = fps
        self.cap = None
        self.running = False
        self.frame = None
        self.lock = threading.Lock()
        self.thread = None
        self._actual_fps = 0.0
        self._fps_lock = threading.Lock()

    @property
    def actual_fps(self):
        with self._fps_lock:
            return self._actual_fps

    def start(self):
        """启动摄像头"""
        if self.running:
            return

        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        # 降低缓冲区大小，减少延迟
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 (ID={self.camera_id})")

        actual_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"[Camera] 实际分辨率: {int(actual_w)}x{int(actual_h)}")

        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        """停止摄像头"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        self.cap = None

    def get_frame(self):
        """
        线程安全获取最新帧
        返回 None 如果没有可用帧
        """
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def _loop(self):
        """主循环 — 纯采集，极简"""
        interval = 1.0 / max(self.target_fps, 1)
        last_time = time.time()
        fps_counter = 0
        fps_timer = time.time()

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.005)
                continue

            # 镜像翻转（更自然）
            frame = cv2.flip(frame, 1)

            with self.lock:
                self.frame = frame

            # FPS 统计
            fps_counter += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                with self._fps_lock:
                    self._actual_fps = fps_counter / (now - fps_timer)
                fps_counter = 0
                fps_timer = now

            # 控制采集帧率
            elapsed = now - last_time
            if elapsed < interval:
                time.sleep(interval - elapsed)
            last_time = time.time()

    @staticmethod
    def list_cameras(max_test=5):
        """列出可用摄像头（忽略 OpenCV DShow 警告）"""
        # 临时屏蔽 OpenCV 日志输出
        try:
            cv_log_level = cv2.utils.logging.getLogLevel()
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LEVEL_ERROR)
        except Exception:
            cv_log_level = None

        available = []
        for i in range(max_test):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap.isOpened():
                    available.append(i)
                    cap.release()
            except Exception:
                pass

        # 恢复日志级别
        if cv_log_level is not None:
            try:
                cv2.utils.logging.setLogLevel(cv_log_level)
            except Exception:
                pass
        return available
