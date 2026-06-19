"""
FaceVision — PyQt6 Windows 11 界面
Mica 背景 · WinUI 3 设计令牌 · Segoe UI Variable · Fluent Design System
"""

import sys
import os
import uuid
import time
import ctypes
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QScrollArea,
    QFrame, QLineEdit, QSlider, QCheckBox, QDialog, QComboBox,
    QMessageBox, QSizePolicy, QFileDialog
)
from PyQt6.QtCore import (
    Qt, QTimer, QObject, pyqtSignal, QRectF, QRegularExpression, QSize, QPoint
)
from PyQt6.QtGui import (
    QPainter, QPen, QBrush, QColor, QPixmap, QImage, QFont,
    QFontDatabase, QPainterPath, QLinearGradient, QIcon,
    QRegularExpressionValidator, QAction
)

from face_database import FaceDatabase
from face_tracker import FaceTracker
from config import APP_SETTINGS, save_settings

# ═══════════════════════════════════════════════════════════════
# WinUI 3 设计令牌
# ═══════════════════════════════════════════════════════════════

ACCENT = "#60CDFF"
ACCENT_HOVER = "#8AD8FF"
ACCENT_PRESS = "#4DB8EC"
ACCENT_SECONDARY = "rgba(96, 205, 255, 0.10)"
ACCENT_TERTIARY = "rgba(96, 205, 255, 0.06)"

DARK_BG = "#202020"
DARK_SURFACE = "#2A2A2A"
DARK_SURFACE_HOVER = "#3A3A3A"
DARK_BORDER = "rgba(255, 255, 255, 0.08)"
DARK_BORDER_HOVER = "rgba(255, 255, 255, 0.12)"
DARK_TEXT = "#FFFFFF"
DARK_TEXT_SECONDARY = "#FFFFFF"

SUCCESS_COLOR = "#FFFFFF"
DANGER_COLOR = "#C42B1C"

CARD_RADIUS = 8
BUTTON_RADIUS = 6
FONT_FAMILY = "Segoe UI Variable"

# 始终使用深色主题
IS_DARK = True

# ═══════════════════════════════════════════════════════════════
# Mica 背景 (ctypes → DWM)
# ═══════════════════════════════════════════════════════════════

def apply_mica(hwnd):
    """对窗口应用 Windows 11 Mica 材质（深色模式）"""
    try:
        DWMWA_SYSTEMBACKDROP_TYPE = 38  # Windows 11 22H2+
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            int(hwnd), DWMWA_SYSTEMBACKDROP_TYPE,
            ctypes.byref(ctypes.c_int(2)),  # DWMSBT_MAINWINDOW (Mica)
            ctypes.sizeof(ctypes.c_int)
        )
        # 深色标题栏
        DWMWA_USE_IMMERSIVE_DARK_MODE = 20
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            int(hwnd), DWMWA_USE_IMMERSIVE_DARK_MODE,
            ctypes.byref(ctypes.c_int(1)),
            ctypes.sizeof(ctypes.c_int)
        )
    except Exception:
        pass

# ═══════════════════════════════════════════════════════════════
# 处理线程
# ═══════════════════════════════════════════════════════════════

class ProcessingThread(QObject):
    """ML 处理线程 — 异步运行检测+识别，结果通过信号发送到主线程"""
    frame_ready = pyqtSignal(np.ndarray, list, list)
    status_update = pyqtSignal(str, str)

    def __init__(self, camera, detector, recognizer, tracker, db):
        super().__init__()
        self.camera = camera
        self.detector = detector
        self.recognizer = recognizer
        self.tracker = tracker
        self.db = db
        self._running = False
        self._paused = False
        self._lock = threading.Lock()

    def start(self):
        self._running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self._running = False

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False
        if hasattr(self, 'tracker') and self.tracker:
            self.tracker.reset()

    def _run(self):
        proc_fps = APP_SETTINGS.get("proc_fps", 30)
        frame_interval = 1.0 / proc_fps if proc_fps > 0 else 0
        last_time = 0
        frame_count = 0

        while self._running:
            try:
                if self._paused:
                    time.sleep(0.05)
                    continue

                if frame_interval > 0:
                    elapsed = time.time() - last_time
                    if elapsed < frame_interval:
                        time.sleep(min(frame_interval - elapsed, 0.01))

                frame = self.camera.get_frame()
                last_time = time.time()
                if frame is None:
                    time.sleep(0.01)
                    continue

                frame_count += 1

                # 更新编码缓存
                known_encodings, known_names = self.db.get_encodings_and_names()
                self.recognizer.update_cache(known_encodings, known_names, self.db.version)

                # 检测 + 特征提取
                faces_with_emb = self.detector.detect_with_embeddings(frame)

                # 时序追踪 + 识别（tracker 内部调用 recognizer）
                try:
                    tracked_faces = self.tracker.update(
                        faces_with_emb, recognizer=self.recognizer
                    )
                except Exception:
                    import traceback
                    traceback.print_exc()
                    tracked_faces = []

                # 构建识别结果列表供 UI 绘制
                recognized = []
                for tf in tracked_faces:
                    x1, y1, x2, y2 = tf['bbox']
                    name = tf['name']
                    conf = tf['conf']
                    recognized.append((x1, y1, x2, y2, conf, name, conf))

                if frame_count % 30 == 0:
                    n_valid = sum(1 for f in faces_with_emb if f[6])
                    print(f"[ProcessingThread] 帧#{frame_count}: "
                          f"检测到 {len(faces_with_emb)} 人脸, "
                          f"{n_valid} 通过质量检查, "
                          f"追踪数={self.tracker.track_count}, "
                          f"帧shape={frame.shape}")

                self.frame_ready.emit(frame, recognized, tracked_faces)

            except Exception:
                import traceback
                print(f"[ProcessingThread] Unhandled error in loop:")
                traceback.print_exc()
                time.sleep(1)

# ═══════════════════════════════════════════════════════════════
# 样式表生成器
# ═══════════════════════════════════════════════════════════════

def make_stylesheet():
    """生成完整的深色 QSS 样式表"""
    bg = DARK_BG
    surface = DARK_SURFACE
    surface_hover = DARK_SURFACE_HOVER
    border = DARK_BORDER
    text = DARK_TEXT
    text_secondary = DARK_TEXT_SECONDARY

    return f"""
        * {{
            font-family: "{FONT_FAMILY}", "Microsoft YaHei UI", sans-serif;
            font-size: 14px;
        }}

        QMainWindow {{
            background: transparent;
        }}

        QDialog {{
            background-color: {bg};
        }}

        #centralWidget {{
            background: transparent;
        }}

        /* ── 卡片 ── */
        QFrame#card {{
            background: {surface};
            border: 1px solid {border};
            border-radius: {CARD_RADIUS}px;
        }}

        /* ── 按钮 ── */
        QPushButton {{
            background: {surface};
            border: 1px solid {border};
            border-radius: {BUTTON_RADIUS}px;
            color: {text};
            padding: 8px 16px;
            font-size: 14px;
        }}
        QPushButton:hover {{
            background: {surface_hover};
            border-color: {DARK_BORDER_HOVER};
        }}
        QPushButton:pressed {{
            background: rgba(255,255,255,0.08);
        }}
        QPushButton:disabled {{
            color: {text_secondary};
            background: transparent;
            border-color: {border};
        }}

        QPushButton#btnAccent {{
            background: {ACCENT};
            border: none;
            color: #FFFFFF;
            font-weight: 600;
        }}
        QPushButton#btnAccent:hover {{
            background: {ACCENT_HOVER};
        }}
        QPushButton#btnAccent:pressed {{
            background: {ACCENT_PRESS};
        }}

        QPushButton#btnDanger {{
            background: {DANGER_COLOR};
            border: none;
            color: #FFFFFF;
            font-weight: 600;
        }}
        QPushButton#btnDanger:hover {{
            background: #E0432D;
        }}

        QPushButton#btnGpuActive {{
            background: #60CDFF;
            color: #FFFFFF;
            border: none;
            font-weight: 600;
            border-radius: {BUTTON_RADIUS}px;
        }}
        QPushButton#btnCpuActive {{
            background: rgba(255,255,255,0.12);
            color: #FFFFFF;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: {BUTTON_RADIUS}px;
        }}

        /* ── 标签 ── */
        QLabel {{
            color: {text};
            background: transparent;
        }}
        QLabel#sectionLabel {{
            color: {text_secondary};
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-top: 8px;
        }}

        /* ── 输入框 ── */
        QLineEdit {{
            background: {surface};
            border: 1px solid {border};
            border-radius: {BUTTON_RADIUS}px;
            color: {text};
            padding: 8px 12px;
            font-size: 14px;
        }}
        QLineEdit:focus {{
            border-color: {ACCENT};
        }}
        QLineEdit::placeholder {{
            color: {text_secondary};
        }}

        /* ── 滑块 ── */
        QSlider::groove:horizontal {{
            background: {surface};
            border-radius: 3px;
            height: 6px;
        }}
        QSlider::handle:horizontal {{
            background: {ACCENT};
            width: 20px;
            height: 20px;
            margin: -7px 0;
            border-radius: 10px;
        }}
        QSlider::handle:horizontal:hover {{
            background: {ACCENT_HOVER};
        }}
        QSlider::sub-page:horizontal {{
            background: {ACCENT};
            border-radius: 3px;
        }}

        /* ── 复选框 ── */
        QCheckBox {{
            color: {text};
            spacing: 8px;
        }}
        QCheckBox::indicator {{
            width: 20px;
            height: 20px;
            border-radius: 4px;
            border: 2px solid {border};
            background: transparent;
        }}
        QCheckBox::indicator:checked {{
            background: {ACCENT};
            border-color: {ACCENT};
        }}

        /* ── 滚动条 ── */
        QScrollArea {{
            background: transparent;
            border: none;
        }}
        QScrollBar:vertical {{
            background: transparent;
            width: 6px;
            margin: 0;
        }}
        QScrollBar::handle:vertical {{
            background: {DARK_BORDER_HOVER};
            border-radius: 3px;
            min-height: 30px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: rgba(255,255,255,0.20);
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0;
        }}

        /* ── 下拉框 ── */
        QComboBox {{
            color: #FFFFFF;
            background: {surface};
            border: 1px solid {border};
            border-radius: {BUTTON_RADIUS}px;
            padding: 6px 12px;
            font-size: 14px;
            min-height: 32px;
        }}
        QComboBox:hover {{
            border-color: rgba(255,255,255,0.20);
        }}
        QComboBox QAbstractItemView {{
            color: #FFFFFF;
            background: #2D2D2D;
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 6px;
            selection-background-color: {ACCENT};
            selection-color: #FFFFFF;
            outline: none;
            padding: 4px;
        }}
    """

# ═══════════════════════════════════════════════════════════════
# 玻璃态对话框基类
# ═══════════════════════════════════════════════════════════════

class GlassDialog(QDialog):
    """基础对话框 — 与主窗口统一风格，支持拖动"""

    def __init__(self, parent=None, title=""):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        self.setModal(True)
        self.setMinimumSize(380, 200)
        self._drag_pos = None

        # 应用深色样式表
        self.setStyleSheet("""
            QDialog {
                background-color: #1F1F1F;
            }
        """)

        if parent:
            self.resize(400, 220)
            parent_geo = parent.geometry()
            x = parent_geo.center().x() - 200
            y = parent_geo.center().y() - 110
            self.move(x, y)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton and self._drag_pos is not None:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        self._drag_pos = None
        event.accept()

# ═══════════════════════════════════════════════════════════════
# 对话框
# ═══════════════════════════════════════════════════════════════

class AddPersonDialogPyQt(GlassDialog):
    """添加人员对话框"""

    def __init__(self, parent=None, title="添加人员", prompt="请输入人员姓名："):
        super().__init__(parent, title)
        self.result = None
        self.setMinimumSize(380, 200)
        self.resize(400, 200)

        layout = QVBoxLayout()
        layout.setContentsMargins(28, 28, 28, 24)
        layout.setSpacing(16)

        label = QLabel(prompt)
        label.setStyleSheet(
            "color: #FFFFFF; font-size: 15px; background: transparent;"
        )
        layout.addWidget(label)

        self.entry = QLineEdit()
        self.entry.setPlaceholderText("输入姓名…")
        self.entry.setMinimumHeight(38)
        self.entry.returnPressed.connect(self._confirm)
        layout.addWidget(self.entry)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)
        btn_layout.addStretch()

        cancel_btn = QPushButton("取消")
        cancel_btn.setFixedSize(100, 38)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        confirm_btn = QPushButton("确认")
        confirm_btn.setObjectName("btnAccent")
        confirm_btn.setFixedSize(100, 38)
        confirm_btn.clicked.connect(self._confirm)
        btn_layout.addWidget(confirm_btn)

        layout.addLayout(btn_layout)
        self.setLayout(layout)
        self.entry.setFocus()

    def _confirm(self):
        name = self.entry.text().strip()
        if name:
            self.result = name
            self.accept()
        else:
            QMessageBox.warning(self, "输入无效", "请输入有效的姓名。")

    def get_result(self):
        return self.result


class BatchDeleteDialogPyQt(GlassDialog):
    """批量删除对话框"""

    def __init__(self, parent=None, names=None):
        super().__init__(parent, "批量删除人员")
        self.result = []
        self.names = names or []
        self.setMinimumSize(400, 420)
        self.resize(420, 440)

        layout = QVBoxLayout()
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)

        label = QLabel("勾选要删除的人员：")
        label.setStyleSheet(
            "color: #FFFFFF; "
            f"font-size: 15px; font-weight: bold; background: transparent;"
        )
        layout.addWidget(label)

        select_btn_layout = QHBoxLayout()
        select_all = QPushButton("全选")
        select_all.setFixedSize(85, 32)
        select_all.clicked.connect(self._select_all)
        select_btn_layout.addWidget(select_all)

        deselect_all = QPushButton("全不选")
        deselect_all.setFixedSize(85, 32)
        deselect_all.clicked.connect(self._deselect_all)
        select_btn_layout.addWidget(deselect_all)
        select_btn_layout.addStretch()
        layout.addLayout(select_btn_layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        scroll_widget = QWidget()
        scroll_widget.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        scroll_widget.setStyleSheet("background: transparent;")
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(4)

        self.checkboxes = {}
        for name in sorted(self.names):
            cb = QCheckBox(f"  {name}")
            self.checkboxes[name] = cb
            scroll_layout.addWidget(cb)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll, 1)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QPushButton("取消")
        cancel_btn.setFixedSize(100, 38)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        delete_btn = QPushButton("删除选中")
        delete_btn.setObjectName("btnDanger")
        delete_btn.setFixedSize(120, 38)
        delete_btn.clicked.connect(self._confirm)
        btn_layout.addWidget(delete_btn)

        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def _select_all(self):
        for cb in self.checkboxes.values():
            cb.setChecked(True)

    def _deselect_all(self):
        for cb in self.checkboxes.values():
            cb.setChecked(False)

    def _confirm(self):
        selected = [name for name, cb in self.checkboxes.items() if cb.isChecked()]
        if not selected:
            QMessageBox.warning(self, "未选择", "请至少选择一名人员。")
            return
        self.result = selected
        self.accept()

    def get_result(self):
        return self.result


class SelectFaceDialogPyQt(GlassDialog):
    """多人脸选择对话框"""

    def __init__(self, parent=None, frame=None, faces=None, title_hint=""):
        super().__init__(parent, "选择要注册的人脸")
        self.result = None
        self.frame = frame
        self.faces = faces or []

        n = len(self.faces)
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        w = cols * 220 + 80
        h = rows * 220 + 120
        self.setMinimumSize(w, h)
        self.resize(w, h)

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        label = QLabel("画面中有多张人脸，请点击要注册的人脸：")
        label.setStyleSheet(
            "color: #FFFFFF; "
            f"font-size: 14px; background: transparent;"
        )
        layout.addWidget(label)

        grid_widget = QWidget()
        grid_widget.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        grid_layout = QHBoxLayout(grid_widget)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setSpacing(12)

        for i, face in enumerate(self.faces):
            x1, y1, x2, y2, det_conf = face[0], face[1], face[2], face[3], face[4]
            hf, wf = self.frame.shape[:2]
            dw = int((x2 - x1) * 0.15)
            dh = int((y2 - y1) * 0.15)
            ex1 = max(0, x1 - dw)
            ey1 = max(0, y1 - dh)
            ex2 = min(wf, x2 + dw)
            ey2 = min(hf, y2 + dh)

            roi = self.frame[ey1:ey2, ex1:ex2]
            if roi.size == 0:
                continue

            thumb = cv2.resize(roi, (120, 120), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
            h_img, w_img, ch = rgb.shape
            bytes_per_line = ch * w_img
            qt_img = QImage(rgb.data, w_img, h_img, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_img)

            face_card = QFrame()
            face_card.setObjectName("glassCard")
            face_card.setFixedSize(160, 190)
            face_card.setCursor(Qt.CursorShape.PointingHandCursor)

            face_card.setStyleSheet("""
                QFrame#glassCard {
                    background: rgba(255,255,255,0.06);
                    border: 1px solid rgba(255,255,255,0.08);
                    border-radius: 8px;
                }
                QFrame#glassCard:hover {
                    background: rgba(96,205,255,0.15);
                    border: 1px solid #60CDFF;
                }
            """)

            card_layout = QVBoxLayout(face_card)
            card_layout.setContentsMargins(8, 8, 8, 8)
            card_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

            img_label = QLabel()
            img_label.setPixmap(pixmap)
            img_label.setFixedSize(120, 120)
            img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            img_label.setStyleSheet("border-radius: 6px; background: rgba(0,0,0,0.05);")
            card_layout.addWidget(img_label, 0, Qt.AlignmentFlag.AlignCenter)

            text_label = QLabel(f"人脸 #{i+1}  ({det_conf:.0%})")
            text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            text_label.setStyleSheet(
                "color: #FFFFFF; font-size: 12px; background: transparent; padding-top: 4px;"
            )
            card_layout.addWidget(text_label, 0, Qt.AlignmentFlag.AlignCenter)

            idx = i
            face_card.mousePressEvent = lambda e, idx=idx: self._select(idx)
            grid_layout.addWidget(face_card)

        grid_layout.addStretch()
        layout.addWidget(grid_widget, 1)

        cancel_btn = QPushButton("取消")
        cancel_btn.setFixedSize(100, 38)
        cancel_btn.clicked.connect(self.reject)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def _select(self, idx):
        self.result = idx
        self.accept()

    def get_result(self):
        return self.result


# ═══════════════════════════════════════════════════════════════
# 摄像头分辨率预设
# ═══════════════════════════════════════════════════════════════

_RESOLUTIONS = [
    "320×240 (4:3)",
    "480×360 (4:3)",
    "640×360 (16:9)",
    "640×480 (4:3)",
    "800×600 (4:3)",
    "960×540 (16:9)",
    "1024×576 (16:9)",
    "1280×720 (HD)",
]


def _res_to_tuple(res_str):
    """将 '640×360 (16:9)' 转为 (640, 360)"""
    parts = res_str.split("×")
    w = int(parts[0].strip())
    h = int(parts[1].split(" ")[0].strip())
    return w, h


def _current_resolution_key():
    """返回当前 APP_SETTINGS 对应的分辨率选项文字"""
    cw = APP_SETTINGS.get("cam_width", 640)
    ch = APP_SETTINGS.get("cam_height", 360)
    for res in _RESOLUTIONS:
        w, h = _res_to_tuple(res)
        if w == cw and h == ch:
            return res
    return f"{cw}×{ch} (自定义)"


class SettingsDialogPyQt(GlassDialog):
    """设置对话框 — 应用全局样式表保持与主窗口风格统一"""

    def __init__(self, parent=None):
        super().__init__(parent, "设置")
        self.parent_app = parent
        self.setMinimumSize(580, 620)
        self.resize(580, 680)

        # 外层布局
        outer = QVBoxLayout()
        outer.setContentsMargins(20, 20, 20, 16)
        outer.setSpacing(8)

        # ── 标题（固定顶部） ──
        title = QLabel("⚙  FaceVision 设置")
        title.setStyleSheet(
            "color: #FFFFFF; font-size: 18px; font-weight: bold; background: transparent;"
        )
        outer.addWidget(title)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background: rgba(255,255,255,0.06); max-height: 1px;")
        outer.addWidget(sep)

        # ── 可滚动内容区 ──
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; } "
            "QScrollArea > QWidget > QWidget { background: transparent; }"
        )
        scroll_content = QWidget()
        scroll_content.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(scroll_content)
        layout.setContentsMargins(8, 8, 12, 8)
        layout.setSpacing(10)

        # ── 推理设备 ──
        self._add_section_label(layout, "推理设备")

        import onnxruntime
        gpu_available = False
        gpu_label = "GPU"
        try:
            available = onnxruntime.get_available_providers()
            if 'CUDAExecutionProvider' in available:
                gpu_available = True
                gpu_label = "GPU (CUDA)"
            elif 'DmlExecutionProvider' in available:
                gpu_available = True
                gpu_label = "GPU (DirectML)"
        except Exception:
            gpu_available = False

        self.device_var = "cuda" if APP_SETTINGS.get("device") == "cuda" else "cpu"

        device_frame = QWidget()
        device_layout = QHBoxLayout(device_frame)
        device_layout.setContentsMargins(0, 0, 0, 0)
        device_layout.setSpacing(12)

        self.btn_cpu = QPushButton("CPU")
        self.btn_cpu.setMinimumWidth(180)
        self.btn_cpu.setFixedHeight(38)
        self.btn_cpu.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_cpu.clicked.connect(lambda: self._on_device_change("cpu"))
        device_layout.addWidget(self.btn_cpu)

        self.btn_gpu = QPushButton(gpu_label if gpu_available else "GPU (不可用)")
        self.btn_gpu.setMinimumWidth(180)
        self.btn_gpu.setFixedHeight(38)
        self.btn_gpu.setCursor(Qt.CursorShape.PointingHandCursor)
        if not gpu_available:
            self.btn_gpu.setEnabled(False)
        self.btn_gpu.clicked.connect(lambda: self._on_device_change("cuda"))
        device_layout.addWidget(self.btn_gpu)

        self._update_device_buttons()
        layout.addWidget(device_frame)

        # ── 摄像头分辨率 ──
        self._add_section_label(layout, "摄像头分辨率")
        self.res_dropdown = QComboBox()
        self.res_dropdown.addItems(_RESOLUTIONS)
        self.res_dropdown.setCurrentText(_current_resolution_key())
        self.res_dropdown.setFixedHeight(38)
        self.res_dropdown.setStyleSheet(
            "QComboBox { color: #FFFFFF; background: #2A2A2A; border: 1px solid rgba(255,255,255,0.08); border-radius: 6px; padding: 6px 12px; font-size: 14px; }"
            "QComboBox:hover { border-color: rgba(255,255,255,0.20); }"
            "QComboBox QAbstractItemView { color: #FFFFFF; background: #2D2D2D; border: 1px solid rgba(255,255,255,0.10); border-radius: 6px; selection-background-color: #60CDFF; selection-color: #FFFFFF; outline: none; padding: 4px; }"
        )
        layout.addWidget(self.res_dropdown)
        layout.addWidget(
            self._make_hint_label("更改将在下次启动摄像头时生效")
        )

        # ── 检测置信度 ──
        self._add_section_label(layout, "检测置信度")
        conf_row = self._make_slider_row(
            "conf", 30, 80,
            int(APP_SETTINGS.get("confidence", 0.50) * 100),
            self._update_conf_label
        )
        self.conf_label = self._make_value_label(f"{APP_SETTINGS.get('confidence', 0.50):.2f}")
        conf_row.layout().addWidget(self.conf_label)
        layout.addWidget(conf_row)

        # ── 识别容差 ──
        self._add_section_label(layout, "识别容差")
        tol_row = self._make_slider_row(
            "tol", 30, 80,
            int(APP_SETTINGS.get("tolerance", 0.45) * 100),
            self._update_tol_label
        )
        self.tol_label = self._make_value_label(f"{APP_SETTINGS.get('tolerance', 0.45):.2f}")
        tol_row.layout().addWidget(self.tol_label)
        layout.addWidget(tol_row)

        # ── 处理帧率 ──
        self._add_section_label(layout, "处理帧率 (FPS)")
        fps_row = self._make_slider_row(
            "fps", 5, 60,
            int(APP_SETTINGS.get("proc_fps", 30)),
            self._update_fps_label
        )
        self.fps_label = self._make_value_label(f"{APP_SETTINGS.get('proc_fps', 30):.0f}")
        fps_row.layout().addWidget(self.fps_label)
        layout.addWidget(fps_row)

        # ── 检测模型尺寸 ──
        self._add_section_label(layout, "检测模型尺寸 (越小越快)")
        det_size_row = QWidget()
        det_size_layout = QHBoxLayout(det_size_row)
        det_size_layout.setContentsMargins(0, 0, 0, 0)
        det_size_layout.setSpacing(8)
        self.det_size_combo = QComboBox()
        self.det_size_combo.addItems(["320 (快速)", "480 (均衡)", "640 (精准)"])
        self.det_size_combo.setStyleSheet(
            "QComboBox { color: #FFFFFF; background: #2A2A2A; border: 1px solid rgba(255,255,255,0.08); border-radius: 6px; padding: 6px 12px; font-size: 14px; }"
            "QComboBox:hover { border-color: rgba(255,255,255,0.20); }"
            "QComboBox QAbstractItemView { color: #FFFFFF; background: #2D2D2D; border: 1px solid rgba(255,255,255,0.10); border-radius: 6px; selection-background-color: #60CDFF; selection-color: #FFFFFF; outline: none; padding: 4px; }"
        )
        current_det = APP_SETTINGS.get("det_size", 640)
        if current_det == 320:
            self.det_size_combo.setCurrentIndex(0)
        elif current_det == 480:
            self.det_size_combo.setCurrentIndex(1)
        else:
            self.det_size_combo.setCurrentIndex(2)
        det_size_layout.addWidget(self.det_size_combo)
        det_size_layout.addWidget(
            self._make_hint_label("需重启程序生效")
        )
        det_size_layout.addStretch()
        layout.addWidget(det_size_row)

        # ── 追踪平滑帧数 ──
        self._add_section_label(layout, "追踪平滑帧数")
        smooth_row = self._make_slider_row(
            "smooth", 3, 10,
            APP_SETTINGS.get("track_smooth", 5),
            self._update_smooth_label
        )
        self.smooth_label = self._make_value_label(f"{APP_SETTINGS.get('track_smooth', 5)} 帧", width=60)
        smooth_row.layout().addWidget(self.smooth_label)
        layout.addWidget(smooth_row)

        # ── 质量过滤 ──
        self._add_section_label(layout, "质量过滤")
        qf_row = QWidget()
        qf_layout = QHBoxLayout(qf_row)
        qf_layout.setContentsMargins(0, 0, 0, 0)
        qf_layout.setSpacing(12)
        self.qf_check = QCheckBox("启用模糊度过滤")
        self.qf_check.setChecked(APP_SETTINGS.get("quality_filter", True))
        qf_layout.addWidget(self.qf_check)
        self.minface_combo = QComboBox()
        self.minface_combo.addItems(["最小 60px", "最小 80px", "最小 100px", "最小 120px"])
        self.minface_combo.setStyleSheet(
            "QComboBox { color: #FFFFFF; background: #2A2A2A; border: 1px solid rgba(255,255,255,0.08); border-radius: 6px; padding: 6px 12px; font-size: 14px; }"
            "QComboBox:hover { border-color: rgba(255,255,255,0.20); }"
            "QComboBox QAbstractItemView { color: #FFFFFF; background: #2D2D2D; border: 1px solid rgba(255,255,255,0.10); border-radius: 6px; selection-background-color: #60CDFF; selection-color: #FFFFFF; outline: none; padding: 4px; }"
        )
        current_min = APP_SETTINGS.get("min_face_size", 60)
        minface_map = {60: 0, 80: 1, 100: 2, 120: 3}
        self.minface_combo.setCurrentIndex(minface_map.get(current_min, 0))
        qf_layout.addWidget(self.minface_combo)
        qf_layout.addStretch()
        layout.addWidget(qf_row)

        layout.addStretch()
        scroll.setWidget(scroll_content)
        outer.addWidget(scroll, 1)

        # ── 应用 / 取消（固定底部） ──
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QPushButton("取消")
        cancel_btn.setFixedSize(120, 40)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        apply_btn = QPushButton("应用")
        apply_btn.setObjectName("btnAccent")
        apply_btn.setFixedSize(120, 40)
        apply_btn.clicked.connect(self._apply_and_close)
        btn_layout.addWidget(apply_btn)

        outer.addLayout(btn_layout)
        self.setLayout(outer)

    def _make_value_label(self, text, width=50):
        """创建一致的数值标签"""
        label = QLabel(text)
        label.setFixedWidth(width)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return label

    def _make_hint_label(self, text):
        """创建提示标签"""
        label = QLabel(text)
        label.setStyleSheet(
            "color: #FFFFFF; font-size: 11px; "
            "background: transparent;"
        )
        return label

    def _add_section_label(self, layout, text):
        label = QLabel(text)
        label.setObjectName("sectionLabel")
        label.setStyleSheet(
            "color: #FFFFFF; font-size: 12px; font-weight: 600; "
            "background: transparent; margin-top: 8px; letter-spacing: 0.5px;"
        )
        layout.addWidget(label)

    def _make_slider_row(self, name, min_val, max_val, current, callback, step=1):
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setSingleStep(step)
        slider.setValue(current)
        slider.valueChanged.connect(callback)
        row_layout.addWidget(slider, 1)
        setattr(self, f"{name}_slider", slider)
        return row

    # ── 回调 ──
    def _update_conf_label(self, v):
        self.conf_label.setText(f"{v / 100:.2f}")
    def _update_tol_label(self, v):
        self.tol_label.setText(f"{v / 100:.2f}")
    def _update_fps_label(self, v):
        self.fps_label.setText(f"{v:.0f}")
    def _update_smooth_label(self, v):
        self.smooth_label.setText(f"{v} 帧")

    def _on_device_change(self, device):
        self.device_var = device
        self._update_device_buttons()

    def _update_device_buttons(self):
        if self.device_var == "gpu" or self.device_var == "cuda":
            self.btn_gpu.setObjectName("btnGpuActive")
            self.btn_cpu.setObjectName("btnCpuActive")
        else:
            self.btn_cpu.setObjectName("btnGpuActive")
            self.btn_gpu.setObjectName("btnCpuActive")
        # Force style refresh
        for btn in [self.btn_gpu, self.btn_cpu]:
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def _apply_and_close(self):
        new_res_str = self.res_dropdown.currentText()
        new_w, new_h = _res_to_tuple(new_res_str)

        # 检测尺寸映射
        det_size_map = {0: 320, 1: 480, 2: 640}
        new_det_size = det_size_map[self.det_size_combo.currentIndex()]

        # 最小人脸尺寸映射
        minface_map = {0: 60, 1: 80, 2: 100, 3: 120}
        new_minface = minface_map[self.minface_combo.currentIndex()]

        settings = {
            "device": self.device_var,
            "cam_width": new_w,
            "cam_height": new_h,
            "confidence": self.conf_slider.value() / 100,
            "tolerance": self.tol_slider.value() / 100,
            "proc_fps": self.fps_slider.value(),
            "det_size": new_det_size,
            "track_smooth": self.smooth_slider.value(),
            "quality_filter": self.qf_check.isChecked(),
            "min_face_size": new_minface,
        }

        # 检查分辨率是否变化
        old_w = APP_SETTINGS.get("cam_width", 640)
        old_h = APP_SETTINGS.get("cam_height", 360)
        res_changed = (new_w != old_w or new_h != old_h)

        save_settings(settings)
        for k, v in settings.items():
            APP_SETTINGS[k] = v
        if self.parent_app and hasattr(self.parent_app, 'on_settings_changed'):
            self.parent_app.on_settings_changed(settings)

        if res_changed:
            QMessageBox.information(
                self, "分辨率更改",
                f"摄像头分辨率已设为 {new_w}×{new_h}。\n"
                "请重启程序使新分辨率生效。"
            )

        self.accept()

# ═══════════════════════════════════════════════════════════════
# 主窗口
# ═══════════════════════════════════════════════════════════════

class FaceVisionWindow(QMainWindow):
    """FaceVision 主窗口 — Windows 11 风格"""

    def __init__(self, camera, db, detector, recognizer):
        super().__init__()
        self.camera = camera
        self.db = db
        self.detector = detector
        self.recognizer = recognizer
        self.tracker = FaceTracker(smooth_frames=APP_SETTINGS.get("track_smooth", 5))

        self.is_dark = IS_DARK
        self.is_running = False
        self.processing = None

        # 人脸照片目录
        self.face_photos_dir = Path(os.path.dirname(__file__)) / "face_photos"
        self.face_photos_dir.mkdir(exist_ok=True)

        # 注册状态
        self.register_cooldown = False
        self._last_frame = None
        self._last_detected_faces = []

        self._setup_ui()
        # 用户通过摄像头按钮手动启动

    def _setup_ui(self):
        self.setWindowTitle("FaceVision")
        self.setMinimumSize(1100, 720)
        self.resize(1280, 800)

        # 无边框窗口 — Qt 级自定义拖动
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self._drag_pos = None

        # 中心 widget
        central = QWidget()
        central.setObjectName("centralWidget")
        self.setCentralWidget(central)

        # 自定义标题栏
        title_bar = self._create_title_bar()
        title_bar.setFixedHeight(48)

        # 主布局
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(12, 0, 12, 12)
        main_layout.setSpacing(12)

        # ── 左侧面板 ──
        left_panel = self._create_left_panel()
        left_panel.setFixedWidth(300)

        # ── 中央画面 ──
        center_panel = self._create_center_panel()

        main_layout.addWidget(center_panel, 1)
        main_layout.addWidget(left_panel)

        # 包装
        wrapper = QVBoxLayout(central)
        wrapper.setContentsMargins(0, 0, 0, 0)
        wrapper.setSpacing(0)
        wrapper.addWidget(title_bar)
        wrapper.addLayout(main_layout, 1)

        # 应用样式
        self.setStyleSheet(make_stylesheet())

    def _create_title_bar(self):
        bar = QWidget()
        bar.setObjectName("titleBar")
        bar.setStyleSheet("background: transparent;")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 0, 8, 0)

        icon_label = QLabel("🔷 FaceVision")
        icon_label.setStyleSheet(
            "color: #FFFFFF; font-size: 14px; font-weight: 600; background: transparent;"
        )
        layout.addWidget(icon_label)
        layout.addStretch()

        # 最小化
        min_btn = QPushButton("─")
        min_btn.setFixedSize(36, 32)
        min_btn.setStyleSheet("""
            QPushButton {
                background: transparent; border: none; color: #FFFFFF;
                font-size: 16px; border-radius: 4px;
            }
            QPushButton:hover { background: rgba(255,255,255,0.06); color: #FFFFFF; }
        """)
        min_btn.clicked.connect(self.showMinimized)
        layout.addWidget(min_btn)

        # 关闭
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(36, 32)
        close_btn.setStyleSheet("""
            QPushButton {
                background: transparent; border: none; color: #FFFFFF;
                font-size: 14px; border-radius: 4px;
            }
            QPushButton:hover { background: #C42B1C; color: #FFFFFF; }
        """)
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

        return bar

    def _create_center_panel(self):
        """中央视频画面区"""
        panel = QFrame()
        panel.setObjectName("card")
        panel.setStyleSheet(
            "QFrame { background-color: #2A2A2A; border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; }"
        )
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setStyleSheet("background: rgba(0,0,0,0.3); border-radius: 8px;")
        self.video_label.setText("📷 点击「启动摄像头」开始\n\n实时人脸识别")
        layout.addWidget(self.video_label)

        # 控制条（摄像头开关 + 状态）
        ctrl_bar = QWidget()
        ctrl_bar.setFixedHeight(42)
        ctrl_layout = QHBoxLayout(ctrl_bar)
        ctrl_layout.setContentsMargins(12, 4, 12, 4)
        ctrl_layout.setSpacing(10)

        self.btn_camera = QPushButton("▶  启动摄像头")
        self.btn_camera.setObjectName("btnAccent")
        self.btn_camera.setFixedHeight(34)
        self.btn_camera.setFixedWidth(150)
        self.btn_camera.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_camera.clicked.connect(self._toggle_camera)
        ctrl_layout.addWidget(self.btn_camera)

        self.status_label = QLabel("● 就绪")
        self.status_label.setStyleSheet("color: #FFFFFF; font-size: 13px;")
        ctrl_layout.addWidget(self.status_label)
        ctrl_layout.addStretch()

        self.fps_label = QLabel("")
        self.fps_label.setStyleSheet("color: #FFFFFF; font-size: 12px;")
        ctrl_layout.addWidget(self.fps_label)

        layout.addWidget(ctrl_bar)
        return panel

    def _toggle_camera(self):
        """启动/停止摄像头"""
        if not self.is_running:
            try:
                self.camera.start()
                self.is_running = True
                self._start_processing()
                self.btn_camera.setText("⏹  停止摄像头")
                self.btn_camera.setObjectName("btnDanger")
                self.btn_camera.style().unpolish(self.btn_camera)
                self.btn_camera.style().polish(self.btn_camera)
                self._set_status("● 运行中", SUCCESS_COLOR)
                self.video_label.setText("")
            except Exception as e:
                QMessageBox.critical(self, "摄像头错误", str(e))
                self.is_running = False
        else:
            if hasattr(self, 'processing') and self.processing:
                self.processing.stop()
                self.processing = None
            self.camera.stop()
            self.is_running = False

            self.btn_camera.setText("▶  启动摄像头")
            self.btn_camera.setObjectName("btnAccent")
            self.btn_camera.style().unpolish(self.btn_camera)
            self.btn_camera.style().polish(self.btn_camera)

            self.video_label.setPixmap(QPixmap())
            self.video_label.setText("📷 点击「启动摄像头」开始\n\n实时人脸识别")
            self._set_status("● 就绪", "#FFFFFF")
            self.fps_label.setText("")

    def _create_left_panel(self):
        """右侧面板（实际放左边布局中放右边）"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        # 深色背景：QScrollArea 自身 + 内部 viewport
        scroll.setStyleSheet(
            "QScrollArea { background: #202020; border: none; }"
            "QScrollArea > QWidget > QWidget { background: #202020; }"
        )

        panel = QWidget()
        panel.setStyleSheet("background-color: #202020;")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # ── 人员列表卡片 ──
        person_card = QFrame()
        person_card.setObjectName("card")
        person_card.setStyleSheet(
            "QFrame { background-color: #2A2A2A; border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; }"
        )
        person_layout = QVBoxLayout(person_card)
        person_layout.setContentsMargins(16, 14, 16, 14)
        person_layout.setSpacing(10)

        person_header = QHBoxLayout()
        person_title = QLabel("👤 已注册人员")
        person_title.setStyleSheet(
            "color: #FFFFFF; font-size: 15px; font-weight: 600; background: transparent;"
        )
        person_header.addWidget(person_title)
        person_header.addStretch()
        person_layout.addLayout(person_header)

        self.person_list = QVBoxLayout()
        self.person_list.setSpacing(4)
        person_layout.addLayout(self.person_list)

        # 按钮
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        add_btn = QPushButton("添加")
        add_btn.setObjectName("btnAccent")
        add_btn.setFixedHeight(36)
        add_btn.clicked.connect(self._on_add_person)
        btn_row.addWidget(add_btn, 1)

        import_btn = QPushButton("从图片")
        import_btn.setFixedHeight(36)
        import_btn.clicked.connect(self._on_import_image)
        btn_row.addWidget(import_btn, 1)

        delete_btn = QPushButton("删除")
        delete_btn.setObjectName("btnDanger")
        delete_btn.setFixedHeight(36)
        delete_btn.clicked.connect(self._on_delete_person)
        btn_row.addWidget(delete_btn, 1)

        person_layout.addLayout(btn_row)
        layout.addWidget(person_card)

        # ── 设置卡片 ──
        settings_card = QFrame()
        settings_card.setObjectName("card")
        settings_card.setStyleSheet(
            "QFrame { background-color: #2A2A2A; border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; }"
        )
        settings_layout = QVBoxLayout(settings_card)
        settings_layout.setContentsMargins(16, 14, 16, 14)
        settings_layout.setSpacing(10)

        settings_title = QLabel("⚙ 快速操作")
        settings_title.setStyleSheet(
            "color: #FFFFFF; font-size: 15px; font-weight: 600; background: transparent;"
        )
        settings_layout.addWidget(settings_title)

        settings_btn = QPushButton("打开完整设置")
        settings_btn.setFixedHeight(36)
        settings_btn.clicked.connect(self._open_settings)
        settings_layout.addWidget(settings_btn)

        layout.addWidget(settings_card)

        # ── 设备信息 ──
        info_card = QFrame()
        info_card.setObjectName("card")
        info_card.setStyleSheet(
            "QFrame { background-color: #2A2A2A; border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; }"
        )
        info_layout = QVBoxLayout(info_card)
        info_layout.setContentsMargins(16, 14, 16, 14)
        info_layout.setSpacing(6)

        info_title = QLabel("ℹ 设备信息")
        info_title.setStyleSheet(
            "color: #FFFFFF; font-size: 15px; font-weight: 600; background: transparent;"
        )
        info_layout.addWidget(info_title)

        device = APP_SETTINGS.get("device", "cpu")
        det_size = APP_SETTINGS.get("det_size", 640)
        self.info_device = QLabel(f"推理设备: {device}")
        self.info_device.setStyleSheet("color: #FFFFFF; font-size: 12px;")
        info_layout.addWidget(self.info_device)

        self.info_det_size = QLabel(f"检测尺寸: {det_size}px")
        self.info_det_size.setStyleSheet("color: #FFFFFF; font-size: 12px;")
        info_layout.addWidget(self.info_det_size)

        person_count = len(self.db.get_names())
        self.info_people = QLabel(f"已注册: {person_count} 人")
        self.info_people.setStyleSheet("color: #FFFFFF; font-size: 12px;")
        info_layout.addWidget(self.info_people)

        layout.addWidget(info_card)
        layout.addStretch()

        scroll.setWidget(panel)
        self._refresh_person_list()
        return scroll

    def _refresh_person_list(self):
        """刷新人员列表"""
        # Clear
        while self.person_list.count():
            item = self.person_list.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        names = self.db.get_names()
        for name in sorted(names):
            label = QLabel(f"  {name}")
            label.setStyleSheet(
                "color: #FFFFFF; font-size: 13px; "
                "padding: 6px 8px; background: transparent;"
            )
            self.person_list.addWidget(label)

        if not names:
            placeholder = QLabel("  暂无注册人员")
            placeholder.setStyleSheet(
                "color: #FFFFFF; font-size: 13px; padding: 6px 8px;"
            )
            self.person_list.addWidget(placeholder)

    # ── 处理 ──

    def _start_processing(self):
        self.processing = ProcessingThread(
            self.camera, self.detector, self.recognizer, self.tracker, self.db
        )
        self.processing.frame_ready.connect(self._on_frame)
        self.processing.start()

        # FPS 计数器
        self._fps_timer = QTimer()
        self._fps_timer.timeout.connect(self._update_fps_display)
        self._fps_timer.start(1000)
        self._frame_count = 0
        self._last_fps_time = time.time()

    def _on_frame(self, frame, recognized, tracks):
        self._frame_count += 1
        self._last_frame = frame
        self._last_detected_faces = recognized

        # 绘制
        display = frame.copy()
        for (x1, y1, x2, y2, conf, name, sim) in recognized:
            # 边框颜色：已确认身份=蓝色，未知=灰色
            is_unknown = (name == "未知" or name == "unknown")
            color = (96, 205, 255) if not is_unknown else (200, 200, 200)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

            # 标签背景
            label = f"{name} ({sim:.0%})" if not is_unknown else f"未知 ({sim:.0%})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(display, (x1, y1 - th - 10), (x1 + tw + 8, y1),
                          (30, 30, 30), -1)
            cv2.putText(display, label, (x1 + 4, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # 缩放并显示
        h, w = display.shape[:2]
        target_h = self.video_label.height()
        target_w = self.video_label.width()
        if target_w > 0 and target_h > 0:
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            if new_w > 0 and new_h > 0:
                display = cv2.resize(display, (new_w, new_h))

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h_img, w_img, ch = rgb.shape
        bytes_per_line = ch * w_img
        qt_img = QImage(rgb.data, w_img, h_img, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        self.video_label.setPixmap(pixmap)

    def _update_fps_display(self):
        elapsed = time.time() - self._last_fps_time
        if elapsed > 0:
            fps = self._frame_count / elapsed
            self.fps_label.setText(f"{fps:.1f} FPS")
        self._frame_count = 0
        self._last_fps_time = time.time()

    # ── 操作 ──

    def _on_add_person(self):
        if self._last_frame is not None and len(self._last_detected_faces) > 0:
            self._add_from_camera()
        else:
            self._add_from_image_prompt()

    def _add_from_camera(self):
        self.processing.pause()
        dialog = AddPersonDialogPyQt(self, "注册人员", "请为此人输入姓名：")
        if dialog.exec() != QDialog.DialogCode.Accepted:
            self._resume_processing()
            return

        name = dialog.get_result()
        if not name:
            self._resume_processing()
            return

        frame = self._last_frame.copy()
        faces = self._last_detected_faces

        if len(faces) == 1:
            target = faces[0]
        else:
            selector = SelectFaceDialogPyQt(self, frame,
                [(f[0], f[1], f[2], f[3], f[4]) for f in faces])
            if selector.exec() != QDialog.DialogCode.Accepted:
                self._resume_processing()
                return
            idx = selector.get_result()
            if idx is None or idx >= len(faces):
                self._resume_processing()
                return
            target = faces[idx]

        x1, y1, x2, y2, conf, name_pred, sim = target
        roi = self.detector.extract_face_roi(frame, (x1, y1, x2, y2, conf))

        # Get embedding via detect_with_embeddings
        faces_full = self.detector.detect_with_embeddings(frame)
        emb = None
        for f in faces_full:
            if abs(f[0] - x1) < 10 and abs(f[1] - y1) < 10:
                emb = f[5]
                break

        if emb is None:
            QMessageBox.warning(self, "编码失败", "无法提取人脸特征，请重试。")
            self._resume_processing()
            return

        filename = f"{name}_{uuid.uuid4().hex[:8]}.jpg"
        img_path = str(self.face_photos_dir / filename)
        cv2.imwrite(img_path, roi)

        success, msg = self.db.add_person(name, img_path, emb)
        if success:
            known_encodings, known_names = self.db.get_encodings_and_names()
            self.recognizer.update_cache(known_encodings, known_names, self.db.version)
            self._refresh_person_list()
            self._set_status(f"✓ {msg}", SUCCESS_COLOR)
        else:
            QMessageBox.warning(self, "添加失败", msg)
            self._set_status("● 就绪", "#FFFFFF")

        self._resume_processing()

    def _add_from_image_prompt(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp)"
        )
        if not file_path:
            return

        dialog = AddPersonDialogPyQt(self, "从图片注册", "请输入人员姓名：")
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        name = dialog.get_result()
        if not name:
            return

        self._add_from_image(name, file_path)

    def _add_from_image(self, name, file_path):
        try:
            img = cv2.imread(file_path)
            if img is None:
                QMessageBox.warning(self, "图片错误", "无法读取图片文件。")
                return
            faces = self.detector.detect_with_embeddings(img)
            if len(faces) == 0:
                QMessageBox.warning(self, "未检测到人脸", "所选图片中未检测到清晰人脸。")
                return

            if len(faces) == 1:
                target_idx = 0
            else:
                selector = SelectFaceDialogPyQt(self, img,
                    [(f[0], f[1], f[2], f[3], f[4]) for f in faces])
                if selector.exec() != QDialog.DialogCode.Accepted:
                    return
                target_idx = selector.get_result()
                if target_idx is None:
                    return

            best = faces[target_idx]
            x1, y1, x2, y2, det_conf = best[0], best[1], best[2], best[3], best[4]
            embedding = best[5]
            if embedding is None:
                QMessageBox.warning(self, "编码失败", "无法提取人脸特征。")
                return

            ext = Path(file_path).suffix
            filename = f"{name}_{uuid.uuid4().hex[:8]}{ext}"
            dest_path = str(self.face_photos_dir / filename)
            roi = self.detector.extract_face_roi(img, (x1, y1, x2, y2, det_conf))
            cv2.imwrite(dest_path, roi)

            success, msg = self.db.add_person(name, dest_path, embedding)
            if success:
                known_encodings, known_names = self.db.get_encodings_and_names()
                self.recognizer.update_cache(known_encodings, known_names, self.db.version)
                self._refresh_person_list()
                self._set_status(f"✓ {msg}", SUCCESS_COLOR)
            else:
                QMessageBox.warning(self, "添加失败", msg)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导入失败: {e}")

    def _on_import_image(self):
        self._add_from_image_prompt()

    def _on_delete_person(self):
        names = self.db.get_names()
        if not names:
            QMessageBox.information(self, "提示", "没有已注册的人员。")
            return
        dialog = BatchDeleteDialogPyQt(self, names)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        selected = dialog.get_result()
        if not selected:
            return
        for name in selected:
            self.db.remove_person(name)
        self._refresh_person_list()
        known_encodings, known_names = self.db.get_encodings_and_names()
        self.recognizer.update_cache(known_encodings, known_names, self.db.version)
        self._set_status(f"✓ 已删除 {len(selected)} 人", SUCCESS_COLOR)

    def _open_settings(self):
        dialog = SettingsDialogPyQt(self)
        dialog.exec()

    def on_settings_changed(self, settings):
        """设置变更回调"""
        # 更新检测器
        if "confidence" in settings or "det_size" in settings or "min_face_size" in settings or "quality_filter" in settings:
            self.detector.confidence = settings.get("confidence", self.detector.confidence)
            self.detector.min_face_size = settings.get("min_face_size", self.detector.min_face_size)
            if "quality_filter" in settings:
                self.detector.quality_filter = settings["quality_filter"]
            if "det_size" in settings:
                self.detector.reload_model(det_size=settings["det_size"])

        # 更新识别器
        if "tolerance" in settings:
            self.recognizer.tolerance = settings["tolerance"]

        # 更新追踪器
        if "track_smooth" in settings and hasattr(self, 'tracker') and self.tracker:
            self.tracker.smooth_frames = settings["track_smooth"]
            self.tracker.reset()

        # 更新处理帧率
        if "proc_fps" in settings and hasattr(self, 'processing') and self.processing:
            pass  # proc_fps is handled per-frame in ProcessingThread loop

        # 更新信息面板
        self.info_device.setText(f"推理设备: {APP_SETTINGS.get('device', 'cpu')}")
        self.info_det_size.setText(f"检测尺寸: {APP_SETTINGS.get('det_size', 640)}px")
        self.info_people.setText(f"已注册: {len(self.db.get_names())} 人")

        self._set_status("✓ 设置已应用", SUCCESS_COLOR)

    def _resume_processing(self):
        if self.processing:
            self.processing.resume()

    def _set_status(self, text, color="#FFFFFF"):
        self.status_label.setText(text)
        self.status_label.setStyleSheet(
            f"color: {color}; background: transparent; font-size: 13px;"
        )

    def mousePressEvent(self, event):
        """标题栏拖动：鼠标按下时记录位置（仅限顶部 48px）"""
        if event.button() == Qt.MouseButton.LeftButton and event.position().y() <= 48:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """标题栏拖动：鼠标移动时移动窗口"""
        if event.buttons() == Qt.MouseButton.LeftButton and self._drag_pos is not None:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """标题栏拖动：鼠标释放时清除拖动状态"""
        if self._drag_pos is not None:
            self._drag_pos = None
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        # Enable Windows 11 Mica backdrop (prevents ghosting on frameless windows)
        apply_mica(int(self.winId()))
        if not hasattr(self, '_started') or not self._started:
            self._started = True
            self._set_status("● 就绪 — 点击「启动摄像头」开始", "#FFFFFF")

    def closeEvent(self, event):
        if self.processing:
            self.processing.stop()
        if self.is_running:
            self.camera.stop()
        event.accept()

# ═══════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════

def run_app(camera, db, detector, recognizer):
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = FaceVisionWindow(camera, db, detector, recognizer)
    window.show()
    return app.exec()