"""
Windows 11 玻璃态仪表盘 UI — 基于 PyQt5
使用 Acrylic / Mica 风格的半透明玻璃效果，呈现现代化外观
"""

import sys
import os
import cv2
import uuid
import time
import threading
import numpy as np
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QScrollArea, QCheckBox,
    QDialog, QLineEdit, QMessageBox, QFileDialog, QListWidget,
    QListWidgetItem, QSlider, QComboBox
)
from PyQt5.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QSize, QRect, QRectF,
    QEasingCurve, QPoint, pyqtSlot
)
from PyQt5.QtGui import (
    QPixmap, QImage, QFont, QColor, QPainter, QBrush, QPen,
    QLinearGradient, QRadialGradient, QPainterPath, QFontDatabase,
    QPalette, QIcon
)

from config import APP_SETTINGS, save_settings

try:
    import onnxruntime
    _ONNX_AVAILABLE = True
except ImportError:
    _ONNX_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════
# 摄像头分辨率预设
# ═══════════════════════════════════════════════════════════════
RESOLUTIONS = [
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
    for res in RESOLUTIONS:
        w, h = _res_to_tuple(res)
        if w == cw and h == ch:
            return res
    return f"{cw}×{ch} (自定义)"


# ═══════════════════════════════════════════════════════════════
# Windows 11 色彩系统
# ═══════════════════════════════════════════════════════════════
class Win11Colors:
    ACCENT = "#0078D4"
    ACCENT_LIGHT = "#E8F4FD"
    ACCENT_HOVER = "#106EBE"
    ACCENT_PRESSED = "#005A9E"
    GLASS_WHITE = "rgba(255, 255, 255, 0.75)"
    GLASS_WHITE_DARK = "rgba(32, 32, 32, 0.80)"
    GLASS_BORDER = "rgba(255, 255, 255, 0.45)"
    GLASS_BORDER_DARK = "rgba(255, 255, 255, 0.08)"
    SURFACE_LIGHT = "#FAFAFA"
    SURFACE_DARK = "#2D2D2D"
    CARD_BG = "rgba(255, 255, 255, 0.85)"
    CARD_BG_DARK = "rgba(45, 45, 45, 0.85)"
    TEXT_PRIMARY = "#1A1A1A"
    TEXT_SECONDARY = "#707070"
    TEXT_ON_DARK = "#FFFFFF"
    TEXT_ON_DARK_SECONDARY = "#B0B0B0"
    SUCCESS = "#107C10"
    DANGER = "#C42B1C"
    WARNING = "#FF8C00"
    INFO = "#0078D4"
    BG_GRADIENT_START = "#F0F2F5"
    BG_GRADIENT_END = "#E8EBF0"


# ═══════════════════════════════════════════════════════════════
# 全局样式表 — Windows 11 玻璃态风格
# ═══════════════════════════════════════════════════════════════
def get_glass_stylesheet(is_dark=False):
    """生成全局玻璃态样式表"""
    if is_dark:
        return """
        /* ====== Windows 11 玻璃态全局样式 (暗色) ====== */
        QMainWindow, QWidget#centralWidget {
            background: transparent;
        }
        
        QLabel {
            color: #FFFFFF;
            font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
        }
        
        QPushButton {
            font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
            font-size: 14px;
            font-weight: 500;
            border: none;
            border-radius: 6px;
            padding: 8px 20px;
            color: #FFFFFF;
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 rgba(255,255,255,0.15), stop:1 rgba(255,255,255,0.05));
            border: 1px solid rgba(255,255,255,0.12);
        }
        QPushButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 rgba(255,255,255,0.25), stop:1 rgba(255,255,255,0.10));
            border: 1px solid rgba(255,255,255,0.20);
        }
        QPushButton:pressed {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 rgba(255,255,255,0.10), stop:1 rgba(255,255,255,0.05));
        }
        QPushButton#btnAccent {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #1682D9, stop:1 #0078D4);
            border: 1px solid rgba(255,255,255,0.2);
        }
        QPushButton#btnAccent:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #1A8DE6, stop:1 #0A7ED4);
        }
        QPushButton#btnDanger {
            color: #FF7B6B;
            background: rgba(196,43,28,0.2);
            border: 1px solid rgba(196,43,28,0.3);
        }
        QPushButton#btnDanger:hover {
            background: rgba(196,43,28,0.35);
        }
        QPushButton#btnSuccess {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #1C9B3C, stop:1 #10893E);
            border: 1px solid rgba(255,255,255,0.2);
        }
        QPushButton#btnSuccess:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #20A842, stop:1 #149343);
        }
        
        QFrame#glassPanel {
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 12px;
        }
        QFrame#glassCard {
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 8px;
        }
        
        QScrollArea {
            border: none;
            background: transparent;
        }
        QScrollBar:vertical {
            background: transparent;
            width: 6px;
            margin: 0;
        }
        QScrollBar::handle:vertical {
            background: rgba(255,255,255,0.15);
            border-radius: 3px;
            min-height: 30px;
        }
        QScrollBar::handle:vertical:hover {
            background: rgba(255,255,255,0.25);
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0;
        }
        
        QLineEdit {
            font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
            font-size: 14px;
            padding: 8px 12px;
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 6px;
            background: rgba(255,255,255,0.06);
            color: #FFFFFF;
        }
        QLineEdit:focus {
            border: 1px solid #0078D4;
            background: rgba(255,255,255,0.10);
        }
        QLineEdit::placeholder {
            color: rgba(255,255,255,0.35);
        }
        
        QListWidget {
            border: none;
            background: transparent;
            outline: none;
        }
        QListWidget::item {
            padding: 6px 10px;
            border-radius: 6px;
            color: #FFFFFF;
        }
        QListWidget::item:hover {
            background: rgba(255,255,255,0.08);
        }
        QListWidget::item:selected {
            background: rgba(0,120,212,0.3);
            color: #FFFFFF;
        }
        
        QCheckBox {
            font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
            font-size: 14px;
            color: #FFFFFF;
            spacing: 8px;
        }
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border-radius: 4px;
            border: 1px solid rgba(255,255,255,0.25);
            background: transparent;
        }
        QCheckBox::indicator:checked {
            background: #0078D4;
            border: 1px solid #0078D4;
        }
        QCheckBox::indicator:hover {
            border: 1px solid rgba(255,255,255,0.40);
        }
        
        QSlider::groove:horizontal {
            border: none;
            height: 4px;
            border-radius: 2px;
            background: rgba(255,255,255,0.10);
        }
        QSlider::handle:horizontal {
            background: #0078D4;
            width: 16px;
            height: 16px;
            margin: -6px 0;
            border-radius: 8px;
        }
        QSlider::handle:horizontal:hover {
            background: #1A8DE6;
        }
        QSlider::sub-page:horizontal {
            background: #0078D4;
            border-radius: 2px;
        }
        
        QComboBox {
            font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
            font-size: 14px;
            padding: 6px 12px;
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 6px;
            background: rgba(255,255,255,0.08);
            color: #FFFFFF;
            min-height: 32px;
        }
        QComboBox:hover {
            border: 1px solid rgba(255,255,255,0.2);
        }
        QComboBox QAbstractItemView {
            font-size: 14px;
            background: #2D2D2D;
            color: #FFFFFF;
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 6px;
            selection-background-color: #0078D4;
            outline: none;
        }
        """
    else:
        return """
        /* ====== Windows 11 玻璃态全局样式 (浅色) ====== */
        QMainWindow, QWidget#centralWidget {
            background: transparent;
        }
        
        QLabel {
            color: #1A1A1A;
            font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
        }
        
        QPushButton {
            font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
            font-size: 14px;
            font-weight: 500;
            border: none;
            border-radius: 6px;
            padding: 8px 20px;
            color: #1A1A1A;
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 rgba(255,255,255,0.95), stop:1 rgba(240,240,240,0.90));
            border: 1px solid rgba(0,0,0,0.08);
        }
        QPushButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 rgba(255,255,255,1.0), stop:1 rgba(248,248,248,0.95));
            border: 1px solid rgba(0,0,0,0.12);
        }
        QPushButton:pressed {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 rgba(230,230,230,0.95), stop:1 rgba(220,220,220,0.90));
        }
        QPushButton#btnAccent {
            color: #FFFFFF;
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #1682D9, stop:1 #0078D4);
            border: 1px solid rgba(255,255,255,0.2);
        }
        QPushButton#btnAccent:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #1A8DE6, stop:1 #0A7ED4);
        }
        QPushButton#btnDanger {
            color: #C42B1C;
            background: rgba(196,43,28,0.08);
            border: 1px solid rgba(196,43,28,0.2);
        }
        QPushButton#btnDanger:hover {
            background: rgba(196,43,28,0.18);
        }
        QPushButton#btnSuccess {
            color: #FFFFFF;
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #1C9B3C, stop:1 #10893E);
            border: 1px solid rgba(255,255,255,0.2);
        }
        QPushButton#btnSuccess:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #20A842, stop:1 #149343);
        }
        
        QFrame#glassPanel {
            background: rgba(255,255,255,0.70);
            border: 1px solid rgba(255,255,255,0.50);
            border-radius: 12px;
        }
        QFrame#glassCard {
            background: rgba(255,255,255,0.55);
            border: 1px solid rgba(255,255,255,0.40);
            border-radius: 8px;
        }
        
        QScrollArea {
            border: none;
            background: transparent;
        }
        QScrollBar:vertical {
            background: transparent;
            width: 6px;
            margin: 0;
        }
        QScrollBar::handle:vertical {
            background: rgba(0,0,0,0.10);
            border-radius: 3px;
            min-height: 30px;
        }
        QScrollBar::handle:vertical:hover {
            background: rgba(0,0,0,0.20);
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0;
        }
        
        QLineEdit {
            font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
            font-size: 14px;
            padding: 8px 12px;
            border: 1px solid rgba(0,0,0,0.10);
            border-radius: 6px;
            background: rgba(255,255,255,0.60);
            color: #1A1A1A;
        }
        QLineEdit:focus {
            border: 1px solid #0078D4;
            background: rgba(255,255,255,0.85);
        }
        QLineEdit::placeholder {
            color: rgba(0,0,0,0.30);
        }
        
        QListWidget {
            border: none;
            background: transparent;
            outline: none;
        }
        QListWidget::item {
            padding: 6px 10px;
            border-radius: 6px;
            color: #1A1A1A;
        }
        QListWidget::item:hover {
            background: rgba(0,0,0,0.04);
        }
        QListWidget::item:selected {
            background: rgba(0,120,212,0.15);
            color: #0078D4;
        }
        
        QCheckBox {
            font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
            font-size: 14px;
            color: #1A1A1A;
            spacing: 8px;
        }
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border-radius: 4px;
            border: 1px solid rgba(0,0,0,0.2);
            background: rgba(255,255,255,0.6);
        }
        QCheckBox::indicator:checked {
            background: #0078D4;
            border: 1px solid #0078D4;
        }
        QCheckBox::indicator:hover {
            border: 1px solid rgba(0,0,0,0.35);
        }
        
        QSlider::groove:horizontal {
            border: none;
            height: 4px;
            border-radius: 2px;
            background: rgba(0,0,0,0.08);
        }
        QSlider::handle:horizontal {
            background: #0078D4;
            width: 16px;
            height: 16px;
            margin: -6px 0;
            border-radius: 8px;
        }
        QSlider::handle:horizontal:hover {
            background: #106EBE;
        }
        QSlider::sub-page:horizontal {
            background: #0078D4;
            border-radius: 2px;
        }
        
        QComboBox {
            font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
            font-size: 14px;
            padding: 6px 12px;
            border: 1px solid rgba(0,0,0,0.10);
            border-radius: 6px;
            background: rgba(255,255,255,0.60);
            color: #1A1A1A;
            min-height: 32px;
        }
        QComboBox:hover {
            border: 1px solid rgba(0,0,0,0.2);
        }
        QComboBox QAbstractItemView {
            font-size: 14px;
            background: #FFFFFF;
            color: #1A1A1A;
            border: 1px solid rgba(0,0,0,0.1);
            border-radius: 6px;
            selection-background-color: #0078D4;
            selection-color: #FFFFFF;
            outline: none;
        }
        """


class BlurBackground(QWidget):
    """玻璃背景绘制控件 — 模拟 Acrylic/Mica 效果，pixmap 缓存"""

    def __init__(self, parent=None, is_dark=False):
        super().__init__(parent)
        self.is_dark = is_dark
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground, False)
        self._cached_pixmap = None
        self._cached_size = None
        self._cached_dark = None

    def _render_blur(self, w, h, is_dark):
        pixmap = QPixmap(w, h)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        if is_dark:
            bg_color = QColor(28, 28, 28, 200)
            highlight = QColor(255, 255, 255, 8)
        else:
            bg_color = QColor(245, 247, 250, 200)
            highlight = QColor(255, 255, 255, 40)

        path = QPainterPath()
        path.addRoundedRect(0, 0, w, h, 12, 12)
        painter.fillPath(path, bg_color)

        highlight_rect = QRect(0, 0, w, int(h * 0.35))
        highlight_path = QPainterPath()
        highlight_path.addRoundedRect(QRectF(highlight_rect), 12, 12)
        painter.fillPath(highlight_path, highlight)

        pen = QPen(
            QColor(255, 255, 255, 60) if not is_dark else QColor(255, 255, 255, 15), 1
        )
        painter.setPen(pen)
        border_path = QPainterPath()
        border_path.addRoundedRect(0, 0, w - 1, h - 1, 12, 12)
        painter.drawPath(border_path)
        painter.end()
        return pixmap

    def paintEvent(self, event):
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            return

        cache_key = (w, h, self.is_dark)
        if (self._cached_pixmap is None or self._cached_size != (w, h)
                or self._cached_dark != self.is_dark):
            self._cached_pixmap = self._render_blur(w, h, self.is_dark)
            self._cached_size = (w, h)
            self._cached_dark = self.is_dark

        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._cached_pixmap)


class GlassPanel(QFrame):
    """玻璃面板 — 带 Acrylic 效果的容器，背景 pixmap 缓存"""

    def __init__(self, parent=None, is_dark=False):
        super().__init__(parent)
        self.is_dark = is_dark
        self.setObjectName("glassPanel")
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("")
        self._corner_radius = 12
        self._cached_pixmap = None
        self._cached_size = None
        self._cached_dark = None

    def _render_panel(self, w, h, is_dark):
        """渲染面板背景到 pixmap"""
        pixmap = QPixmap(w, h)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        if is_dark:
            bg = QColor(32, 32, 32, 190)
            top_light = QColor(255, 255, 255, 10)
            border = QColor(255, 255, 255, 15)
        else:
            bg = QColor(255, 255, 255, 210)
            top_light = QColor(255, 255, 255, 80)
            border = QColor(255, 255, 255, 120)

        path = QPainterPath()
        path.addRoundedRect(1, 1, w - 2, h - 2, self._corner_radius, self._corner_radius)
        painter.fillPath(path, bg)

        highlight_path = QPainterPath()
        highlight_path.addRoundedRect(
            QRectF(1, 1, w - 2, h * 0.3),
            self._corner_radius, self._corner_radius
        )
        painter.fillPath(highlight_path, top_light)

        pen = QPen(border, 1)
        painter.setPen(pen)
        border_path = QPainterPath()
        border_path.addRoundedRect(1, 1, w - 3, h - 3,
                                   self._corner_radius, self._corner_radius)
        painter.drawPath(border_path)
        painter.end()
        return pixmap

    def paintEvent(self, event):
        w, h = self.width(), self.height()
        if w <= 2 or h <= 2:
            return

        cache_key = (w, h, self.is_dark)
        if (self._cached_pixmap is None or self._cached_size != (w, h)
                or self._cached_dark != self.is_dark):
            self._cached_pixmap = self._render_panel(w, h, self.is_dark)
            self._cached_size = (w, h)
            self._cached_dark = self.is_dark

        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._cached_pixmap)


class BackgroundWidget(QWidget):
    """背景渐变绘制 — 缓存 pixmap，仅 resize 时重绘"""

    def __init__(self, parent=None, is_dark=False):
        super().__init__(parent)
        self.is_dark = is_dark
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._cached_pixmap = None
        self._cached_size = None
        self._cached_dark = None

    def _render_background(self, width, height, is_dark):
        """渲染背景到 pixmap 并缓存"""
        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        if is_dark:
            gradient = QLinearGradient(0, 0, width, height)
            gradient.setColorAt(0.0, QColor(20, 20, 24))
            gradient.setColorAt(0.5, QColor(24, 24, 30))
            gradient.setColorAt(1.0, QColor(18, 18, 22))
            painter.fillRect(0, 0, width, height, gradient)

            glow_center = QPoint(width // 3, height // 3)
            glow_radius = min(width, height) * 0.4
            glow = QRadialGradient(glow_center, glow_radius)
            glow.setColorAt(0.0, QColor(60, 60, 80, 30))
            glow.setColorAt(1.0, QColor(60, 60, 80, 0))
            painter.fillRect(0, 0, width, height, glow)
        else:
            gradient = QLinearGradient(0, 0, width, height)
            gradient.setColorAt(0.0, QColor(240, 243, 248))
            gradient.setColorAt(0.5, QColor(235, 238, 245))
            gradient.setColorAt(1.0, QColor(228, 232, 240))
            painter.fillRect(0, 0, width, height, gradient)

            glow_center = QPoint(width // 4, height // 4)
            glow_radius = min(width, height) * 0.5
            glow = QRadialGradient(glow_center, glow_radius)
            glow.setColorAt(0.0, QColor(255, 255, 255, 80))
            glow.setColorAt(0.5, QColor(255, 255, 255, 20))
            glow.setColorAt(1.0, QColor(255, 255, 255, 0))
            painter.fillRect(0, 0, width, height, glow)

        painter.end()
        return pixmap

    def paintEvent(self, event):
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            return

        # 仅在尺寸或主题变化时重建缓存
        cache_key = (w, h, self.is_dark)
        if (self._cached_pixmap is None or self._cached_size != (w, h)
                or self._cached_dark != self.is_dark):
            self._cached_pixmap = self._render_background(w, h, self.is_dark)
            self._cached_size = (w, h)
            self._cached_dark = self.is_dark

        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._cached_pixmap)


# ═══════════════════════════════════════════════════════════════
# 后台 ML 处理线程 — 集成时序追踪 + 质量过滤 + 编码缓存
# ═══════════════════════════════════════════════════════════════
class ProcessingThread(QThread):
    """后台 ML 处理线程 — 使用 QThread + signal"""
    frame_updated = pyqtSignal(np.ndarray, list, float)

    def __init__(self, camera, detector, recognizer, database):
        super().__init__()
        self.camera = camera
        self.detector = detector
        self.recognizer = recognizer
        self.db = database
        self.running = False
        self.paused = False
        self.lock = threading.Lock()
        self._frame_idx = 0
        self._latest_results = []
        self._latest_fps = 0.0

        # 时序追踪器
        from face_tracker import FaceTracker
        smooth = APP_SETTINGS.get("track_smooth", 5)
        self.tracker = FaceTracker(smooth_frames=smooth, iou_threshold=0.30, max_missed=10)

        # 处理帧率（0 = 不限，始终处理最新帧）
        proc_fps = APP_SETTINGS.get("proc_fps", 30)
        self._interval = 1.0 / proc_fps if proc_fps > 0 else 0

    def stop(self):
        self.running = False
        self.wait(3000)

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False
        if hasattr(self, 'tracker') and self.tracker:
            self.tracker.reset()

    def run(self):
        self.running = True
        frame_count = 0
        fps_timer = time.time()
        last_process_time = 0

        while self.running:
            if self.paused:
                time.sleep(0.050)
                continue

            # latest-frame 模式：如果 proc_fps=0 则不限制帧率
            if self._interval > 0:
                now = time.time()
                if now - last_process_time < self._interval:
                    time.sleep(0.001)
                    continue

            loop_start = time.time()

            frame = self.camera.get_frame()
            if frame is None:
                time.sleep(0.005)
                continue

            # frame 已经是 get_frame() 返回的副本，无需再 copy
            # 直接在 frame 上绘制
            display = frame

            # ── 编码缓存更新 ──
            known_encodings, known_names = self.db.get_encodings_and_names()
            self.recognizer.update_cache(known_encodings, known_names, self.db.version)

            # ── 检测 + 特征提取 ──
            try:
                faces_with_emb = self.detector.detect_with_embeddings(frame)
            except Exception:
                print("[ProcessingThread] ❌ detect_with_embeddings 抛出异常!")
                import traceback
                traceback.print_exc()
                time.sleep(0.005)
                continue

            # 诊断：前 3 帧 & 每 30 帧输出一次检测统计
            self._frame_idx += 1
            if self._frame_idx <= 3 or self._frame_idx % 30 == 1:
                n_faces = len(faces_with_emb)
                n_quality = sum(1 for f in faces_with_emb if f[6])  # quality_pass
                print(f"[ProcessingThread] 帧#{self._frame_idx}: "
                      f"检测到 {n_faces} 人脸, {n_quality} 通过质量检查, "
                      f"追踪数={self.tracker.track_count}, "
                      f"帧shape={frame.shape}")

            # ── 时序追踪更新（使用 recognizer 内置缓存） ──
            try:
                tracked_faces = self.tracker.update(
                    faces_with_emb,
                    recognizer=self.recognizer
                )
            except Exception:
                tracked_faces = []

            # ── 绘制结果 ──
            results = []
            for tf in tracked_faces:
                x1, y1, x2, y2 = tf['bbox']
                name = tf['name']
                conf = tf['conf']
                is_confirmed = tf['is_confirmed']

                # 颜色：已确认=绿色，未确认=蓝色，未知=灰色
                if is_confirmed:
                    color = (19, 161, 14)   # 绿色 — 确认身份
                elif name != "未知":
                    color = (0, 120, 215)   # 蓝色 — 识别中
                else:
                    color = (128, 128, 128) # 灰色 — 未知

                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

                # 标签
                if is_confirmed:
                    label = f"{name} {conf:.0%}"
                elif name != "未知":
                    label = f"{name}?"
                else:
                    label = "?"

                (tw, th), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1
                )
                label_y1 = max(0, y1 - th - 8)
                cv2.rectangle(display, (x1, label_y1), (x1 + tw + 8, y1), color, -1)
                cv2.putText(display, label, (x1 + 4, y1 - 4),
                           cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

                results.append((name, conf, (x1, y1, x2, y2, 0.0), is_confirmed))

            self._latest_results = results

            # ── FPS 统计 ──
            frame_count += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                self._latest_fps = frame_count / (now - fps_timer)
                frame_count = 0
                fps_timer = now

            self.frame_updated.emit(display, results, self._latest_fps)

            last_process_time = time.time()


# ═══════════════════════════════════════════════════════════════
# 主窗口
# ═══════════════════════════════════════════════════════════════
class FaceVisionApp(QMainWindow):
    """主应用程序窗口 — Windows 11 玻璃态仪表盘"""
    
    def __init__(self, camera, database, detector, recognizer):
        super().__init__()
        self.camera = camera
        self.db = database
        self.detector = detector
        self.recognizer = recognizer
        self.processing = None
        self.is_running = False
        self.is_dark = False
        self.face_photos_dir = Path("face_photos")
        self.face_photos_dir.mkdir(exist_ok=True)
        
        self.setWindowTitle("FaceVision — 实时人脸识别")
        self.setMinimumSize(1000, 680)
        self.resize(1200, 780)
        
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground, False)
        
        self._setup_ui()
        self._setup_timer()
        
        self._update_stylesheet()
    
    def _setup_ui(self):
        """构建主界面"""
        self.central = QWidget()
        self.central.setObjectName("centralWidget")
        self.setCentralWidget(self.central)
        
        main_layout = QVBoxLayout(self.central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        self.bg_widget = BackgroundWidget(self.central, self.is_dark)
        self.bg_widget.lower()
        
        content_wrapper = QWidget()
        content_wrapper.setAttribute(Qt.WA_TranslucentBackground)
        content_layout = QVBoxLayout(content_wrapper)
        content_layout.setContentsMargins(20, 16, 20, 16)
        content_layout.setSpacing(12)
        
        self._build_header(content_layout)
        
        body = QWidget()
        body.setAttribute(Qt.WA_TranslucentBackground)
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(16)
        
        self._build_camera_section(body_layout)
        
        self._build_control_section(body_layout)
        
        content_layout.addWidget(body, 1)
        
        self._build_status_bar(content_layout)
        
        main_layout.addWidget(content_wrapper)
    
    def _build_header(self, parent_layout):
        """构建顶部标题栏（不含最小化和关闭按钮）"""
        header = QWidget()
        header.setAttribute(Qt.WA_TranslucentBackground)
        header.setFixedHeight(48)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 4, 8, 4)
        
        title_label = QLabel("✦ FaceVision")
        title_font = QFont("Microsoft YaHei UI", 18, QFont.Bold)
        title_label.setFont(title_font)
        title_color = "#0078D4" if not self.is_dark else "#4CC2FF"
        title_label.setStyleSheet(f"color: {title_color}; background: transparent;")
        header_layout.addWidget(title_label)
        
        subtitle = QLabel("  实时人脸识别系统")
        subtitle_font = QFont("Microsoft YaHei UI", 11)
        subtitle.setFont(subtitle_font)
        subtitle.setStyleSheet("color: #888888; background: transparent; padding-top: 6px;")
        header_layout.addWidget(subtitle)
        
        header_layout.addStretch()
        
        self.theme_btn = QPushButton("🌙" if not self.is_dark else "☀️")
        self.theme_btn.setFixedSize(36, 36)
        self.theme_btn.setCursor(Qt.PointingHandCursor)
        self.theme_btn.clicked.connect(self._toggle_theme)
        header_layout.addWidget(self.theme_btn)
        
        parent_layout.addWidget(header)
    
    def _build_camera_section(self, parent_layout):
        """构建摄像头画面区域（玻璃面板）"""
        self.cam_panel = GlassPanel(self.central, self.is_dark)
        self.cam_panel.setMinimumWidth(520)
        cam_layout = QVBoxLayout(self.cam_panel)
        cam_layout.setContentsMargins(12, 12, 12, 12)
        cam_layout.setSpacing(8)
        
        self.video_label = QLabel()
        self.video_label.setMinimumSize(480, 320)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            background: rgba(0,0,0,0.04);
            border-radius: 8px;
            color: #999999;
            font-size: 15px;
        """ if not self.is_dark else """
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
            color: #888888;
            font-size: 15px;
        """)
        self.video_label.setText("📷 点击「启动摄像头」开始\n\n实时人脸识别")
        cam_layout.addWidget(self.video_label, 1)
        
        cam_controls = QWidget()
        cam_controls.setAttribute(Qt.WA_TranslucentBackground)
        cam_ctrl_layout = QHBoxLayout(cam_controls)
        cam_ctrl_layout.setContentsMargins(0, 0, 0, 0)
        
        self.btn_camera = QPushButton("▶  启动摄像头")
        self.btn_camera.setObjectName("btnSuccess")
        self.btn_camera.setFixedHeight(40)
        self.btn_camera.setCursor(Qt.PointingHandCursor)
        self.btn_camera.clicked.connect(self._toggle_camera)
        cam_ctrl_layout.addWidget(self.btn_camera)
        
        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setStyleSheet(
            "color: #888888; background: transparent; font-size: 14px; padding: 0 12px;"
        )
        self.fps_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        cam_ctrl_layout.addWidget(self.fps_label)
        
        cam_layout.addWidget(cam_controls)
        
        parent_layout.addWidget(self.cam_panel, 3)
    
    def _build_control_section(self, parent_layout):
        """构建右侧控制面板（玻璃面板）"""
        self.ctrl_panel = GlassPanel(self.central, self.is_dark)
        self.ctrl_panel.setMaximumWidth(340)
        self.ctrl_panel.setMinimumWidth(280)
        ctrl_layout = QVBoxLayout(self.ctrl_panel)
        ctrl_layout.setContentsMargins(16, 16, 16, 16)
        ctrl_layout.setSpacing(8)
        
        section_title = QLabel("控制面板")
        section_title.setStyleSheet(
            f"color: #0078D4; font-size: 16px; font-weight: bold; background: transparent;"
            if not self.is_dark else
            f"color: #4CC2FF; font-size: 16px; font-weight: bold; background: transparent;"
        )
        ctrl_layout.addWidget(section_title)
        
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("background: rgba(0,0,0,0.06); max-height: 1px;" if not self.is_dark
                         else "background: rgba(255,255,255,0.08); max-height: 1px;")
        ctrl_layout.addWidget(sep)
        
        btn_style = "min-height: 38px; font-size: 14px;"
        
        self.btn_add = QPushButton("＋  添加人员")
        self.btn_add.setObjectName("btnAccent")
        self.btn_add.setStyleSheet(btn_style)
        self.btn_add.setCursor(Qt.PointingHandCursor)
        self.btn_add.clicked.connect(self._show_add_dialog)
        ctrl_layout.addWidget(self.btn_add)
        
        self.btn_import = QPushButton("📁  从图片导入")
        self.btn_import.setStyleSheet(btn_style)
        self.btn_import.setCursor(Qt.PointingHandCursor)
        self.btn_import.clicked.connect(self._import_from_file)
        ctrl_layout.addWidget(self.btn_import)
        
        self.btn_remove = QPushButton("✕  删除选中")
        self.btn_remove.setObjectName("btnDanger")
        self.btn_remove.setStyleSheet(btn_style)
        self.btn_remove.setCursor(Qt.PointingHandCursor)
        self.btn_remove.clicked.connect(self._remove_selected)
        ctrl_layout.addWidget(self.btn_remove)
        
        self.btn_settings = QPushButton("⚙  设置")
        self.btn_settings.setStyleSheet(btn_style)
        self.btn_settings.setCursor(Qt.PointingHandCursor)
        self.btn_settings.clicked.connect(self._show_settings)
        ctrl_layout.addWidget(self.btn_settings)
        
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet("background: rgba(0,0,0,0.06); max-height: 1px; margin-top: 4px;" 
                          if not self.is_dark else
                          "background: rgba(255,255,255,0.08); max-height: 1px; margin-top: 4px;")
        ctrl_layout.addWidget(sep2)
        
        person_title = QLabel("已注册人员")
        person_title.setStyleSheet(
            "color: #0078D4; font-size: 14px; font-weight: bold; background: transparent; padding-top: 4px;"
            if not self.is_dark else
            "color: #4CC2FF; font-size: 14px; font-weight: bold; background: transparent; padding-top: 4px;"
        )
        ctrl_layout.addWidget(person_title)
        
        self.person_list = QListWidget()
        self.person_list.setSpacing(3)
        ctrl_layout.addWidget(self.person_list, 1)
        
        result_title = QLabel("识别结果")
        result_title.setStyleSheet(
            "color: #0078D4; font-size: 14px; font-weight: bold; background: transparent; padding-top: 4px;"
            if not self.is_dark else
            "color: #4CC2FF; font-size: 14px; font-weight: bold; background: transparent; padding-top: 4px;"
        )
        ctrl_layout.addWidget(result_title)
        
        self.result_label = QLabel("等待识别…")
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet(
            "color: #888888; background: transparent; font-size: 13px; padding: 4px 0;"
        )
        ctrl_layout.addWidget(self.result_label)
        
        parent_layout.addWidget(self.ctrl_panel, 2)
    
    def _build_status_bar(self, parent_layout):
        """构建底部状态栏"""
        status_panel = GlassPanel(self.central, self.is_dark)
        status_panel.setFixedHeight(42)
        status_panel._corner_radius = 8
        status_layout = QHBoxLayout(status_panel)
        status_layout.setContentsMargins(16, 0, 16, 0)
        
        self.status_label = QLabel("● 就绪")
        self.status_label.setStyleSheet(
            "color: #888888; background: transparent; font-size: 13px;"
        )
        status_layout.addWidget(self.status_label)
        
        status_layout.addStretch()
        
        info_label = QLabel("FaceVision v2.0 · PyQt5")
        info_label.setStyleSheet(
            "color: #AAAAAA; background: transparent; font-size: 12px;"
        )
        status_layout.addWidget(info_label)
        
        parent_layout.addWidget(status_panel)
    
    def _setup_timer(self):
        """显示刷新定时器"""
        self.display_timer = QTimer()
        self.display_timer.timeout.connect(self._update_display)
        self.display_timer.start(50)
    
    def _update_stylesheet(self):
        """更新全局样式"""
        style = get_glass_stylesheet(self.is_dark)
        self.setStyleSheet(style)
    
    def resizeEvent(self, event):
        """窗口大小变化时更新背景"""
        super().resizeEvent(event)
        if hasattr(self, 'bg_widget'):
            self.bg_widget.setGeometry(self.rect())
    
    def _toggle_theme(self):
        """切换浅色/深色主题"""
        self.is_dark = not self.is_dark
        self.theme_btn.setText("🌙" if not self.is_dark else "☀️")
        self._update_stylesheet()
        self.bg_widget.is_dark = self.is_dark
        self.cam_panel.is_dark = self.is_dark
        self.ctrl_panel.is_dark = self.is_dark
        for widget in self.findChildren(GlassPanel):
            widget.is_dark = self.is_dark
            widget.update()
        self.bg_widget.update()
        self.update()
    
    def _update_display(self):
        """定时更新画面显示"""
        if self.is_running and self.processing:
            fps = self._current_fps if hasattr(self, '_current_fps') else 0
            self.fps_label.setText(f"FPS: {fps:.0f}")
    
    @pyqtSlot(np.ndarray, list, float)
    def _on_frame_updated(self, frame, results, fps):
        """处理线程发来的新帧（result 格式已包含 is_confirmed 标记）"""
        self._current_fps = fps
        self.fps_label.setText(f"ML FPS: {fps:.0f}")
        self._show_frame(frame)

        if results:
            lines = []
            for item in results:
                name, conf = item[0], item[1]
                is_confirmed = item[3] if len(item) > 3 else (name != "未知")
                if is_confirmed and name != "未知":
                    lines.append(f"✓ {name}  {conf:.0%}")
                elif name != "未知":
                    lines.append(f"⟳ {name}? 识别中…")
                else:
                    lines.append("? 未知人员")
            color = "#1A1A1A" if not self.is_dark else "#FFFFFF"
            self.result_label.setStyleSheet(f"color: {color}; background: transparent; font-size: 13px;")
            self.result_label.setText("\n".join(lines))
        else:
            self.result_label.setText("正在检测…")
    
    def _show_frame(self, frame):
        """在 QLabel 上显示 OpenCV 帧"""
        w = self.video_label.width() - 4
        h = self.video_label.height() - 4
        if w < 10 or h < 10:
            w, h = 640, 480
        
        fh, fw = frame.shape[:2]
        scale = min(w / fw, h / fh)
        new_w, new_h = int(fw * scale), int(fh * scale)
        
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)
        
        rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        
        self.video_label.setPixmap(pixmap)
        self.video_label.setAlignment(Qt.AlignCenter)
    
    # ── 摄像头控制 ──
    def _toggle_camera(self):
        if not self.is_running:
            try:
                self.camera.start()
                self.is_running = True
                
                self.processing = ProcessingThread(
                    self.camera, self.detector,
                    self.recognizer, self.db
                )
                self.processing.frame_updated.connect(self._on_frame_updated)
                self.processing.start()
                
                self.btn_camera.setText("⏹  停止摄像头")
                self.btn_camera.setObjectName("btnDanger")
                self.btn_camera.setStyleSheet("min-height: 38px; font-size: 14px;")
                self.btn_camera.update()
                self._set_status("● 运行中", "#107C10")
            except RuntimeError as e:
                QMessageBox.critical(self, "摄像头错误", str(e))
        else:
            if self.processing:
                self.processing.stop()
                self.processing = None
            self.camera.stop()
            self.is_running = False
            
            self.btn_camera.setText("▶  启动摄像头")
            self.btn_camera.setObjectName("btnSuccess")
            self.btn_camera.setStyleSheet("min-height: 38px; font-size: 14px;")
            self.btn_camera.update()
            
            self.video_label.setPixmap(QPixmap())
            self.video_label.setText("📷 点击「启动摄像头」开始\n\n实时人脸识别")
            self._set_status("● 就绪", "#888888")
            self.fps_label.setText("FPS: 0")
            self.result_label.setText("等待识别…")
    
    # ── 人员管理 ──
    def _show_add_dialog(self):
        dialog = AddPersonDialogPyQt(self)
        if dialog.exec_() == QDialog.Accepted:
            name = dialog.get_result()
            if name:
                self._capture_and_add(name)
    
    def _import_from_file(self):
        dialog = AddPersonDialogPyQt(self, "输入人员姓名", "请输入要导入的人员姓名：")
        if dialog.exec_() != QDialog.Accepted:
            return
        name = dialog.get_result()
        if not name:
            return
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择包含人脸的图片", "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*)"
        )
        if not file_path:
            return
        self._add_from_image(name, file_path)
    
    def _show_settings(self):
        dialog = SettingsDialogPyQt(self)
        dialog.exec_()
    
    def _remove_selected(self):
        names = self.db.get_names()
        if not names:
            return
        
        dialog = BatchDeleteDialogPyQt(self, names)
        if dialog.exec_() == QDialog.Accepted:
            selected = dialog.get_result()
            if selected:
                msg = f"确定要删除以下 {len(selected)} 名人员吗？\n该操作不可撤销。\n\n" + "\n".join(f"• {n}" for n in selected)
                reply = QMessageBox.question(
                    self, "确认批量删除", msg,
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    removed, not_found = self.db.remove_persons(selected)
                    self._refresh_person_list()
                    if not_found:
                        self._set_status(f"✓ 已删除 {len(removed)} 人，{len(not_found)} 人未找到",
                                        "#107C10")
                    else:
                        self._set_status(f"✓ 已删除 {len(removed)} 人", "#107C10")
    
    def _refresh_person_list(self):
        """刷新人员列表"""
        self.person_list.clear()
        names = self.db.get_names()
        if not names:
            item = QListWidgetItem("暂无注册人员")
            item.setFlags(item.flags() & ~Qt.ItemIsSelectable)
            self.person_list.addItem(item)
        else:
            for name in sorted(names):
                item = QListWidgetItem(f"  👤  {name}")
                self.person_list.addItem(item)
    
    def _capture_and_add(self, name):
        """多帧注册（保持原有逻辑）"""
        if not self.is_running:
            QMessageBox.warning(self, "无画面", "请先启动摄像头再添加人员。")
            return
        
        if self.processing:
            self.processing.pause()
            QTimer.singleShot(100, lambda: self._do_capture(name))
        else:
            self._do_capture(name)
    
    def _do_capture(self, name):
        """实际多帧采集（带质量优选）"""
        self._set_status("⏳ 正在采集人脸，请保持不动…", "#0078D4")

        frame = self.camera.get_frame()
        if frame is None:
            print("[_do_capture] ❌ get_frame() 返回 None!")
            self._set_status("✗ 获取画面失败", "#C42B1C")
            self._resume_processing()
            return

        print(f"[_do_capture] 帧shape={frame.shape}, dtype={frame.dtype}")

        faces = self.detector.detect(frame)  # 只用检测，不做 embedding
        print(f"[_do_capture] detect() 返回 {len(faces)} 个人脸, "
              f"confidence阈值={self.detector.confidence:.2f}")
        if len(faces) == 0:
            self._set_status("✗ 未检测到人脸", "#C42B1C")
            QMessageBox.warning(self, "未检测到", "画面中未检测到清晰人脸。")
            self._resume_processing()
            return

        if len(faces) == 1:
            target_idx = 0
        else:
            # 构建临时 faces 列表给选择对话框（兼容旧格式）
            temp_faces = [(x1, y1, x2, y2, conf, None) for (x1, y1, x2, y2, conf) in faces]
            selector = SelectFaceDialogPyQt(self, frame, temp_faces, name)
            if selector.exec_() != QDialog.Accepted:
                self._set_status("● 已取消", "#888888")
                self._resume_processing()
                return
            target_idx = selector.get_result()
            if target_idx is None:
                self._set_status("● 已取消", "#888888")
                self._resume_processing()
                return

        x1, y1, x2, y2, _ = faces[target_idx]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        self._set_status("⏳ 正在采集多帧，请保持不动…", "#0078D4")

        # ── 多帧采集 + 质量评分 ──
        candidates = []  # [(embedding, roi, blur_score)]

        for i in range(15):
            frame = self.camera.get_frame()
            if frame is None:
                continue

            current_faces = self.detector.detect_with_embeddings(frame)
            if len(current_faces) == 0:
                continue

            min_dist = float('inf')
            matched = None
            for f in current_faces:
                fx1, fy1, fx2, fy2, det_conf, emb, quality_pass = f
                f_cx = (fx1 + fx2) / 2
                f_cy = (fy1 + fy2) / 2
                dist = (f_cx - cx) ** 2 + (f_cy - cy) ** 2
                if dist < min_dist and dist < (frame.shape[1] * 0.2) ** 2:
                    min_dist = dist
                    matched = f

            if matched is None:
                continue

            mx1, my1, mx2, my2, mdet_conf, membedding, mquality = matched
            if membedding is None:
                continue

            # 计算人脸 ROI 清晰度
            roi = self.detector.extract_face_roi(
                frame, (mx1, my1, mx2, my2, mdet_conf)
            )
            if roi.size == 0:
                continue

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            candidates.append((membedding, roi, blur_score))

        if len(candidates) < 3:
            self._set_status("✗ 采集失败，请正对摄像头", "#C42B1C")
            QMessageBox.warning(
                self, "采集失败",
                f"仅采集到 {len(candidates)}/3 帧有效人脸，请正对摄像头保持不动。"
            )
            self._resume_processing()
            return

        # ── 质量优选：取前 2/3 最清晰的帧做 median ──
        candidates.sort(key=lambda x: x[2], reverse=True)  # 按清晰度降序
        top_n = max(3, int(len(candidates) * 0.67))
        best_candidates = candidates[:top_n]

        enc_stack = np.stack([c[0] for c in best_candidates])
        encoding = np.median(enc_stack, axis=0)
        encoding = encoding / (np.linalg.norm(encoding) + 1e-8)

        # 选最清晰的 ROI 保存
        best_roi = best_candidates[0][1]  # 已按清晰度排序，第一个就是最清晰的

        filename = f"{name}_{uuid.uuid4().hex[:8]}.jpg"
        img_path = self.face_photos_dir / filename
        cv2.imwrite(str(img_path), best_roi)

        success, msg = self.db.add_person(name, str(img_path), encoding)
        if success:
            # 编码缓存重建
            known_encodings, known_names = self.db.get_encodings_and_names()
            self.recognizer.update_cache(known_encodings, known_names, self.db.version)
            self._refresh_person_list()
            self._set_status(f"✓ {msg}", "#107C10")
        else:
            QMessageBox.warning(self, "添加失败", msg)
            self._set_status("● 就绪", "#888888")

        self._resume_processing()
    
    def _add_from_image(self, name, file_path):
        """从图片文件注册（适配新 detect_with_embeddings 7元素格式）"""
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
                selector = SelectFaceDialogPyQt(self, img, faces, name + "（从图片）")
                if selector.exec_() != QDialog.Accepted:
                    return
                target_idx = selector.get_result()
                if target_idx is None:
                    return

            best = faces[target_idx]
            x1, y1, x2, y2, det_conf = best[0], best[1], best[2], best[3], best[4]
            embedding = best[5]  # 第6个元素（索引5）是 embedding
            if embedding is None:
                QMessageBox.warning(self, "编码失败", "无法提取人脸特征。")
                return

            ext = Path(file_path).suffix
            filename = f"{name}_{uuid.uuid4().hex[:8]}{ext}"
            dest_path = self.face_photos_dir / filename
            roi = self.detector.extract_face_roi(img, (x1, y1, x2, y2, det_conf))
            cv2.imwrite(str(dest_path), roi)

            success, msg = self.db.add_person(name, str(dest_path), embedding)
            if success:
                # 编码缓存重建
                known_encodings, known_names = self.db.get_encodings_and_names()
                self.recognizer.update_cache(known_encodings, known_names, self.db.version)
                self._refresh_person_list()
                self._set_status(f"✓ {msg}", "#107C10")
            else:
                QMessageBox.warning(self, "添加失败", msg)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导入失败: {e}")
    
    def _resume_processing(self):
        if self.processing:
            self.processing.resume()
    
    def _set_status(self, text, color="#888888"):
        self.status_label.setText(text)
        self.status_label.setStyleSheet(
            f"color: {color}; background: transparent; font-size: 13px;"
        )
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        if self.processing:
            self.processing.stop()
        if self.is_running:
            self.camera.stop()
        event.accept()


# ═══════════════════════════════════════════════════════════════
# 对话框 — 玻璃态风格
# ═══════════════════════════════════════════════════════════════

class GlassDialog(QDialog):
    """带玻璃效果的基础对话框 — 支持拖动"""
    
    def __init__(self, parent=None, title="", is_dark=False):
        super().__init__(parent)
        self.is_dark = is_dark
        self.setWindowTitle(title)
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setModal(True)
        self.setMinimumSize(380, 200)
        self._drag_pos = None
        
        if parent:
            self.resize(400, 220)
            parent_geo = parent.geometry()
            x = parent_geo.center().x() - 200
            y = parent_geo.center().y() - 110
            self.move(x, y)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()
    
    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self._drag_pos is not None:
            self.move(event.globalPos() - self._drag_pos)
            event.accept()
    
    def mouseReleaseEvent(self, event):
        self._drag_pos = None
        event.accept()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w, h = self.width(), self.height()
        margin = 6
        
        if self.is_dark:
            bg = QColor(35, 35, 40, 235)
            border = QColor(255, 255, 255, 20)
            shadow = QColor(0, 0, 0, 80)
        else:
            bg = QColor(248, 249, 252, 240)
            border = QColor(255, 255, 255, 180)
            shadow = QColor(0, 0, 0, 40)
        
        # 阴影绘制在背景后面，全部在窗口边界内
        painter.setPen(Qt.NoPen)
        for i in range(3):
            painter.setBrush(QColor(shadow.red(), shadow.green(), shadow.blue(), 
                                   shadow.alpha() - i * 20))
            offset = 2 + i * 2
            shadow_rect = QRectF(offset, offset + 3, w - offset * 2, h - offset * 2 - 6)
            painter.drawRoundedRect(shadow_rect, 12, 12)
        
        # 背景覆盖布局可用区域（四周各留 margin）
        content_rect = QRectF(margin, margin, w - margin * 2, h - margin * 2)
        painter.setBrush(bg)
        pen = QPen(border, 1)
        painter.setPen(pen)
        painter.drawRoundedRect(content_rect, 12, 12)


class AddPersonDialogPyQt(GlassDialog):
    """添加人员对话框"""
    
    def __init__(self, parent=None, title="添加人员", prompt="请输入人员姓名："):
        super().__init__(parent, title, parent.is_dark if parent else False)
        self.result = None
        self.setMinimumSize(380, 200)
        self.resize(400, 200)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(28, 28, 28, 24)
        layout.setSpacing(16)
        
        label = QLabel(prompt)
        label.setStyleSheet(
            "color: #1A1A1A; font-size: 15px; background: transparent;"
            if not self.is_dark else
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
        super().__init__(parent, "批量删除人员", parent.is_dark if parent else False)
        self.result = []
        self.names = names or []
        self.setMinimumSize(400, 420)
        self.resize(420, 440)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)
        
        label = QLabel("勾选要删除的人员：")
        label.setStyleSheet(
            "color: #1A1A1A; font-size: 15px; font-weight: bold; background: transparent;"
            if not self.is_dark else
            "color: #FFFFFF; font-size: 15px; font-weight: bold; background: transparent;"
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
        scroll_widget.setAttribute(Qt.WA_TranslucentBackground)
        scroll_widget.setStyleSheet("background: transparent;")
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(4)
        
        self.checkboxes = {}
        for name in sorted(self.names):
            cb = QCheckBox(f"  {name}")
            cb.setStyleSheet(
                "QCheckBox { background: transparent; color: #1A1A1A; font-size: 14px; spacing: 8px; }"
                "QCheckBox::indicator { width: 18px; height: 18px; border-radius: 4px; "
                "border: 1px solid rgba(0,0,0,0.2); background: rgba(255,255,255,0.6); }"
                "QCheckBox::indicator:checked { background: #0078D4; border: 1px solid #0078D4; }"
                if not self.is_dark else
                "QCheckBox { background: transparent; color: #FFFFFF; font-size: 14px; spacing: 8px; }"
                "QCheckBox::indicator { width: 18px; height: 18px; border-radius: 4px; "
                "border: 1px solid rgba(255,255,255,0.25); background: transparent; }"
                "QCheckBox::indicator:checked { background: #0078D4; border: 1px solid #0078D4; }"
            )
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
        super().__init__(parent, "选择要注册的人脸", parent.is_dark if parent else False)
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
            "color: #1A1A1A; font-size: 14px; background: transparent;"
            if not self.is_dark else
            "color: #FFFFFF; font-size: 14px; background: transparent;"
        )
        layout.addWidget(label)
        
        grid_widget = QWidget()
        grid_widget.setAttribute(Qt.WA_TranslucentBackground)
        grid_layout = QHBoxLayout(grid_widget)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setSpacing(12)
        
        for i, face in enumerate(self.faces):
            x1, y1, x2, y2, det_conf, _ = face
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
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_img)
            
            face_card = QFrame()
            face_card.setObjectName("glassCard")
            face_card.setFixedSize(160, 190)
            face_card.setCursor(Qt.PointingHandCursor)
            face_card.setStyleSheet("""
                QFrame#glassCard {
                    background: rgba(255,255,255,0.55);
                    border: 1px solid rgba(255,255,255,0.4);
                    border-radius: 8px;
                }
                QFrame#glassCard:hover {
                    background: rgba(0,120,212,0.15);
                    border: 1px solid #0078D4;
                }
            """ if not self.is_dark else """
                QFrame#glassCard {
                    background: rgba(255,255,255,0.06);
                    border: 1px solid rgba(255,255,255,0.08);
                    border-radius: 8px;
                }
                QFrame#glassCard:hover {
                    background: rgba(0,120,212,0.2);
                    border: 1px solid #0078D4;
                }
            """)
            
            card_layout = QVBoxLayout(face_card)
            card_layout.setContentsMargins(8, 8, 8, 8)
            card_layout.setAlignment(Qt.AlignCenter)
            
            img_label = QLabel()
            img_label.setPixmap(pixmap)
            img_label.setFixedSize(120, 120)
            img_label.setAlignment(Qt.AlignCenter)
            img_label.setStyleSheet("border-radius: 6px; background: rgba(0,0,0,0.05);")
            card_layout.addWidget(img_label, 0, Qt.AlignCenter)
            
            text_label = QLabel(f"人脸 #{i+1}  ({det_conf:.0%})")
            text_label.setAlignment(Qt.AlignCenter)
            text_label.setStyleSheet(
                "color: #666666; font-size: 12px; background: transparent; padding-top: 4px;"
                if not self.is_dark else
                "color: #BBBBBB; font-size: 12px; background: transparent; padding-top: 4px;"
            )
            card_layout.addWidget(text_label, 0, Qt.AlignCenter)
            
            face_card.mousePressEvent = lambda e, idx=i: self._select(idx)
            
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


class SettingsDialogPyQt(GlassDialog):
    """设置对话框 — 支持在程序内修改所有设置参数"""
    
    def __init__(self, parent=None):
        super().__init__(parent, "设置", parent.is_dark if parent else False)
        self.parent_app = parent
        self.setMinimumSize(560, 780)
        self.resize(560, 800)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(12)
        
        # ── 标题 ──
        title = QLabel("⚙  FaceVision 设置")
        title.setStyleSheet(
            "color: #0078D4; font-size: 18px; font-weight: bold; background: transparent;"
            if not self.is_dark else
            "color: #4CC2FF; font-size: 18px; font-weight: bold; background: transparent;"
        )
        layout.addWidget(title)
        
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("background: rgba(0,0,0,0.06); max-height: 1px;" if not self.is_dark
                         else "background: rgba(255,255,255,0.08); max-height: 1px;")
        layout.addWidget(sep)
        
        # ── 推理设备 ──
        self._add_section_label(layout, "推理设备")
        
        self.device_var = "cuda" if APP_SETTINGS.get("device") == "cuda" else "cpu"
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

        device_frame = QWidget()
        device_frame.setAttribute(Qt.WA_TranslucentBackground)
        device_layout = QHBoxLayout(device_frame)
        device_layout.setContentsMargins(0, 0, 0, 0)
        device_layout.setSpacing(12)

        self.btn_cpu = QPushButton("CPU")
        self.btn_cpu.setMinimumWidth(180)
        self.btn_cpu.setFixedHeight(38)
        self.btn_cpu.setCursor(Qt.PointingHandCursor)
        self.btn_cpu.clicked.connect(lambda: self._on_device_change("cpu"))
        device_layout.addWidget(self.btn_cpu)

        self.btn_gpu = QPushButton(gpu_label if gpu_available else "GPU (不可用)")
        self.btn_gpu.setMinimumWidth(180)
        self.btn_gpu.setFixedHeight(38)
        self.btn_gpu.setCursor(Qt.PointingHandCursor)
        if not gpu_available:
            self.btn_gpu.setEnabled(False)
        self.btn_gpu.clicked.connect(lambda: self._on_device_change("cuda"))
        device_layout.addWidget(self.btn_gpu)
        
        self._update_device_buttons()
        layout.addWidget(device_frame)
        
        # ── 检测置信度 ──
        self._add_section_label(layout, "检测置信度")
        
        conf_row = QWidget()
        conf_row.setAttribute(Qt.WA_TranslucentBackground)
        conf_layout = QHBoxLayout(conf_row)
        conf_layout.setContentsMargins(0, 0, 0, 0)
        
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setMinimum(30)
        self.conf_slider.setMaximum(80)
        self.conf_slider.setSingleStep(1)
        self.conf_slider.setValue(int(APP_SETTINGS.get("confidence", 0.50) * 100))
        self.conf_slider.valueChanged.connect(self._update_conf_label)
        conf_layout.addWidget(self.conf_slider, 1)
        
        self.conf_label = QLabel(f"{APP_SETTINGS.get('confidence', 0.25):.2f}")
        self.conf_label.setFixedWidth(50)
        self.conf_label.setAlignment(Qt.AlignCenter)
        self.conf_label.setStyleSheet(
            "color: #1A1A1A; font-size: 15px; font-weight: bold; background: transparent;"
            if not self.is_dark else
            "color: #FFFFFF; font-size: 15px; font-weight: bold; background: transparent;"
        )
        conf_layout.addWidget(self.conf_label)
        layout.addWidget(conf_row)
        
        # ── 识别容差 ──
        self._add_section_label(layout, "识别容差")
        
        tol_row = QWidget()
        tol_row.setAttribute(Qt.WA_TranslucentBackground)
        tol_layout = QHBoxLayout(tol_row)
        tol_layout.setContentsMargins(0, 0, 0, 0)
        
        self.tol_slider = QSlider(Qt.Horizontal)
        self.tol_slider.setMinimum(30)
        self.tol_slider.setMaximum(80)
        self.tol_slider.setSingleStep(1)
        self.tol_slider.setValue(int(APP_SETTINGS.get("tolerance", 0.45) * 100))
        self.tol_slider.valueChanged.connect(self._update_tol_label)
        tol_layout.addWidget(self.tol_slider, 1)
        
        self.tol_label = QLabel(f"{APP_SETTINGS.get('tolerance', 0.45):.2f}")
        self.tol_label.setFixedWidth(50)
        self.tol_label.setAlignment(Qt.AlignCenter)
        self.tol_label.setStyleSheet(
            "color: #1A1A1A; font-size: 15px; font-weight: bold; background: transparent;"
            if not self.is_dark else
            "color: #FFFFFF; font-size: 15px; font-weight: bold; background: transparent;"
        )
        tol_layout.addWidget(self.tol_label)
        layout.addWidget(tol_row)
        
        # ── 处理帧率 ──
        self._add_section_label(layout, "处理帧率 (FPS)")
        
        fps_row = QWidget()
        fps_row.setAttribute(Qt.WA_TranslucentBackground)
        fps_layout = QHBoxLayout(fps_row)
        fps_layout.setContentsMargins(0, 0, 0, 0)
        
        self.fps_slider = QSlider(Qt.Horizontal)
        self.fps_slider.setMinimum(5)
        self.fps_slider.setMaximum(60)
        self.fps_slider.setSingleStep(1)
        self.fps_slider.setValue(int(APP_SETTINGS.get("proc_fps", 12)))
        self.fps_slider.valueChanged.connect(self._update_fps_label)
        fps_layout.addWidget(self.fps_slider, 1)
        
        self.fps_label_w = QLabel(f"{APP_SETTINGS.get('proc_fps', 12):.0f}")
        self.fps_label_w.setFixedWidth(50)
        self.fps_label_w.setAlignment(Qt.AlignCenter)
        self.fps_label_w.setStyleSheet(
            "color: #1A1A1A; font-size: 15px; font-weight: bold; background: transparent;"
            if not self.is_dark else
            "color: #FFFFFF; font-size: 15px; font-weight: bold; background: transparent;"
        )
        fps_layout.addWidget(self.fps_label_w)
        layout.addWidget(fps_row)

        # ── 检测模型尺寸 ──
        self._add_section_label(layout, "检测模型尺寸 (越小越快)")

        det_size_row = QWidget()
        det_size_row.setAttribute(Qt.WA_TranslucentBackground)
        det_size_layout = QHBoxLayout(det_size_row)
        det_size_layout.setContentsMargins(0, 0, 0, 0)
        det_size_layout.setSpacing(8)

        self.det_size_combo = QComboBox()
        self.det_size_combo.addItems(["320 (快速)", "480 (均衡)", "640 (精准)"])
        current_det = APP_SETTINGS.get("det_size", 480)
        if current_det == 320:
            self.det_size_combo.setCurrentIndex(0)
        elif current_det == 640:
            self.det_size_combo.setCurrentIndex(2)
        else:
            self.det_size_combo.setCurrentIndex(1)
        det_size_layout.addWidget(self.det_size_combo)

        det_note = QLabel("需重启程序生效")
        det_note.setStyleSheet("color: #888888; font-size: 11px; background: transparent;")
        det_size_layout.addWidget(det_note)
        det_size_layout.addStretch()
        layout.addWidget(det_size_row)

        # ── 追踪平滑帧数 ──
        self._add_section_label(layout, "追踪平滑帧数")

        smooth_row = QWidget()
        smooth_row.setAttribute(Qt.WA_TranslucentBackground)
        smooth_layout = QHBoxLayout(smooth_row)
        smooth_layout.setContentsMargins(0, 0, 0, 0)

        self.smooth_slider = QSlider(Qt.Horizontal)
        self.smooth_slider.setMinimum(3)
        self.smooth_slider.setMaximum(10)
        self.smooth_slider.setSingleStep(1)
        self.smooth_slider.setValue(int(APP_SETTINGS.get("track_smooth", 5)))
        self.smooth_slider.valueChanged.connect(
            lambda v: self.smooth_label.setText(f"{v} 帧"))
        smooth_layout.addWidget(self.smooth_slider, 1)

        self.smooth_label = QLabel(f"{APP_SETTINGS.get('track_smooth', 5)} 帧")
        self.smooth_label.setFixedWidth(50)
        self.smooth_label.setAlignment(Qt.AlignCenter)
        self.smooth_label.setStyleSheet(
            "color: #1A1A1A; font-size: 15px; font-weight: bold; background: transparent;"
            if not self.is_dark else
            "color: #FFFFFF; font-size: 15px; font-weight: bold; background: transparent;"
        )
        smooth_layout.addWidget(self.smooth_label)
        layout.addWidget(smooth_row)

        # ── 质量过滤 ──
        self._add_section_label(layout, "质量过滤")

        qf_row = QWidget()
        qf_row.setAttribute(Qt.WA_TranslucentBackground)
        qf_layout = QHBoxLayout(qf_row)
        qf_layout.setContentsMargins(0, 0, 0, 0)
        qf_layout.setSpacing(12)

        self.qf_check = QCheckBox("启用模糊度过滤")
        self.qf_check.setChecked(APP_SETTINGS.get("quality_filter", True))
        qf_layout.addWidget(self.qf_check)

        self.minface_combo = QComboBox()
        self.minface_combo.addItems(["最小 60px", "最小 80px", "最小 100px", "最小 120px"])
        current_min = APP_SETTINGS.get("min_face_size", 80)
        minface_map = {60: 0, 80: 1, 100: 2, 120: 3}
        self.minface_combo.setCurrentIndex(minface_map.get(current_min, 1))
        qf_layout.addWidget(self.minface_combo)
        qf_layout.addStretch()
        layout.addWidget(qf_row)

        # ── 摄像头分辨率 ──
        self._add_section_label(layout, "摄像头分辨率")
        
        self.res_combo = QComboBox()
        self.res_combo.addItems(RESOLUTIONS)
        current_res = _current_resolution_key()
        if current_res in RESOLUTIONS:
            self.res_combo.setCurrentText(current_res)
        layout.addWidget(self.res_combo)
        
        # ── 提示 ──
        note = QLabel("💡 部分更改将在下次启动摄像头或重启程序后生效。")
        note.setWordWrap(True)
        note.setStyleSheet(
            "color: #888888; font-size: 12px; background: transparent; padding: 4px 0;"
        )
        layout.addWidget(note)
        
        layout.addStretch()
        
        # ── 按钮行 ──
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)
        btn_layout.addStretch()
        
        cancel_btn = QPushButton("取消")
        cancel_btn.setFixedSize(100, 38)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        apply_btn = QPushButton("应用")
        apply_btn.setObjectName("btnAccent")
        apply_btn.setFixedSize(100, 38)
        apply_btn.clicked.connect(self._apply)
        btn_layout.addWidget(apply_btn)
        
        layout.addLayout(btn_layout)
        
        self.setLayout(layout)
    
    def _add_section_label(self, layout, text):
        lbl = QLabel(text)
        lbl.setStyleSheet(
            "color: #0078D4; font-size: 14px; font-weight: bold; background: transparent; padding-top: 6px;"
            if not self.is_dark else
            "color: #4CC2FF; font-size: 14px; font-weight: bold; background: transparent; padding-top: 6px;"
        )
        layout.addWidget(lbl)
    
    def _update_device_buttons(self):
        if self.device_var == "cpu":
            self.btn_cpu.setStyleSheet(
                "background: #0078D4; color: #FFFFFF; border: none; border-radius: 6px; font-size: 14px; font-weight: bold; min-height: 38px;"
            )
            self.btn_gpu.setStyleSheet(
                "background: rgba(128,128,128,0.2); color: #888888; border: 1px solid rgba(128,128,128,0.3); border-radius: 6px; font-size: 14px; min-height: 38px;"
            )
        else:
            self.btn_gpu.setStyleSheet(
                "background: #0078D4; color: #FFFFFF; border: none; border-radius: 6px; font-size: 14px; font-weight: bold; min-height: 38px;"
            )
            self.btn_cpu.setStyleSheet(
                "background: rgba(128,128,128,0.2); color: #888888; border: 1px solid rgba(128,128,128,0.3); border-radius: 6px; font-size: 14px; min-height: 38px;"
            )
    
    def _on_device_change(self, device):
        self.device_var = device
        self._update_device_buttons()
    
    def _update_conf_label(self, v):
        self.conf_label.setText(f"{v / 100:.2f}")
    
    def _update_tol_label(self, v):
        self.tol_label.setText(f"{v / 100:.2f}")
    
    def _update_fps_label(self, v):
        self.fps_label_w.setText(f"{v:.0f}")
    
    def _apply(self):
        new_device = self.device_var
        new_conf = self.conf_slider.value() / 100.0
        new_tol = self.tol_slider.value() / 100.0
        new_fps = self.fps_slider.value()
        new_smooth = self.smooth_slider.value()
        new_res_str = self.res_combo.currentText()
        new_w, new_h = _res_to_tuple(new_res_str)

        # 检测尺寸
        det_size_map = {0: 320, 1: 480, 2: 640}
        new_det_size = det_size_map[self.det_size_combo.currentIndex()]

        # 质量过滤
        new_quality = self.qf_check.isChecked()
        minface_map = {0: 60, 1: 80, 2: 100, 3: 120}
        new_minface = minface_map[self.minface_combo.currentIndex()]

        old_device = APP_SETTINGS.get("device", "cpu")
        old_w = APP_SETTINGS.get("cam_width", 640)
        old_h = APP_SETTINGS.get("cam_height", 360)
        old_det_size = APP_SETTINGS.get("det_size", 480)

        # 更新全局设置
        APP_SETTINGS["device"] = new_device
        APP_SETTINGS["confidence"] = float(new_conf)
        APP_SETTINGS["tolerance"] = float(new_tol)
        APP_SETTINGS["proc_fps"] = new_fps
        APP_SETTINGS["track_smooth"] = new_smooth
        APP_SETTINGS["det_size"] = new_det_size
        APP_SETTINGS["cam_width"] = new_w
        APP_SETTINGS["cam_height"] = new_h
        APP_SETTINGS["quality_filter"] = new_quality
        APP_SETTINGS["min_face_size"] = new_minface

        save_settings(APP_SETTINGS)

        # 即时生效：更新检测器和识别器参数
        if self.parent_app:
            self.parent_app.detector.confidence = float(new_conf)
            self.parent_app.detector.quality_filter = new_quality
            self.parent_app.detector.min_face_size = new_minface
            self.parent_app.recognizer.tolerance = float(new_tol)

            if self.parent_app.processing:
                self.parent_app.processing._interval = 1.0 / new_fps if new_fps > 0 else 0
                if hasattr(self.parent_app.processing, 'tracker'):
                    self.parent_app.processing.tracker.smooth_frames = new_smooth

        messages = []

        if new_w != old_w or new_h != old_h:
            messages.append(f"摄像头分辨率已设为 {new_w}×{new_h}")

        if new_det_size != old_det_size:
            messages.append(f"检测模型尺寸已设为 {new_det_size}（重启后生效）")

        if new_device != old_device:
            messages.append(f"设备已切换为 {new_device.upper()}")

        if new_device != old_device or new_w != old_w or new_h != old_h:
            messages.append("请重启程序使摄像头/设备更改生效。")

        if messages:
            QMessageBox.information(self, "设置已保存", "\n".join(messages))

        self.accept()


# ═══════════════════════════════════════════════════════════════
# 运行入口（兼容旧版 main.py）
# ═══════════════════════════════════════════════════════════════

def run_app(camera, database, detector, recognizer):
    """运行 PyQt5 版 UI（替代原来 ui.FaceVisionApp）"""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
        app.setStyle("Fusion")
    
    # 设置清晰易读的字体
    font = QFont("Microsoft YaHei UI", 10)
    font.setHintingPreference(QFont.PreferFullHinting)
    font.setStyleStrategy(QFont.PreferAntialias)
    app.setFont(font)
    
    window = FaceVisionApp(camera, database, detector, recognizer)
    window.show()
    
    window._refresh_person_list()
    
    return app.exec_()