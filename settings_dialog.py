"""
FaceVision 设置对话框
GPU/CPU 切换、检测参数、帧率调节、摄像头分辨率
"""

import customtkinter as ctk
from tkinter import messagebox
import onnxruntime

from config import APP_SETTINGS, ACCENT_COLOR, ACCENT_HOVER, TEXT_SECONDARY, BORDER_COLOR, TEXT_PRIMARY, save_settings

# 摄像头分辨率预设
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


class SettingsDialog(ctk.CTkToplevel):
    """设置对话框"""

    def __init__(self, parent, detector, recognizer, camera, processing):
        super().__init__(parent)
        self.parent = parent
        self.detector = detector
        self.recognizer = recognizer
        self.camera = camera
        self.processing = processing

        self.title("FaceVision 设置")
        self.geometry("420x520")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self.update_idletasks()
        px = parent.winfo_rootx() + (parent.winfo_width() - 420) // 2
        py = parent.winfo_rooty() + (parent.winfo_height() - 520) // 2
        self.geometry(f"+{px}+{py}")

        self._build()

    def _build(self):
        spacer = {"padx": 24, "pady": (10, 0), "anchor": "w"}

        # ── 设备选择 ──
        ctk.CTkLabel(self, text="推理设备",
                     font=("Segoe UI Variable", 14, "bold"),
                     text_color=ACCENT_COLOR).pack(**spacer)

        self.device_var = ctk.StringVar(value=APP_SETTINGS["device"])
        self.gpu_available = 'DmlExecutionProvider' in onnxruntime.get_available_providers()

        device_frame = ctk.CTkFrame(self, fg_color="transparent")
        device_frame.pack(padx=24, pady=(6, 0), fill="x")

        self.btn_cpu = ctk.CTkButton(
            device_frame, text="CPU",
            font=("Segoe UI Variable", 12),
            height=36, corner_radius=8,
            fg_color=ACCENT_COLOR if APP_SETTINGS["device"] == "cpu" else "#4A4A4A",
            command=lambda: self._on_device_change("cpu")
        )
        self.btn_cpu.pack(side="left", fill="x", expand=True, padx=(0, 4))

        self.btn_gpu = ctk.CTkButton(
            device_frame,
            text="GPU (DirectML)" if self.gpu_available else "GPU (不可用)",
            font=("Segoe UI Variable", 12),
            height=36, corner_radius=8,
            fg_color=ACCENT_COLOR if APP_SETTINGS["device"] == "cuda" and self.gpu_available else "#4A4A4A",
            command=lambda: self._on_device_change("cuda")
        )
        self.btn_gpu.pack(side="left", fill="x", expand=True, padx=(4, 0))

        if not self.gpu_available:
            self.btn_gpu.configure(state="disabled")

        # ── 检测置信度 ──
        ctk.CTkLabel(self, text="检测置信度",
                     font=("Segoe UI Variable", 14, "bold"),
                     text_color=ACCENT_COLOR).pack(padx=24, pady=(18, 0), anchor="w")

        conf_frame = ctk.CTkFrame(self, fg_color="transparent")
        conf_frame.pack(padx=24, pady=(6, 0), fill="x")

        self.conf_slider = ctk.CTkSlider(
            conf_frame, from_=0.15, to=0.70, number_of_steps=11,
            width=260, command=self._update_conf_label
        )
        self.conf_slider.pack(side="left")
        self.conf_slider.set(APP_SETTINGS["confidence"])

        self.conf_label = ctk.CTkLabel(
            conf_frame, text=f"{APP_SETTINGS['confidence']:.2f}",
            font=("Segoe UI Variable", 13), width=50
        )
        self.conf_label.pack(side="right")

        # ── 识别容差 ──
        ctk.CTkLabel(self, text="识别容差",
                     font=("Segoe UI Variable", 14, "bold"),
                     text_color=ACCENT_COLOR).pack(padx=24, pady=(18, 0), anchor="w")

        tol_frame = ctk.CTkFrame(self, fg_color="transparent")
        tol_frame.pack(padx=24, pady=(6, 0), fill="x")

        self.tol_slider = ctk.CTkSlider(
            tol_frame, from_=0.30, to=0.80, number_of_steps=10,
            width=260, command=self._update_tol_label
        )
        self.tol_slider.pack(side="left")
        self.tol_slider.set(APP_SETTINGS["tolerance"])

        self.tol_label = ctk.CTkLabel(
            tol_frame, text=f"{APP_SETTINGS['tolerance']:.2f}",
            font=("Segoe UI Variable", 13), width=50
        )
        self.tol_label.pack(side="right")

        # ── 处理帧率 ──
        ctk.CTkLabel(self, text="处理帧率 (FPS)",
                     font=("Segoe UI Variable", 14, "bold"),
                     text_color=ACCENT_COLOR).pack(padx=24, pady=(18, 0), anchor="w")

        fps_frame = ctk.CTkFrame(self, fg_color="transparent")
        fps_frame.pack(padx=24, pady=(6, 0), fill="x")

        self.fps_slider = ctk.CTkSlider(
            fps_frame, from_=5, to=25, number_of_steps=20,
            width=260, command=self._update_fps_label
        )
        self.fps_slider.pack(side="left")
        self.fps_slider.set(APP_SETTINGS["proc_fps"])

        self.fps_label_w = ctk.CTkLabel(
            fps_frame, text=f"{APP_SETTINGS['proc_fps']:.0f}",
            font=("Segoe UI Variable", 13), width=50
        )
        self.fps_label_w.pack(side="right")

        # ── 摄像头分辨率 ──
        ctk.CTkLabel(self, text="摄像头分辨率",
                     font=("Segoe UI Variable", 14, "bold"),
                     text_color=ACCENT_COLOR).pack(padx=24, pady=(18, 0), anchor="w")

        res_frame = ctk.CTkFrame(self, fg_color="transparent")
        res_frame.pack(padx=24, pady=(6, 0), fill="x")

        self.res_var = ctk.StringVar(value=_current_resolution_key())
        self.res_dropdown = ctk.CTkOptionMenu(
            res_frame, variable=self.res_var,
            values=RESOLUTIONS,
            font=("Segoe UI Variable", 12),
            dropdown_font=("Segoe UI Variable", 12),
            height=34, corner_radius=8,
            fg_color="#3A3A3A", button_color=ACCENT_COLOR,
            button_hover_color=ACCENT_HOVER,
            dropdown_fg_color="#2D2D2D",
            dropdown_hover_color=ACCENT_COLOR,
            dropdown_text_color=TEXT_PRIMARY,
            text_color=TEXT_PRIMARY,
        )
        self.res_dropdown.pack(fill="x")

        # ── 提示 ──
        ctk.CTkLabel(
            self, text="更改将在下次启动摄像头时生效",
            font=("Segoe UI Variable", 10),
            text_color=TEXT_SECONDARY
        ).pack(padx=24, pady=(18, 0), anchor="w")

        # ── 按钮 ──
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(pady=(14, 0))

        ctk.CTkButton(
            btn_frame, text="应用", width=110,
            fg_color=ACCENT_COLOR,
            command=self._apply
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            btn_frame, text="关闭", width=110,
            fg_color="transparent", border_width=1,
            border_color=BORDER_COLOR,
            command=self.destroy
        ).pack(side="left", padx=5)

    def _update_conf_label(self, v):
        self.conf_label.configure(text=f"{float(v):.2f}")

    def _update_tol_label(self, v):
        self.tol_label.configure(text=f"{float(v):.2f}")

    def _update_fps_label(self, v):
        self.fps_label_w.configure(text=f"{int(float(v))}")

    def _on_device_change(self, device):
        self.device_var.set(device)
        if device == "cpu":
            self.btn_cpu.configure(fg_color=ACCENT_COLOR)
            self.btn_gpu.configure(fg_color="#4A4A4A")
        else:
            self.btn_cpu.configure(fg_color="#4A4A4A")
            self.btn_gpu.configure(fg_color=ACCENT_COLOR)

    def _apply(self):
        new_device = self.device_var.get()
        new_conf = self.conf_slider.get()
        new_tol = self.tol_slider.get()
        new_fps = int(self.fps_slider.get())
        new_res_str = self.res_var.get()
        new_w, new_h = _res_to_tuple(new_res_str)

        old_device = APP_SETTINGS.get("device", "cpu")
        old_w = APP_SETTINGS.get("cam_width", 640)
        old_h = APP_SETTINGS.get("cam_height", 360)

        APP_SETTINGS["device"] = new_device
        APP_SETTINGS["confidence"] = float(new_conf)
        APP_SETTINGS["tolerance"] = float(new_tol)
        APP_SETTINGS["proc_fps"] = new_fps
        APP_SETTINGS["cam_width"] = new_w
        APP_SETTINGS["cam_height"] = new_h

        save_settings(APP_SETTINGS)

        self.detector.confidence = float(new_conf)
        self.recognizer.tolerance = float(new_tol)

        if self.processing:
            self.processing._interval = 1.0 / max(1, new_fps)

        # 分辨率变化 → 通知用户重启
        if new_w != old_w or new_h != old_h:
            messagebox.showinfo(
                "分辨率更改",
                f"摄像头分辨率已设为 {new_w}×{new_h}。\n"
                "请重启程序使新分辨率生效。"
            )

        if new_device != old_device:
            if new_device == "cuda":
                print(f"[Settings] >>> 切换为 GPU 模式 (DirectML) <<<")
            else:
                print(f"[Settings] 切换为 CPU 模式")
            messagebox.showinfo(
                "设备切换",
                f"设备已设为 {new_device.upper()}。\n"
                "请重启程序使 GPU/CPU 切换生效。"
            )

        self.destroy()