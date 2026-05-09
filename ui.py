"""
Windows 11 风格用户界面模块
基于 CustomTkinter 构建简洁现代的UI
使用独立后台线程进行 ML 推理，保持 UI 流畅
"""

import os
import cv2
import uuid
import time
import threading
import numpy as np
from pathlib import Path
from PIL import Image
import customtkinter as ctk
from tkinter import messagebox, filedialog

from config import (
    APP_SETTINGS,
    ACCENT_COLOR, ACCENT_HOVER,
    DARK_BG, DARK_SURFACE, DARKER_SURFACE,
    LIGHT_BG, LIGHT_SURFACE,
    TEXT_PRIMARY, TEXT_SECONDARY,
    BORDER_COLOR, SUCCESS_COLOR, DANGER_COLOR,
)
from settings_dialog import SettingsDialog


class ProcessingThread:
    """后台 ML 处理线程 — 承担所有检测+识别+绘制工作"""

    def __init__(self, camera, detector, recognizer, database):
        self.camera = camera
        self.detector = detector
        self.recognizer = recognizer
        self.db = database
        self.running = False
        self.lock = threading.Lock()
        self.display_frame = None  # 已绘制的显示帧
        self.latest_results = []  # 最新识别结果
        self.fps = 0.0
        self._interval = 1.0 / APP_SETTINGS.get("proc_fps", 15)
        self._frame_idx = 0

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=3.0)

    def get_display_frame(self):
        with self.lock:
            if self.display_frame is not None:
                return self.display_frame.copy()
        return None

    def get_results(self):
        with self.lock:
            return list(self.latest_results)

    def get_fps(self):
        with self.lock:
            return self.fps

    def _loop(self):
        frame_count = 0
        fps_timer = time.time()

        while self.running:
            loop_start = time.time()

            # 获取摄像头帧
            frame = self.camera.get_frame()
            if frame is None:
                time.sleep(0.010)
                continue

            display = frame.copy()

            # ── 检测 + 特征提取一次搞定（共享推理） ──
            known_encodings, known_names = self.db.get_encodings_and_names()

            try:
                faces_with_emb = self.detector.detect_with_embeddings(frame)
            except Exception:
                time.sleep(0.010)
                continue

            # 调试
            self._frame_idx += 1
            if self._frame_idx % 30 == 1:
                print(f"[Processing] frame {self._frame_idx}, "
                      f"detected {len(faces_with_emb)} faces")

            results = []

            for face_item in faces_with_emb:
                x1, y1, x2, y2, det_conf, embedding = face_item

                # ── 默认蓝色 = 未知 ──
                color = (0, 120, 215)
                name = "?"

                # ── 直接使用 embedding 做 1:N 匹配，无需二次推理 ──
                if embedding is not None and len(known_encodings) > 0:
                    try:
                        name, rec_conf = self.recognizer.recognize(
                            embedding, known_encodings, known_names
                        )
                        if name != "未知":
                            color = (19, 161, 14)  # 绿色 = 已注册
                            results.append((name, rec_conf,
                                            (x1, y1, x2, y2, det_conf)))
                        else:
                            results.append(("未知", 0.0,
                                            (x1, y1, x2, y2, det_conf)))
                    except Exception:
                        results.append(("未知", 0.0,
                                        (x1, y1, x2, y2, det_conf)))
                else:
                    results.append(("未知", 0.0,
                                    (x1, y1, x2, y2, det_conf)))

                # 绘制矩形框
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

                # 绘制标签
                label = f"{name}" if name != "未知" else "?"
                if name != "未知" and color == (19, 161, 14):
                    # rec_conf 是余弦相似度，显示为百分比
                    label += f" {rec_conf:.0%}"

                (tw, th), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1
                )
                label_y1 = max(0, y1 - th - 8)
                cv2.rectangle(
                    display,
                    (x1, label_y1),
                    (x1 + tw + 8, y1),
                    color, -1
                )
                cv2.putText(
                    display, label, (x1 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1,
                    cv2.LINE_AA
                )

            # 即使未检测到人脸，也把原始帧发出去显示
            with self.lock:
                self.display_frame = display
                self.latest_results = results

            # FPS 统计
            frame_count += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                with self.lock:
                    self.fps = frame_count / (now - fps_timer)
                frame_count = 0
                fps_timer = now

            # 控制处理帧率
            elapsed = time.time() - loop_start
            sleep_time = self._interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


class FaceVisionApp(ctk.CTk):
    """主应用程序窗口"""

    def __init__(self, camera, database, detector, recognizer):
        super().__init__()

        self.camera = camera
        self.db = database
        self.detector = detector
        self.recognizer = recognizer
        self.processing = None
        self.is_running = False
        self.face_photos_dir = Path("face_photos")
        self.face_photos_dir.mkdir(exist_ok=True)

        # 窗口设置
        self.title("FaceVision — 实时人脸识别")
        self.geometry("1100x750")
        self.minsize(860, 560)

        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        self._build_ui()
        self._refresh_person_list()
        self._start_display_loop()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ─── 界面构建 ────────────────────────────────────────────
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)

        self._build_camera_panel()
        self._build_control_panel()
        self._build_status_bar()

    def _build_camera_panel(self):
        cam_frame = ctk.CTkFrame(
            self, fg_color=DARK_SURFACE,
            corner_radius=10, border_width=1,
            border_color=BORDER_COLOR
        )
        cam_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        cam_frame.grid_rowconfigure(0, weight=1)
        cam_frame.grid_columnconfigure(0, weight=1)

        self.video_label = ctk.CTkLabel(
            cam_frame, text="摄像头未启动",
            fg_color="#141414", corner_radius=8,
            font=("Segoe UI Variable", 14),
            text_color=TEXT_SECONDARY
        )
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)

    def _build_control_panel(self):
        panel = ctk.CTkFrame(
            self, fg_color=DARK_SURFACE,
            corner_radius=10, border_width=1,
            border_color=BORDER_COLOR
        )
        panel.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)
        panel.grid_columnconfigure(0, weight=1)

        r = 0

        # 标题
        ctk.CTkLabel(
            panel, text="FaceVision",
            font=("Segoe UI Variable", 26, "bold"),
            text_color=ACCENT_COLOR
        ).grid(row=r, column=0, padx=20, pady=(18, 2), sticky="w")
        r += 1

        ctk.CTkLabel(
            panel, text="实时人脸识别系统",
            font=("Segoe UI Variable", 11),
            text_color=TEXT_SECONDARY
        ).grid(row=r, column=0, padx=20, pady=(0, 14), sticky="w")
        r += 1

        # 分隔线
        ctk.CTkFrame(panel, height=1, fg_color=BORDER_COLOR).grid(
            row=r, column=0, padx=20, pady=(0, 12), sticky="ew"
        )
        r += 1

        # 摄像头按钮
        self.btn_camera = ctk.CTkButton(
            panel, text="▶  启动摄像头",
            font=("Segoe UI Variable", 13, "bold"),
            height=44, corner_radius=8,
            fg_color=SUCCESS_COLOR, hover_color="#0C6E2E",
            command=self._toggle_camera
        )
        self.btn_camera.grid(row=r, column=0, padx=20, pady=(0, 8), sticky="ew")
        r += 1

        # 添加人员
        self.btn_add = ctk.CTkButton(
            panel, text="＋  添加人员",
            font=("Segoe UI Variable", 13),
            height=38, corner_radius=8,
            fg_color=ACCENT_COLOR, hover_color=ACCENT_HOVER,
            command=self._show_add_dialog
        )
        self.btn_add.grid(row=r, column=0, padx=20, pady=(0, 6), sticky="ew")
        r += 1

        # 从文件导入
        self.btn_import = ctk.CTkButton(
            panel, text="📁  从图片导入",
            font=("Segoe UI Variable", 13),
            height=38, corner_radius=8,
            fg_color="#4A4A4A", hover_color="#3A3A3A",
            command=self._import_from_file
        )
        self.btn_import.grid(row=r, column=0, padx=20, pady=(0, 6), sticky="ew")
        r += 1

        # 删除选中
        self.btn_remove = ctk.CTkButton(
            panel, text="✕  删除选中",
            font=("Segoe UI Variable", 13),
            height=38, corner_radius=8,
            fg_color="transparent", hover_color="#CA3B13",
            border_width=1, border_color=DANGER_COLOR,
            text_color=DANGER_COLOR,
            command=self._remove_selected
        )
        self.btn_remove.grid(row=r, column=0, padx=20, pady=(0, 6), sticky="ew")
        r += 1

        # 设置
        self.btn_settings = ctk.CTkButton(
            panel, text="⚙  设置",
            font=("Segoe UI Variable", 13),
            height=38, corner_radius=8,
            fg_color="#3A3A3A", hover_color="#4A4A4A",
            command=self._show_settings
        )
        self.btn_settings.grid(row=r, column=0, padx=20, pady=(0, 12), sticky="ew")
        r += 1

        # 分隔线
        ctk.CTkFrame(panel, height=1, fg_color=BORDER_COLOR).grid(
            row=r, column=0, padx=20, pady=(0, 10), sticky="ew"
        )
        r += 1

        # 已注册人员
        ctk.CTkLabel(
            panel, text="已注册人员",
            font=("Segoe UI Variable", 12, "bold"),
            text_color=ACCENT_COLOR
        ).grid(row=r, column=0, padx=20, pady=(0, 6), sticky="w")
        r += 1

        # 人员列表
        self.person_listbox = ctk.CTkScrollableFrame(
            panel, fg_color=DARKER_SURFACE, corner_radius=8,
            border_width=1, border_color=BORDER_COLOR
        )
        self.person_listbox.grid(
            row=r, column=0, padx=20, pady=(0, 8), sticky="nsew"
        )
        panel.grid_rowconfigure(r, weight=1)
        r += 1

        # 识别结果
        self.result_label = ctk.CTkLabel(
            panel, text="等待识别…",
            font=("Segoe UI Variable", 12),
            text_color=TEXT_SECONDARY,
            wraplength=240, justify="left"
        )
        self.result_label.grid(row=r, column=0, padx=20, pady=(4, 16), sticky="ew")

    def _build_status_bar(self):
        status_frame = ctk.CTkFrame(
            self, height=34, fg_color=DARK_SURFACE,
            corner_radius=8, border_width=1,
            border_color=BORDER_COLOR
        )
        status_frame.grid(
            row=1, column=0, columnspan=2, sticky="ew",
            padx=10, pady=(0, 8)
        )

        self.status_label = ctk.CTkLabel(
            status_frame, text="● 就绪",
            font=("Segoe UI Variable", 11),
            text_color=TEXT_SECONDARY
        )
        self.status_label.pack(side="left", padx=14, pady=2)

        self.fps_label = ctk.CTkLabel(
            status_frame, text="FPS: 0",
            font=("Segoe UI Variable", 11),
            text_color=TEXT_SECONDARY
        )
        self.fps_label.pack(side="right", padx=14, pady=2)

    # ─── 显示循环（纯 UI，不做 ML） ──────────────────────────
    def _start_display_loop(self):
        """每 ~30ms 刷新一次画面（仅绘制，不推理）"""
        self._display_frame()
        self.after(30, self._start_display_loop)

    def _display_frame(self):
        """从后台线程获取已处理的显示帧并展示"""
        if self.processing and self.is_running:
            frame = self.processing.get_display_frame()
            if frame is not None:
                self._show_frame(frame)

        if self.is_running:
            fps = self.processing.get_fps() if self.processing else 0
            self.fps_label.configure(text=f"FPS: {fps:.0f}")
            results = self.processing.get_results() if self.processing else []
            if results:
                lines = []
                for name, conf, _ in results:
                    if name != "未知":
                        lines.append(f"✓ {name}  {conf:.0%}")
                    else:
                        lines.append("? 未知人员")
                self.result_label.configure(
                    text="\n".join(lines), text_color=TEXT_PRIMARY
                )
            else:
                self.result_label.configure(
                    text="正在检测…", text_color=TEXT_SECONDARY
                )

    # ─── 摄像头控制 ────────────────────────────────────────
    def _toggle_camera(self):
        if not self.is_running:
            try:
                self.camera.start()
                self.is_running = True

                self.processing = ProcessingThread(
                    self.camera, self.detector,
                    self.recognizer, self.db
                )
                self.processing.start()

                self.btn_camera.configure(
                    text="⏹  停止摄像头",
                    fg_color=DANGER_COLOR, hover_color="#9E148C"
                )
                self._set_status("● 运行中", SUCCESS_COLOR)
            except RuntimeError as e:
                messagebox.showerror("摄像头错误", str(e))
        else:
            if self.processing:
                self.processing.stop()
                self.processing = None
            self.camera.stop()
            self.is_running = False
            self.btn_camera.configure(
                text="▶  启动摄像头",
                fg_color=SUCCESS_COLOR, hover_color="#0C6E2E"
            )
            self.video_label.configure(image="", text="摄像头已停止")
            self._set_status("● 就绪", TEXT_SECONDARY)
            self.fps_label.configure(text="FPS: 0")
            self.result_label.configure(
                text="等待识别…", text_color=TEXT_SECONDARY
            )

    # ─── 添加/删除人员 ─────────────────────────────────────
    def _show_add_dialog(self):
        dialog = AddPersonDialog(self)
        self.wait_window(dialog)
        name = dialog.get_result()
        if name:
            self._capture_and_add(name)

    def _import_from_file(self):
        name = self._ask_name("输入人员姓名", "请输入要导入的人员姓名：")
        if not name:
            return
        file_path = filedialog.askopenfilename(
            title="选择包含人脸的图片",
            filetypes=[
                ("图片文件", "*.jpg *.jpeg *.png *.bmp"),
                ("所有文件", "*.*")
            ]
        )
        if not file_path:
            return
        self._add_from_image(name, file_path)

    def _capture_and_add(self, name):
        """
        多帧注册：连续采集多帧，用 detect_with_embeddings 提取特征
        如果画面中有多张人脸，弹出选择框让用户指定注册谁
        合并多帧编码（中位数），提高注册质量
        """
        if not self.is_running:
            messagebox.showwarning("无画面", "请先启动摄像头再添加人员。")
            return

        self._set_status("⏳ 正在采集人脸，请保持不动…", ACCENT_COLOR)

        # ── 第1步：获取一帧，检测所有人脸 ──
        frame = self.camera.get_frame()
        if frame is None:
            self._set_status("✗ 获取画面失败", DANGER_COLOR)
            return

        faces = self.detector.detect_with_embeddings(frame)
        if len(faces) == 0:
            self._set_status("✗ 未检测到人脸", DANGER_COLOR)
            messagebox.showwarning("未检测到", "画面中未检测到清晰人脸。")
            return

        # ── 第2步：多人同框 → 让用户选择 ──
        if len(faces) == 1:
            target_idx = 0
        else:
            # 显示选择对话框
            selector = SelectFaceDialog(self, frame, faces, name)
            self.wait_window(selector)
            target_idx = selector.get_result()
            if target_idx is None:
                self._set_status("● 已取消", TEXT_SECONDARY)
                return

        # 记录选中人脸的位置（用于后续帧跟踪锁定）
        x1, y1, x2, y2, _, _ = faces[target_idx]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        self._set_status("⏳ 正在采集多帧，请保持不动…", ACCENT_COLOR)

        encodings = []
        best_roi = None

        for i in range(15):
            frame = self.camera.get_frame()
            if frame is None:
                continue

            # 使用 detect_with_embeddings 保证检测与特征一致
            current_faces = self.detector.detect_with_embeddings(frame)
            if len(current_faces) == 0:
                continue

            # 找最接近选中人脸位置的人脸（允许小幅移动）
            min_dist = float('inf')
            matched = None
            for f in current_faces:
                fx1, fy1, fx2, fy2, _, emb = f
                f_cx = (fx1 + fx2) / 2
                f_cy = (fy1 + fy2) / 2
                dist = (f_cx - cx) ** 2 + (f_cy - cy) ** 2
                # 位置偏差不超过画面宽度的 20%
                if dist < min_dist and dist < (frame.shape[1] * 0.2) ** 2:
                    min_dist = dist
                    matched = f

            if matched is None:
                continue

            x1, y1, x2, y2, det_conf, embedding = matched

            if embedding is not None:
                encodings.append(embedding)
                roi = self.detector.extract_face_roi(
                    frame, (x1, y1, x2, y2, det_conf)
                )
                if best_roi is None or roi.size > best_roi.size:
                    best_roi = roi

        if len(encodings) < 3:
            self._set_status("✗ 采集失败，请正对摄像头", DANGER_COLOR)
            messagebox.showwarning(
                "采集失败",
                f"仅采集到 {len(encodings)}/3 帧有效人脸，请正对摄像头保持不动。"
            )
            return

        # 合并编码：取中位数（抗离群值）
        enc_stack = np.stack(encodings)
        encoding = np.median(enc_stack, axis=0)
        # 确保 L2 归一化
        encoding = encoding / (np.linalg.norm(encoding) + 1e-8)

        if best_roi is None:
            self._set_status("✗ 编码失败", DANGER_COLOR)
            return

        filename = f"{name}_{uuid.uuid4().hex[:8]}.jpg"
        img_path = self.face_photos_dir / filename
        cv2.imwrite(str(img_path), best_roi)

        success, msg = self.db.add_person(name, str(img_path), encoding)
        if success:
            self._refresh_person_list()
            self._set_status(f"✓ {msg}", SUCCESS_COLOR)
        else:
            messagebox.showwarning("添加失败", msg)
            self._set_status("● 就绪", TEXT_SECONDARY)

    def _add_from_image(self, name, file_path):
        """从图片文件提取特征并注册"""
        try:
            img = cv2.imread(file_path)
            if img is None:
                messagebox.showwarning("图片错误", "无法读取图片文件。")
                return
            faces = self.detector.detect_with_embeddings(img)
            if len(faces) == 0:
                messagebox.showwarning("未检测到人脸",
                                       "所选图片中未检测到清晰人脸。")
                return

            # 多人同框时让用户选择
            if len(faces) == 1:
                target_idx = 0
            else:
                selector = SelectFaceDialog(self, img, faces,
                                            name + "（从图片）")
                self.wait_window(selector)
                target_idx = selector.get_result()
                if target_idx is None:
                    return

            best = faces[target_idx]
            x1, y1, x2, y2, det_conf, embedding = best

            if embedding is None:
                messagebox.showwarning("编码失败", "无法提取人脸特征。")
                return

            ext = Path(file_path).suffix
            filename = f"{name}_{uuid.uuid4().hex[:8]}{ext}"
            dest_path = self.face_photos_dir / filename
            roi = self.detector.extract_face_roi(
                img, (x1, y1, x2, y2, det_conf)
            )
            cv2.imwrite(str(dest_path), roi)

            success, msg = self.db.add_person(name, str(dest_path), embedding)
            if success:
                self._refresh_person_list()
                self._set_status(f"✓ {msg}", SUCCESS_COLOR)
            else:
                messagebox.showwarning("添加失败", msg)
        except Exception as e:
            messagebox.showerror("错误", f"导入失败: {e}")

    def _show_settings(self):
        dialog = SettingsDialog(self, self.detector, self.recognizer,
                                self.camera, self.processing)
        self.wait_window(dialog)

    def _remove_selected(self):
        names = self.db.get_names()
        if not names:
            return

        dialog = BatchDeleteDialog(self, names)
        self.wait_window(dialog)
        selected = dialog.get_result()
        if selected:
            msg = f"确定要删除以下 {len(selected)} 名人员吗？\n该操作不可撤销。\n\n" + "\n".join(f"• {n}" for n in selected)
            if messagebox.askyesno("确认批量删除", msg):
                removed, not_found = self.db.remove_persons(selected)
                self._refresh_person_list()
                if not_found:
                    self._set_status(f"✓ 已删除 {len(removed)} 人，{len(not_found)} 人未找到",
                                     SUCCESS_COLOR)
                else:
                    self._set_status(f"✓ 已删除 {len(removed)} 人", SUCCESS_COLOR)

    def _ask_name(self, title, prompt):
        dialog = AddPersonDialog(self, title, prompt)
        self.wait_window(dialog)
        return dialog.get_result()

    def _refresh_person_list(self):
        for w in self.person_listbox.winfo_children():
            w.destroy()

        names = self.db.get_names()
        if not names:
            ctk.CTkLabel(
                self.person_listbox,
                text="暂无注册人员",
                font=("Segoe UI Variable", 12),
                text_color=TEXT_SECONDARY
            ).pack(pady=20)
        else:
            for name in sorted(names):
                item = ctk.CTkFrame(
                    self.person_listbox, fg_color=DARK_SURFACE,
                    corner_radius=6, height=34
                )
                item.pack(fill="x", padx=4, pady=2)

                ctk.CTkLabel(
                    item, text="👤",
                    font=("Segoe UI Variable", 14),
                    width=26
                ).pack(side="left", padx=(8, 0))

                ctk.CTkLabel(
                    item, text=name,
                    font=("Segoe UI Variable", 12),
                    text_color=TEXT_PRIMARY
                ).pack(side="left", padx=4)

    # ─── 辅助方法 ────────────────────────────────────────
    def _show_frame(self, frame):
        w = self.video_label.winfo_width()
        h = self.video_label.winfo_height()
        if w < 10 or h < 10:
            w, h = 640, 480

        fh, fw = frame.shape[:2]
        scale = min(w / fw, h / fh)

        # 对显示进行平滑缩放，保持框的锐利度
        # 缩小时用 INTER_AREA，放大时用 INTER_LINEAR
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        new_w, new_h = int(fw * scale), int(fh * scale)
        frame_resized = cv2.resize(frame, (new_w, new_h),
                                   interpolation=interp)

        rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img,
                               size=(new_w, new_h))
        self.video_label.configure(image=ctk_img, text="")
        self.video_label.image = ctk_img

    def _set_status(self, text, color=TEXT_SECONDARY):
        self.status_label.configure(text=text, text_color=color)

    def _on_close(self):
        if self.processing:
            self.processing.stop()
        if self.is_running:
            self.camera.stop()
        self.destroy()


# ─── 选择人脸对话框（多人同框时让用户指定注册谁） ──────────
class SelectFaceDialog(ctk.CTkToplevel):
    """画面中有多张人脸时，显示缩略图让用户选择要注册的人"""

    def __init__(self, parent, frame, faces, title_hint=""):
        super().__init__(parent)
        self.result = None  # 选中的索引

        self.title(f"选择要注册的人脸 {title_hint}" if title_hint else "选择要注册的人脸")
        n = len(faces)
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        w = cols * 210 + 60
        h = rows * 210 + 120
        self.geometry(f"{w}x{max(h, 200)}")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self.update_idletasks()
        px = parent.winfo_rootx() + (parent.winfo_width() - w) // 2
        py = parent.winfo_rooty() + (parent.winfo_height() - h) // 2
        self.geometry(f"+{px}+{py}")

        # 提示
        ctk.CTkLabel(
            self,
            text="画面中有多张人脸，请点击要注册的人脸：",
            font=("Segoe UI Variable", 12)
        ).pack(pady=(14, 8))

        # 人脸缩略图网格
        grid = ctk.CTkFrame(self, fg_color="transparent")
        grid.pack(padx=20, pady=(0, 10), fill="both", expand=True)

        for i, face in enumerate(faces):
            x1, y1, x2, y2, det_conf, _ = face
            # 裁剪人脸 ROI
            hf, wf = frame.shape[:2]
            dw = int((x2 - x1) * 0.15)
            dh = int((y2 - y1) * 0.15)
            ex1 = max(0, x1 - dw)
            ey1 = max(0, y1 - dh)
            ex2 = min(wf, x2 + dw)
            ey2 = min(hf, y2 + dh)

            roi = frame[ey1:ey2, ex1:ex2]
            if roi.size == 0:
                continue

            # 缩略图缩放
            thumb = cv2.resize(roi, (140, 140), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img,
                                   size=(140, 140))

            row = i // cols
            col = i % cols

            item_frame = ctk.CTkFrame(grid, fg_color=DARK_SURFACE,
                                      corner_radius=8, width=190, height=190)
            item_frame.grid(row=row, column=col, padx=6, pady=6)
            item_frame.grid_propagate(False)

            # 缩略图（可点击）
            btn = ctk.CTkButton(
                item_frame, image=ctk_img, text="",
                width=144, height=144,
                fg_color="transparent", hover_color=ACCENT_COLOR,
                corner_radius=6,
                command=lambda idx=i: self._select(idx)
            )
            btn.pack(pady=(8, 2))

            # 标签
            ctk.CTkLabel(
                item_frame,
                text=f"人脸 #{i+1}  ({det_conf:.0%})",
                font=("Segoe UI Variable", 10),
                text_color=TEXT_SECONDARY
            ).pack()

    def _select(self, idx):
        self.result = idx
        self.destroy()

    def get_result(self):
        return self.result


# ─── 对话框 ─────────────────────────────────────────────────
class AddPersonDialog(ctk.CTkToplevel):
    def __init__(self, parent, title="添加人员", prompt="请输入人员姓名："):
        super().__init__(parent)
        self.result = None

        self.title(title)
        self.geometry("380x190")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self.update_idletasks()
        px = parent.winfo_rootx() + (parent.winfo_width() - 380) // 2
        py = parent.winfo_rooty() + (parent.winfo_height() - 190) // 2
        self.geometry(f"+{px}+{py}")

        ctk.CTkLabel(
            self, text=prompt, font=("Segoe UI Variable", 13)
        ).pack(pady=(24, 8))

        self.entry = ctk.CTkEntry(
            self, width=290, height=36,
            font=("Segoe UI Variable", 13),
            corner_radius=8,
            placeholder_text="输入姓名…"
        )
        self.entry.pack(pady=(0, 8))
        self.entry.focus_set()
        self.entry.bind("<Return>", lambda e: self._confirm())

        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(pady=(6, 0))

        ctk.CTkButton(
            btn_frame, text="取消", width=90,
            fg_color="transparent", border_width=1,
            border_color=BORDER_COLOR,
            command=self.destroy
        ).pack(side="left", padx=5)
        ctk.CTkButton(
            btn_frame, text="确认", width=90,
            fg_color=ACCENT_COLOR,
            command=self._confirm
        ).pack(side="left", padx=5)

    def _confirm(self):
        name = self.entry.get().strip()
        if name:
            self.result = name
            self.destroy()
        else:
            messagebox.showwarning("输入无效", "请输入有效的姓名。")

    def get_result(self):
        return self.result


class BatchDeleteDialog(ctk.CTkToplevel):
    """批量删除人员对话框 — 支持多选（Checkbox）"""

    def __init__(self, parent, names):
        super().__init__(parent)
        self.result = []  # 返回选中的人员姓名列表

        self.title("批量删除人员")
        self.geometry("400x380")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self.update_idletasks()
        px = parent.winfo_rootx() + (parent.winfo_width() - 400) // 2
        py = parent.winfo_rooty() + (parent.winfo_height() - 380) // 2
        self.geometry(f"+{px}+{py}")

        # 提示
        ctk.CTkLabel(
            self, text="勾选要删除的人员：",
            font=("Segoe UI Variable", 13)
        ).pack(pady=(16, 6), anchor="w", padx=24)

        # 全选按钮
        select_frame = ctk.CTkFrame(self, fg_color="transparent")
        select_frame.pack(padx=24, pady=(0, 6), fill="x")

        ctk.CTkButton(
            select_frame, text="全选", width=66, height=28,
            font=("Segoe UI Variable", 11),
            fg_color="#4A4A4A", hover_color=ACCENT_COLOR,
            corner_radius=6,
            command=self._select_all
        ).pack(side="left", padx=(0, 4))

        ctk.CTkButton(
            select_frame, text="全不选", width=66, height=28,
            font=("Segoe UI Variable", 11),
            fg_color="#4A4A4A", hover_color=ACCENT_COLOR,
            corner_radius=6,
            command=self._deselect_all
        ).pack(side="left", padx=(0, 4))

        # 人员 Checkbox 列表
        scroll = ctk.CTkScrollableFrame(
            self, fg_color=DARKER_SURFACE, corner_radius=8,
            border_width=1, border_color=BORDER_COLOR
        )
        scroll.pack(padx=24, pady=(0, 8), fill="both", expand=True)

        self.check_vars = {}
        for name in sorted(names):
            var = ctk.BooleanVar()
            self.check_vars[name] = var
            ctk.CTkCheckBox(
                scroll, text=f"  {name}",
                font=("Segoe UI Variable", 12),
                variable=var,
                corner_radius=4,
                fg_color=ACCENT_COLOR,
                hover_color=ACCENT_HOVER,
                text_color=TEXT_PRIMARY
            ).pack(padx=10, pady=4, anchor="w")

        # 按钮
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(pady=(0, 14))

        ctk.CTkButton(
            btn_frame, text="取消", width=90,
            fg_color="transparent", border_width=1,
            border_color=BORDER_COLOR,
            command=self.destroy
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            btn_frame, text="删除选中", width=110,
            fg_color=DANGER_COLOR, hover_color="#CA3B13",
            command=self._confirm
        ).pack(side="left", padx=5)

    def _select_all(self):
        for var in self.check_vars.values():
            var.set(True)

    def _deselect_all(self):
        for var in self.check_vars.values():
            var.set(False)

    def _confirm(self):
        selected = [name for name, var in self.check_vars.items() if var.get()]
        if not selected:
            messagebox.showwarning("未选择", "请至少选择一名人员。")
            return
        self.result = selected
        self.destroy()

    def get_result(self):
        return self.result
