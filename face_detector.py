"""
人脸检测模块
使用 insightface RetinaFace 精确检测人脸
支持 GPU (DirectML) 加速，推理失败时自动回退 CPU
内置人脸质量评估（模糊度检测）+ 模型预热
"""

import cv2
import numpy as np
import logging
import threading
import sys

logger = logging.getLogger("FaceVision.detector")


class FaceDetector:
    """人脸检测器 — 基于 insightface RetinaFace"""

    def __init__(self, confidence=0.50, device="cpu", det_size=640,
                 quality_filter=True, min_face_size=80):
        self.confidence = confidence
        self.device = device
        self.det_size = det_size
        self.quality_filter = quality_filter
        self.min_face_size = min_face_size
        self.app = None
        self._lock = threading.Lock()
        self._inference_error_count = 0  # 推理失败计数
        self._gpu_name = "CPU"           # 实际使用的设备名
        self._load_model()
        self._warmup()

    def _resolve_providers(self, force_cpu=False):
        """解析 ONNX Runtime 执行提供器，返回 (providers, ctx_id, gpu_name)"""
        from insightface.app import FaceAnalysis
        import onnxruntime as ort

        available = ort.get_available_providers()

        if self.device == "cuda" and not force_cpu:
            if 'CUDAExecutionProvider' in available:
                return ['CUDAExecutionProvider', 'CPUExecutionProvider'], 0, "GPU(CUDA)"
            elif 'DmlExecutionProvider' in available:
                return ['DmlExecutionProvider', 'CPUExecutionProvider'], 0, "GPU(DirectML)"
            else:
                logger.info("[FaceDetector] 无可用 GPU，使用 CPU")
                return ['CPUExecutionProvider'], -1, "CPU (无可用GPU)"
        else:
            if force_cpu:
                print("[FaceDetector] [WARN] Inference falling back to CPU mode")
            logger.info("[FaceDetector] 使用 CPU")
            return ['CPUExecutionProvider'], -1, "CPU"

    def _create_app(self, det_size):
        """创建并返回一个已 prepare 的 FaceAnalysis 实例"""
        from insightface.app import FaceAnalysis

        providers, ctx_id, gpu_name = self._resolve_providers()
        app = FaceAnalysis(name="buffalo_l", providers=providers)
        app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
        return app, gpu_name

    def _load_model(self, fallback_to_cpu=False):
        """加载模型。自动检测 CUDA > DirectML > CPU"""
        try:
            self.app, gpu_name = self._create_app(self.det_size)
            print(f"[FaceDetector] [OK] Model loaded (device={gpu_name}, det_size={self.det_size})")
        except ImportError:
            logger.error("insightface 未安装")
            raise
        except Exception as e:
            if not fallback_to_cpu and self.device == "cuda":
                logger.warning(f"[FaceDetector] GPU 加载失败, 回退 CPU: {e}")
                print(f"[FaceDetector] [WARN] GPU load failed: {e}")
                print("[FaceDetector] → 回退到 CPU…")
                self.device = "cpu"
                self._load_model(fallback_to_cpu=True)
            else:
                logger.error(f"FaceDetector 加载彻底失败: {e}")
                raise

    def _warmup(self):
        """模型预热：跑一次 dummy 推理，同时检测推理是否正常工作"""
        try:
            dummy = np.random.randint(0, 255, (self.det_size, self.det_size, 3), dtype=np.uint8)
            with self._lock:
                faces = self.app.get(dummy)
            print(f"[FaceDetector] [OK] Warm-up complete (detected {len(faces)} fake faces)")
        except Exception as e:
            err_msg = str(e)
            print(f"[FaceDetector] [WARN] Warm-up failed (size={self.det_size}): {err_msg}")

            # DirectML 1.24.x 在特定输入尺寸（如 480）上 Reshape 节点有 bug，
            # 尝试用全新 app 实例 + 640 尺寸重试（已验证 640 可用）
            if self.device == "cuda" and "UnicodeDecodeError" in type(e).__name__:
                try:
                    print(f"[FaceDetector] → DirectML 不兼容 det_size={self.det_size}，重建为 640…")
                    new_app, _ = self._create_app(640)
                    dummy2 = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                    new_app.get(dummy2)
                    self.app = new_app
                    self.det_size = 640
                    print("[FaceDetector] [OK] Auto-adjusted to det_size=640 (DirectML compat)")
                    return
                except Exception as inner:
                    print(f"[FaceDetector] → 640 重试也失败: {inner}")
                print("[FaceDetector] → DirectML 推理失败，回退到 CPU…")
                self.device = "cpu"
                self._load_model(fallback_to_cpu=True)
            elif self.device == "cuda":
                print("[FaceDetector] → 推理异常，尝试切换到 CPU…")
                self.device = "cpu"
                self._load_model(fallback_to_cpu=True)

    # ── 人脸质量评估 ────────────────────────────────────────────

    @staticmethod
    def _face_quality(face_roi):
        """
        评估人脸区域质量
        返回: (is_good: bool, blur_score: float)
          blur_score: Laplacian 方差，越高越清晰。>100=清晰, >50=可用, <50=模糊
        """
        if face_roi is None or face_roi.size == 0 or face_roi.shape[0] < 20 or face_roi.shape[1] < 20:
            return False, 0.0

        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_good = blur_score >= 50.0  # 清晰度阈值
        return is_good, blur_score

    @staticmethod
    def _face_roi(frame, bbox, expand=0.20):
        """裁切人脸区域（带小幅扩展），返回 ROI 数组"""
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        dw = int((x2 - x1) * expand)
        dh = int((y2 - y1) * expand)
        ex1 = max(0, x1 - dw)
        ey1 = max(0, y1 - dh)
        ex2 = min(w, x2 + dw)
        ey2 = min(h, y2 + dh)

        if ex2 <= ex1 or ey2 <= ey1:
            return np.array([])

        return frame[ey1:ey2, ex1:ex2]

    # ── 核心检测接口 ────────────────────────────────────────────

    def _run_inference(self, frame):
        """
        执行推理，带自动回退。
        如果 DirectML 推理连续失败，自动切换到 CPU。
        """
        with self._lock:
            faces = self.app.get(frame)
        return faces

    def _handle_inference_error(self, e, frame):
        """处理推理错误，必要时回退 CPU"""
        self._inference_error_count += 1
        err_type = type(e).__name__

        # GPU 推理错误，回退 CPU
        if self.device == "cuda":
            print(f"[FaceDetector] [WARN] GPU inference error ({err_type}: {e}), switching to CPU...")
            try:
                self.device = "cpu"
                self._load_model(fallback_to_cpu=True)
                with self._lock:
                    return self.app.get(frame)
            except Exception as e2:
                print(f"[FaceDetector] [ERROR] CPU fallback also failed: {e2}")
                raise
        raise

    def detect(self, frame):
        """
        检测画面中所有人脸
        返回: [(x1, y1, x2, y2, conf), ...]  坐标均为整数
        """
        if self.app is None or frame is None or frame.size == 0:
            return []

        try:
            faces = self._run_inference(frame)
        except Exception as e:
            try:
                faces = self._handle_inference_error(e, frame)
            except Exception:
                return []

        results = []
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            conf = float(face.det_score) if hasattr(face, 'det_score') else 0.95
            if conf >= self.confidence:
                results.append((x1, y1, x2, y2, conf))

        if len(faces) > 0 and len(results) == 0:
            raw_confs = [float(f.det_score) if hasattr(f, 'det_score') else 0.95 for f in faces]
            print(f"[FaceDetector] detect: {len(faces)} 人脸, "
                  f"全部低于阈值 {self.confidence:.2f} "
                  f"(置信度: {[f'{c:.2f}' for c in raw_confs]})")
        return results

    def detect_with_embeddings(self, frame):
        """
        检测 + 特征提取一次搞定（共享推理）
        返回: [(x1, y1, x2, y2, det_conf, embedding, quality_pass), ...]
        """
        if self.app is None or frame is None or frame.size == 0:
            return []

        try:
            faces = self._run_inference(frame)
        except Exception as e:
            try:
                faces = self._handle_inference_error(e, frame)
            except Exception:
                return []

        results = []
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            det_conf = float(face.det_score) if hasattr(face, 'det_score') else 0.95

            if det_conf < self.confidence:
                continue

            emb = face.normed_embedding if hasattr(face, 'normed_embedding') else None

            # 质量过滤
            quality_pass = True
            if self.quality_filter:
                face_w = x2 - x1
                face_h = y2 - y1
                if face_w < self.min_face_size or face_h < self.min_face_size:
                    quality_pass = False
                else:
                    roi = self._face_roi(frame, (x1, y1, x2, y2), expand=0.10)
                    is_clear, _blur = self._face_quality(roi)
                    if not is_clear:
                        quality_pass = False

            results.append((x1, y1, x2, y2, det_conf, emb, quality_pass))

        if len(faces) > 0 and len(results) == 0:
            raw_confs = [float(f.det_score) if hasattr(f, 'det_score') else 0.95 for f in faces]
            print(f"[FaceDetector] detect_with_emb: {len(faces)} 人脸, "
                  f"全部低于阈值 {self.confidence:.2f} "
                  f"(置信度: {[f'{c:.2f}' for c in raw_confs]})")
        return results

    def extract_face_roi(self, frame, face_rect):
        """裁切人脸区域（带小幅扩展），兼容旧接口"""
        x1, y1, x2, y2, _ = face_rect
        return self._face_roi(frame, (x1, y1, x2, y2), expand=0.20)

    def reload_model(self, det_size=None):
        """
        用新的参数重新加载模型（用于运行时切换检测尺寸）
        """
        if det_size is not None:
            self.det_size = det_size
        self._load_model()
        self._warmup()