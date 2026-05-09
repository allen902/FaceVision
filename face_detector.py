"""
人脸检测模块
使用 insightface RetinaFace 精确检测人脸
支持 GPU (CUDA) 加速
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger("FaceVision.detector")


class FaceDetector:
    """人脸检测器 — 基于 insightface RetinaFace"""

    def __init__(self, confidence=0.45, device="cpu"):
        self.confidence = confidence
        self.device = device
        self.fp16 = device == "cuda"  # GPU 开启半精度
        self.app = None
        self._load_model()

    def _load_model(self):
        try:
            from insightface.app import FaceAnalysis

            # GPU / CPU providers
            if self.device == "cuda":
                providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
                ctx_id = 0
                logger.info("[FaceDetector] 使用 DirectML (GPU) 加速")
            else:
                providers = ['CPUExecutionProvider']
                ctx_id = -1
                logger.info("[FaceDetector] 使用 CPU")

            self.app = FaceAnalysis(
                name="buffalo_l",
                providers=providers
            )
            self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            logger.info("[FaceDetector] RetinaFace (buffalo_l) loaded")
        except ImportError:
            logger.error("insightface 未安装")
            raise
        except Exception as e:
            logger.warning(f"[FaceDetector] CUDA 加载失败, 回退 CPU: {e}")
            # 回退 CPU
            try:
                from insightface.app import FaceAnalysis
                self.app = FaceAnalysis(
                    name="buffalo_l",
                    providers=['CPUExecutionProvider']
                )
                self.app.prepare(ctx_id=-1, det_size=(640, 640))
                self.device = "cpu"
                logger.info("[FaceDetector] 回退到 CPU 模式")
            except Exception as e2:
                logger.error(f"FaceDetector 加载彻底失败: {e2}")
                raise

    def detect(self, frame):
        """
        检测画面中所有人脸
        返回: [(x1, y1, x2, y2, conf), ...]  坐标均为整数
        """
        if self.app is None or frame is None or frame.size == 0:
            return []

        try:
            faces = self.app.get(frame)
            results = []
            for face in faces:
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                conf = float(face.det_score) if hasattr(face, 'det_score') else 0.95
                if conf >= self.confidence:
                    results.append((x1, y1, x2, y2, conf))
            return results
        except Exception:
            return []

    def detect_with_embeddings(self, frame):
        """
        检测 + 特征提取一次搞定（共享推理）
        返回: [(x1, y1, x2, y2, det_conf, embedding), ...]
        """
        if self.app is None or frame is None or frame.size == 0:
            return []

        try:
            faces = self.app.get(frame)
            results = []
            for face in faces:
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                det_conf = float(face.det_score) if hasattr(face, 'det_score') else 0.95
                if det_conf >= self.confidence:
                    emb = face.normed_embedding if hasattr(face, 'normed_embedding') else None
                    results.append((x1, y1, x2, y2, det_conf, emb))
            return results
        except Exception:
            return []

    def extract_face_roi(self, frame, face_rect):
        """裁切人脸区域（带小幅扩展）"""
        x1, y1, x2, y2, _ = face_rect
        h, w = frame.shape[:2]

        dw = int((x2 - x1) * 0.20)
        dh = int((y2 - y1) * 0.20)
        ex1 = max(0, x1 - dw)
        ey1 = max(0, y1 - dh)
        ex2 = min(w, x2 + dw)
        ey2 = min(h, y2 + dh)

        if ex2 <= ex1 or ey2 <= ey1:
            return np.array([])

        return frame[ey1:ey2, ex1:ex2]