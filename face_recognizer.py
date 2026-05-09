"""
人脸识别模块 (简化版)
仅做 1:N 余弦相似度匹配
特征由 FaceDetector.detect_with_embeddings() 提供，避免重复推理
"""

import numpy as np
import logging

logger = logging.getLogger("FaceVision.recognizer")


class FaceRecognizer:
    """人脸识别器 — 仅做 1:N 特征匹配"""

    def __init__(self, tolerance=0.30, device=None):
        """
        tolerance: 余弦相似度阈值
                   buffalo_l 的 512 维 normed_embedding 下：
                   0.25 很严格，0.30 推荐，0.40 宽松，>0.45 过于宽松
        """
        self.tolerance = tolerance
        logger.info(f"[FaceRecognizer] tolerance={tolerance}")

    def recognize(self, unknown_encoding, known_encodings, known_names):
        """
        1:N 识别：对未知编码与注册库进行余弦相似度比对

        参数:
            unknown_encoding: np.ndarray (512,) — 未知人脸的编码
            known_encodings:  list of np.ndarray — 注册库编码列表
            known_names:      list of str — 注册库姓名列表

        返回: (name, confidence)  confidence 为 0.0 ~ 1.0
        """
        if len(known_encodings) == 0 or len(known_names) == 0:
            return "未知", 0.0
        if unknown_encoding is None:
            return "未知", 0.0

        # 确保 unknown_encoding 是 float32 且 L2 归一化
        unknown_encoding = np.asarray(unknown_encoding, dtype=np.float32).ravel()
        norm = np.linalg.norm(unknown_encoding)
        if norm > 0:
            unknown_encoding = unknown_encoding / norm

        # 余弦相似度（点积 = 余弦，因为已知编码也均已 L2 归一化）
        known_array = np.array(known_encodings, dtype=np.float32)
        similarities = unknown_encoding @ known_array.T

        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])

        if best_sim < self.tolerance:
            return "未知", 0.0

        return known_names[best_idx], best_sim