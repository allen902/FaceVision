"""
人脸识别模块
1:N 余弦相似度匹配，内置编码矩阵缓存
特征由 FaceDetector.detect_with_embeddings() 提供，避免重复推理
"""

import numpy as np
import logging

from i18n import UNKNOWN_SENTINEL

logger = logging.getLogger("FaceVision.recognizer")


class FaceRecognizer:
    """人脸识别器 — 1:N 特征匹配 + 编码矩阵缓存"""

    def __init__(self, tolerance=0.45, device=None):
        """
        tolerance: 余弦相似度阈值
                   buffalo_l 的 512 维 normed_embedding 下：
                   0.40 严格，0.45 推荐，0.50 宽松，>0.50 过于宽松
        """
        self.tolerance = tolerance
        # 编码缓存 —— 仅在数据库变更时重建
        self._cached_encodings = None   # np.ndarray (N, 512) 或 None
        self._cached_names = []         # list of str
        self._db_version = -1           # 与数据库版本号比对
        logger.info(f"[FaceRecognizer] tolerance={tolerance}")

    @property
    def cached_names(self):
        return self._cached_names

    def update_cache(self, known_encodings, known_names, db_version=0):
        """
        更新编码矩阵缓存（仅在数据库变更时调用）
        参数:
            known_encodings: list of np.ndarray
            known_names:     list of str
            db_version:      数据库版本号（变化时重建缓存）
        返回: bool — 是否实际重建了缓存
        """
        if db_version == self._db_version and self._cached_encodings is not None:
            return False  # 缓存有效，无需重建

        if len(known_encodings) == 0:
            self._cached_encodings = None
            self._cached_names = []
        else:
            # 预构建 numpy 矩阵，后续识别只需一次点积
            self._cached_encodings = np.array(known_encodings, dtype=np.float32)
            self._cached_names = list(known_names)

        self._db_version = db_version
        return True

    def recognize(self, unknown_encoding, known_encodings=None, known_names=None):
        """
        1:N 识别：对未知编码与注册库进行余弦相似度比对

        兼容旧接口（传入 known_encodings/known_names），
        但推荐使用 update_cache() 预构建缓存以加速。

        参数:
            unknown_encoding: np.ndarray (512,) — 未知人脸的编码
            known_encodings:  list of np.ndarray — 注册库编码列表（可选）
            known_names:      list of str — 注册库姓名列表（可选）

        返回: (name, confidence)  confidence 为 0.0 ~ 1.0
        """
        # 确定使用缓存还是参数
        if known_encodings is not None and len(known_encodings) > 0:
            encodings = np.array(known_encodings, dtype=np.float32)
            names = known_names if known_names else []
        elif self._cached_encodings is not None and len(self._cached_names) > 0:
            encodings = self._cached_encodings
            names = self._cached_names
        else:
            return UNKNOWN_SENTINEL, 0.0

        if unknown_encoding is None or len(names) == 0:
            return UNKNOWN_SENTINEL, 0.0

        # 确保 unknown_encoding 是 float32 且 L2 归一化
        unknown_encoding = np.asarray(unknown_encoding, dtype=np.float32).ravel()
        norm = np.linalg.norm(unknown_encoding)
        if norm > 0:
            unknown_encoding = unknown_encoding / norm

        # 余弦相似度（点积 = 余弦，因为已知编码也均已 L2 归一化）
        similarities = unknown_encoding @ encodings.T

        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])

        if best_sim < self.tolerance:
            return UNKNOWN_SENTINEL, 0.0

        return names[best_idx], best_sim
