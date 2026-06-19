"""
人脸时序追踪模块
基于 IoU 的轻量级多目标追踪 + 身份投票平滑

工作原理：
1. 每帧检测到的人脸通过 IoU 与现有追踪器匹配
2. 每个追踪器维护最近 N 帧的识别结果
3. 身份确认 = 最近 smooth_frames 帧中多数投票一致且超过阈值
4. 避免单帧误检导致的身份闪烁
"""

import numpy as np
import logging

logger = logging.getLogger("FaceVision.tracker")


def _iou(boxA, boxB):
    """计算两个边界框的 IoU (Intersection over Union)"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(areaA + areaB - interArea)


class FaceTrack:
    """单个人脸的追踪状态"""

    __slots__ = ('id', 'bbox', 'name_history', 'conf_history',
                 'quality_history', 'frames_since_update', 'total_frames',
                 'confirmed_name', 'confirmed_conf', 'latest_name', 'latest_conf')

    def __init__(self, track_id, bbox):
        self.id = track_id
        self.bbox = bbox                  # 当前边界框 (x1, y1, x2, y2)
        self.name_history = []            # 最近 N 帧识别名（最多 smooth_frames*3 条）
        self.conf_history = []            # 对应的置信度
        self.quality_history = []         # 对应帧的质量标记
        self.frames_since_update = 0      # 未匹配的连续帧数
        self.total_frames = 1             # 总追踪帧数
        self.confirmed_name = "未知"       # 确认后的身份
        self.confirmed_conf = 0.0         # 确认后的置信度
        self.latest_name = "未知"          # 最新一帧的识别结果（用于显示）
        self.latest_conf = 0.0            # 最新一帧的置信度

    def update(self, bbox, name, conf, quality_pass=True):
        """用新检测更新追踪器"""
        self.bbox = bbox
        self.frames_since_update = 0
        self.total_frames += 1
        self.name_history.append(name)
        self.conf_history.append(conf)
        self.quality_history.append(quality_pass)
        self.latest_name = name
        self.latest_conf = conf

    def mark_missed(self):
        """标记本帧未匹配"""
        self.frames_since_update += 1

    def is_stale(self, max_missed=10):
        """追踪是否已过期"""
        return self.frames_since_update >= max_missed

    def _majority_vote(self, smooth_frames):
        """
        在最近 smooth_frames 帧中做多数投票
        使用自适应阈值：帧数少时降低要求
        返回: (name, avg_confidence, is_confirmed)
        """
        if not self.name_history:
            return self.latest_name, self.latest_conf, False

        # 取最近 smooth_frames 条（优先质量通过的帧）
        recent_qualified = [
            (n, c) for n, c, q in zip(
                self.name_history[-smooth_frames:],
                self.conf_history[-smooth_frames:],
                self.quality_history[-smooth_frames:]
            )
            if q  # 仅统计质量通过的帧
        ]

        # 如果没有质量通过的帧，用全部最近帧
        recent = recent_qualified if recent_qualified else list(zip(
            self.name_history[-smooth_frames:],
            self.conf_history[-smooth_frames:]
        ))

        if not recent:
            return self.latest_name, self.latest_conf, False

        # 统计每个名字的票数
        votes = {}
        conf_sums = {}
        for name, conf in recent:
            votes[name] = votes.get(name, 0) + 1
            conf_sums[name] = conf_sums.get(name, 0.0) + conf

        # 得票最多的名字
        best_name = max(votes, key=votes.get)
        best_votes = votes[best_name]
        avg_conf = conf_sums[best_name] / best_votes if best_votes > 0 else 0.0

        # 自适应阈值：根据实际帧数调整
        total = len(recent)
        if total >= 6:
            min_votes = total // 2 + 1          # >50%
        elif total >= 3:
            min_votes = total // 2 + 1          # >50%，3帧需2票
        else:
            min_votes = total                   # 1-2帧：1票即可（初步判断）

        is_confirmed = (best_votes >= min_votes
                        and best_name != "未知"
                        and avg_conf >= 0.30)    # 额外：相似度必须 ≥ 0.30

        return best_name, avg_conf, is_confirmed

    def resolve_identity(self, smooth_frames=5):
        """
        解析当前帧的最终身份
        返回: (display_name, confidence, is_confirmed)
          display_name: 始终返回最新识别名（未确认也显示姓名+?）
        """
        name, conf, is_confirmed = self._majority_vote(smooth_frames)
        if is_confirmed:
            self.confirmed_name = name
            self.confirmed_conf = conf
            return name, conf, True
        else:
            # 未确认但已有识别结果：显示最新识别名
            if self.latest_name != "未知":
                return self.latest_name, self.latest_conf, False
            return self.latest_name, 0.0, False


class FaceTracker:
    """
    多目标人脸追踪器
    - 基于 IoU 匹配检测 ↔ 追踪
    - 身份平滑投票
    - 自动创建/销毁追踪
    """

    def __init__(self, smooth_frames=5, iou_threshold=0.30, max_missed=10):
        self.smooth_frames = smooth_frames
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.tracks = []           # list of FaceTrack
        self._next_id = 0

    def update(self, detections, recognizer=None, known_encodings=None, known_names=None):
        """
        用当前帧检测结果更新追踪器

        参数:
            detections: list of (x1, y1, x2, y2, det_conf, embedding, quality_pass)
            recognizer: FaceRecognizer 实例（可选，用于在追踪器内做识别）
            known_encodings, known_names: 用于识别（推荐使用 recognizer 缓存，传 None）

        返回:
            results: list of dict [...]
        """
        # ── 1. 对每个检测做识别（始终尝试识别，质量标记仅影响投票） ──
        recognized = []
        for det in detections:
            x1, y1, x2, y2, det_conf, embedding, quality_pass = det
            name, rec_conf = "未知", 0.0
            if embedding is not None and recognizer is not None:
                try:
                    # 优先使用缓存（传 None 触发缓存路径）
                    name, rec_conf = recognizer.recognize(
                        embedding, None, None
                    )
                except Exception:
                    pass
            recognized.append({
                'bbox': (x1, y1, x2, y2),
                'name': name,
                'conf': rec_conf,
                'det_conf': det_conf,
                'embedding': embedding,
                'quality_pass': quality_pass,
            })

        # ── 2. IoU 匹配检测到现有追踪 ──
        matched_track_ids = set()
        matched_det_ids = set()

        if self.tracks and recognized:
            # 构建 IoU 矩阵
            iou_matrix = np.zeros((len(self.tracks), len(recognized)))
            for ti, track in enumerate(self.tracks):
                for di, det in enumerate(recognized):
                    iou_matrix[ti, di] = _iou(track.bbox, det['bbox'])

            # 贪心匹配：从最高 IoU 开始
            while True:
                if iou_matrix.size == 0:
                    break
                max_iou = np.max(iou_matrix)
                if max_iou < self.iou_threshold:
                    break
                ti, di = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                # 匹配
                det = recognized[di]
                self.tracks[ti].update(
                    det['bbox'], det['name'], det['conf'], det['quality_pass']
                )
                matched_track_ids.add(ti)
                matched_det_ids.add(di)
                # 将该行/列置零
                iou_matrix[ti, :] = 0
                iou_matrix[:, di] = 0

        # ── 3. 未匹配的追踪器标记为 missed ──
        for ti, track in enumerate(self.tracks):
            if ti not in matched_track_ids:
                track.mark_missed()

        # ── 4. 未匹配的检测创建新追踪 ──
        for di, det in enumerate(recognized):
            if di not in matched_det_ids:
                new_track = FaceTrack(self._next_id, det['bbox'])
                self._next_id += 1
                new_track.update(det['bbox'], det['name'], det['conf'], det['quality_pass'])
                self.tracks.append(new_track)

        # ── 5. 清理过期追踪 ──
        self.tracks = [t for t in self.tracks if not t.is_stale(self.max_missed)]

        # ── 6. 限制历史长度 ──
        max_history = self.smooth_frames * 3
        for t in self.tracks:
            if len(t.name_history) > max_history:
                t.name_history = t.name_history[-self.smooth_frames:]
                t.conf_history = t.conf_history[-self.smooth_frames:]
                t.quality_history = t.quality_history[-self.smooth_frames:]

        # ── 7. 解析身份并构建输出 ──
        results = []
        for track in self.tracks:
            display_name, display_conf, is_confirmed = track.resolve_identity(
                self.smooth_frames
            )
            results.append({
                'bbox': track.bbox,
                'name': display_name,
                'conf': display_conf,
                'det_conf': 0.0,
                'track_id': track.id,
                'is_confirmed': is_confirmed,
                'quality_pass': True,
            })

        return results

    def reset(self):
        """重置所有追踪"""
        self.tracks.clear()
        self._next_id = 0

    @property
    def track_count(self):
        return len(self.tracks)
