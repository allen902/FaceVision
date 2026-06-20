# CLAUDE.md

> FaceVision — 实时人脸识别系统 · AI Agent 上下文指南

---

## 项目概述

Windows 实时人脸识别桌面应用。摄像头采集 → insightface RetinaFace 检测 → ArcFace 512d 特征提取 → 余弦相似度 1:N 匹配。UI 使用 PyQt6 + Windows 11 Mica 深色玻璃态仪表盘。

**主入口:** `main.py` → 初始化 ML 组件 → 启动 `ui_pyqt6.py`

---

## 架构

```
main.py                  # 入口：依次加载 DB → Detector → Recognizer → Camera → UI
├── config.py            # 全局配置 (settings.json 读写, APP_SETTINGS 字典)
├── camera.py            # 摄像头采集线程 (独立线程, 双缓冲)
├── face_detector.py     # insightface RetinaFace 检测 + 特征提取
├── face_recognizer.py   # 余弦相似度 1:N 匹配 (编码缓存 + 版本号)
├── face_database.py     # JSON + pickle 持久化 (face_db.json, encodings.pkl)
├── face_tracker.py      # 时序追踪 (IoU 匹配 + 滑动窗口身份投票)
└── ui_pyqt6.py          # PyQt6 深色仪表盘 UI (~1700行, 当前主界面)
    ├── ProcessingThread   # 异步 ML 推理线程 (Frame → Detect → Track → Emit)
    ├── FaceVisionWindow   # 主窗口 (无边框, Mica 背景, 自定义标题栏拖动)
    ├── SettingsDialogPyQt  # 设置对话框 (可滚动, 固定底部按钮)
    ├── GlassDialog         # 对话框基类 (无边框, 拖动, 深色背景)
    ├── AddPersonDialogPyQt # 添加人员
    ├── BatchDeleteDialogPyQt # 批量删除
    └── SelectFaceDialogPyQt  # 多人脸选择
```

**数据流:**
```
CameraThread.get_frame() → ProcessingThread._run()
  → Detector.detect_with_embeddings(frame)  # 检测 + 特征
  → Tracker.update(faces, recognizer)        # IoU 追踪 + 身份投票
  → frame_ready.emit(frame, results)         # 信号 → UI 主线程
  → FaceVisionWindow._on_frame()             # cv2 绘制 → QPixmap → QLabel
```

---

## UI 状态 (PyQt6)

### 主题
- **强制深色模式** — `IS_DARK = True` (不再读取 Windows 注册表)
- 所有字体颜色: `#FFFFFF`
- 卡片背景: `#2A2A2A`
- 主背景: `#202020` + Mica 模糊效果
- Mica 通过 `apply_mica(int(self.winId()))` 在 `showEvent` 中启用

### 主窗口布局
```
┌── 自定义标题栏 (48px, 可拖动) ──────────────────────┐
│  🔷 FaceVision                    ─  ✕              │
├──────────────────────┬───────────────────────────────┤
│   中央画面 (QFrame    │  右侧面板 (QScrollArea, 300px) │
│   objectName="card")  │  ┌ 已注册人员 ──────────────┐ │
│                      │  │ 人员列表                  │ │
│   📷 视频画面         │  │ [添加] [从图片] [删除]   │ │
│                      │  └──────────────────────────┘ │
│                      │  ┌ 快速操作 ────────────────┐ │
│                      │  │ [打开完整设置]            │ │
│                      │  └──────────────────────────┘ │
│   [启动/停止摄像头]   │  ┌ 设备信息 ────────────────┐ │
│   ● 运行中    FPS     │  │ 推理设备 / 检测尺寸 / 人数│ │
│                      │  └──────────────────────────┘ │
└──────────────────────┴───────────────────────────────┘
```

### 卡片背景 → 直接 setStyleSheet
由于 PyQt6 的 QSS 级联在 QMainWindow 上不可靠，4 张卡片全部在 Python 代码中通过 `.setStyleSheet()` 直接设置：
```python
"QFrame { background-color: #2A2A2A; border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; }"
```

### QComboBox 样式 → 内联 setStyleSheet
Qt 中 QComboBox 的弹出列表是独立顶层窗口，不继承父级 QSS。
三个下拉框 (`res_dropdown`, `det_size_combo`, `minface_combo`) 各自通过
`setStyleSheet()` 内联设置，包含 `QComboBox QAbstractItemView` 规则确保
弹出列表也是深色主题。

### 已知注意事项
- **不要给 QComboBox 设置 `::drop-down` / `::down-arrow` 子控件样式** — PyQt6 下会导致下拉框变空白且无法选择
- **scroll area viewport 需显式设置背景色** — `QScrollArea > QWidget > QWidget { background: #202020; }`
- **无边框窗口 + Mica 必须同时设置** — 缺少 Mica 会导致 ghosting/double-render

---

## 组件详解

### config.py
- `DEFAULT_SETTINGS` 定义所有可配置项及其默认值
- `APP_SETTINGS = load_settings()` 模块级单例
- 键: `device`, `confidence`, `tolerance`, `cam_width`, `cam_height`, `cam_fps`, `proc_fps`, `det_size`, `track_smooth`, `min_face_size`, `quality_filter`

### face_detector.py
- 基于 `insightface.app.FaceAnalysis(providers=['...ExecutionProvider'])`
- `detect_with_embeddings(frame)` → 返回 `[(x1,y1,x2,y2,det_conf,embedding,quality_pass), ...]` 共 7 元素
- `reload_model(det_size)` 支持运行时切换检测尺寸
- `quality_filter` + `min_face_size` 过滤低质量人脸

### face_recognizer.py
- 余弦相似度 = 归一化向量点积
- `update_cache(encodings, names, version)` — 版本号避免重复重建
- `tolerance` 阈值: 越低越严格 (推荐 0.45)

### face_tracker.py
- IoU 多目标追踪 + 滑动窗口 (默认 5 帧)
- 身份确认: 最近 `smooth_frames` 帧中 ≥60% 投票一致
- `track_count` 属性: 当前活跃追踪器数量

### camera.py
- `CameraThread(camera_id, width, height, fps)`
- 独立线程循环采集，`get_frame()` 返回最新帧副本
- 线程安全的 `_lock` 保护帧缓冲

---

## 设置对话框 (SettingsDialogPyQt)

```
┌ QVBoxLayout (outer) ──────────────────────────────┐
│  ⚙ FaceVision 设置 (固定)                          │
│  ───────────── (分隔线, 固定)                       │
│  ┌ QScrollArea (可滚动) ──────────────────────────┐ │
│  │  推理设备: [CPU] [GPU (DirectML)]               │ │
│  │  摄像头分辨率: [640×360 (16:9) ▼]               │ │
│  │  检测置信度: ───●─── 0.50                       │ │
│  │  识别容差:   ───●─── 0.45                       │ │
│  │  处理帧率:   ───●─── 30                         │ │
│  │  检测模型尺寸: [640 (精准) ▼]                    │ │
│  │  追踪平滑帧数: ───●─── 5 帧                     │ │
│  │  质量过滤: [✓ 启用模糊度过滤] [最小 60px ▼]     │ │
│  └───────────────────────────────────────────────┘ │
│                    [取消]  [应用] (固定底部)         │
└────────────────────────────────────────────────────┘
```

设置项目与 `APP_SETTINGS` 键一一对应。应用时调用 `on_settings_changed()` 回调，
即时更新 detector/recognizer/tracker 参数。

---

## 运行

```bash
python main.py
```

流程:
1. 加载人脸数据库 → 打印已注册人数
2. 加载 RetinaFace 检测模型 (DirectML GPU / CPU)
3. 初始化识别器 (余弦相似度)
4. 枚举摄像头 → 启动采集线程
5. 启动 PyQt6 UI (`QApplication` + Fusion style)

---

## 依赖

```
PyQt6>=6.6.0
opencv-python>=4.8.0
insightface>=0.7.3
onnxruntime>=1.18.0
onnxruntime-directml>=1.24.0
Pillow>=10.0.0
numpy>=1.24.0
```

## 文件清单

| 文件 | 用途 |
|------|------|
| `main.py` | 入口, 组件初始化, 日志输出 |
| `config.py` | 全局配置, settings.json 读写 |
| `camera.py` | 摄像头线程 |
| `face_detector.py` | insightface 检测 + 特征提取 |
| `face_recognizer.py` | 余弦相似度匹配 + 缓存 |
| `face_database.py` | JSON + pickle 持久化 |
| `face_tracker.py` | IoU 时序追踪 + 身份投票 |
| `ui_pyqt6.py` | **主 UI** (PyQt6 深色仪表盘, ~1700 行) |
| `ui_pyqt.py` | 旧版 PyQt5 UI (保留参考, 不再使用) |
| `requirements.txt` | Python 依赖 |
| `settings.json` | 运行时配置 (自动生成) |
| `face_db.json` | 注册人员数据 (自动生成) |
| `encodings.pkl` | 特征向量 (自动生成) |
| `face_photos/` | 注册照片目录 (自动生成) |
