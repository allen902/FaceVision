# FaceVision — 实时人脸识别系统

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue" alt="Python 3.9+"/>
  <img src="https://img.shields.io/badge/GUI-PyQt6_(Win11_Mica)-0078D4" alt="GUI: PyQt6"/>
  <img src="https://img.shields.io/badge/Detection-InsightFace%20RetinaFace-brightgreen" alt="Detection: RetinaFace"/>
  <img src="https://img.shields.io/badge/Recognition-Cosine%20Similarity-success" alt="Recognition: Cosine Similarity"/>
  <img src="https://img.shields.io/badge/GPU-DirectML-orange" alt="GPU: DirectML"/>
  <img src="https://img.shields.io/badge/Theme-Dark%20Only-202020" alt="Theme: Dark"/>
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT"/>
</p>

> 基于 `insightface` 和 `PyQt6` 的 Windows 实时人脸识别系统，深色 Windows 11 Mica 玻璃态 UI，支持 DirectML GPU 加速。

<p align="center">
  <img src="pic/image1.png" alt="FaceVision 主界面" width="80%"/>
</p>

---

## 📋 目录

- [功能特性](#-功能特性)
- [技术栈](#-技术栈)
- [环境要求](#-环境要求)
- [安装指南](#-安装指南)
- [快速开始](#-快速开始)
- [主要功能](#-主要功能)
- [配置文件说明](#-配置文件说明)
- [项目结构](#-项目结构)
- [常见问题](#-常见问题)

---

## ✨ 功能特性

- 实时摄像头人脸检测与识别
- `insightface` `buffalo_l` 模型 (RetinaFace + ArcFace)
- 512 维归一化特征向量 + 余弦相似度 1:N 匹配
- **时序追踪** — IoU 多目标追踪 + 滑动窗口身份投票，消除闪烁
- **质量过滤** — 模糊度检测 + 最小人脸尺寸过滤
- DirectML GPU 加速，GPU 不可用时自动回退 CPU
- 多帧注册 (15 帧 + 清晰度优选 + 中位数融合)
- 本地图片导入并选择目标人脸注册
- **PyQt6 深色仪表盘 UI** — Windows 11 Mica 背景 · 纯黑风格 · 全局白色字体
- **无边框窗口** — 自定义标题栏拖动 · 最小化/关闭按钮
- **可滚动设置面板** — 10 项可调参数，即时生效
- UI 与 ML 推理线程完全分离，流畅不卡顿

---

## 🛠 技术栈

| 组件 | 技术 |
|------|------|
| **编程语言** | Python 3.9+ |
| **GUI 框架** | PyQt6 + Windows 11 Mica |
| **人脸检测** | InsightFace RetinaFace |
| **特征提取** | ArcFace (512 维嵌入) |
| **相似度计算** | 余弦相似度 (1:N) |
| **时序追踪** | IoU + 滑动窗口投票 |
| **推理后端** | ONNX Runtime (DirectML / CPU) |
| **图像处理** | OpenCV、Pillow |
| **数字计算** | NumPy |

---

## 💻 环境要求

- Windows 10 / Windows 11
- Python 3.9 或更高
- USB 或内置摄像头

---

## 📦 安装指南

```bash
git clone https://github.com/allen902/FaceVision.git
cd facevision_py
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 可选：安装 DirectML GPU 支持

```bash
pip uninstall onnxruntime onnxruntime-directml -y
pip install onnxruntime-directml==1.24.0
```

---

## 🚀 快速开始

```bash
python main.py
```

1. 点击 `启动摄像头`
2. 点击 `添加人员`，输入姓名并正对摄像头注册
3. 或点击 `从图片`，选择本地照片注册人脸
4. 已注册人员出现时，画面中显示姓名与置信度

---

## 🔧 主要功能

### 人脸检测

- `face_detector.py` 使用 `insightface.app.FaceAnalysis(name='buffalo_l')`
- 单次推理同时获得检测框 + 置信度 + 512d 特征向量
- `confidence` 阈值 + `quality_filter` 模糊过滤 + `min_face_size` 尺寸过滤

### 人脸识别

- `face_recognizer.py` 余弦相似度 1:N 匹配
- 内置编码缓存 + 版本号机制，避免重复重建

### 时序追踪

- `face_tracker.py` IoU 匹配 + 滑动窗口身份投票
- `track_smooth` 帧内多数投票一致才确认身份，避免闪烁

### 人员注册

- 实时注册：15 帧采集 → 清晰度排序 → 取前 2/3 中位数融合
- 图片导入：支持本地图片注册，多张人脸可点选目标
- 数据保存：`face_db.json` + `encodings.pkl` + `face_photos/`

### 参数设置 (10 项)

| 设置项 | 类型 | 范围 | 说明 |
|--------|------|------|------|
| 推理设备 | 按钮切换 | CPU / GPU | DirectML / CUDA 自动检测 |
| 摄像头分辨率 | 下拉框 | 8 档 | 320×240 ~ 1280×720 |
| 检测置信度 | 滑块 | 0.30 ~ 0.80 | RetinaFace 检测阈值 |
| 识别容差 | 滑块 | 0.30 ~ 0.80 | 余弦相似度阈值 |
| 处理帧率 | 滑块 | 5 ~ 60 | ML 推理帧率上限 |
| 检测模型尺寸 | 下拉框 | 320 / 480 / 640 | 越小越快 |
| 追踪平滑帧数 | 滑块 | 3 ~ 10 | 身份确认所需连续帧数 |
| 质量过滤 | 复选框 | 开/关 | 模糊度检测 |
| 最小人脸尺寸 | 下拉框 | 60 / 80 / 100 / 120 px | 小于此值过滤 |

<p align="center">
  <img src="pic/image2.png" alt="FaceVision 设置界面" width="80%"/>
</p>

---

## ⚙ 配置文件说明

程序自动生成 `settings.json`：

```json
{
  "device": "cuda",
  "confidence": 0.50,
  "tolerance": 0.45,
  "cam_width": 640,
  "cam_height": 360,
  "cam_fps": 30,
  "proc_fps": 30,
  "det_size": 640,
  "track_smooth": 5,
  "min_face_size": 60,
  "quality_filter": true
}
```

运行时生成：

- `face_db.json` — 已注册人员信息
- `encodings.pkl` — 人脸特征向量
- `face_photos/` — 注册人脸截图

---

## 📁 项目结构

```text
facevision_py/
├── main.py                # 程序入口
├── config.py              # 配置管理 (settings.json)
├── camera.py              # 摄像头采集线程
├── face_detector.py       # 人脸检测 + 特征提取
├── face_recognizer.py     # 人脸识别 (余弦相似度)
├── face_database.py       # 数据持久化 (JSON + pickle)
├── face_tracker.py        # 时序追踪 (IoU + 投票)
├── ui_pyqt6.py            # ★ PyQt6 深色 UI (当前主界面)
├── ui_pyqt.py             # PyQt5 旧版 UI (保留参考)
├── requirements.txt       # Python 依赖
├── CLAUDE.md              # AI Agent 上下文指南
├── README.md              # 项目说明
├── settings.json          # 运行时配置 (自动生成)
├── face_db.json           # 人员数据库 (自动生成)
├── encodings.pkl          # 特征向量 (自动生成)
└── face_photos/           # 注册照片 (自动生成)
```

---

## ❓ 常见问题

- **未检测到摄像头** — 确认摄像头已连接并允许访问
- **GPU 按钮不可用** — 安装 `onnxruntime-directml` 或改用 CPU
- **注册失败** — 正对摄像头保持不动，避免遮挡和快速移动
- **识别为"未知"** — 降低 `tolerance` 或重新注册更清晰的样本
- **下拉框空白** — 已修复，QComboBox 弹出列表需独立设置 QSS

---

## 📄 许可证

MIT License — 详见 [LICENSE](LICENSE) 文件。

Copyright (c) 2026 [Allen](https://github.com/allen902)
