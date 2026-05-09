# FaceVision — 实时人脸识别系统

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue" alt="Python 3.9+"/>
  <img src="https://img.shields.io/badge/GUI-CustomTkinter-0078D4" alt="GUI: CustomTkinter"/>
  <img src="https://img.shields.io/badge/Detection-InsightFace%20RetinaFace-brightgreen" alt="Detection: RetinaFace"/>
  <img src="https://img.shields.io/badge/Recognition-Cosine%20Similarity-success" alt="Recognition: Cosine Similarity"/>
  <img src="https://img.shields.io/badge/GPU-DirectML-orange" alt="GPU: DirectML"/>
</p>

> 一个基于 `insightface` 和 `CustomTkinter` 的 Windows 实时人脸识别系统，支持 DirectML GPU 加速与自动 CPU 回退。

---

## 📋 目录

- [功能特性](#-功能特性)
- [环境要求](#-环境要求)
- [安装指南](#-安装指南)
- [快速开始](#-快速开始)
- [主要功能](#-主要功能)
- [配置文件说明](#-配置文件说明)
- [项目结构](#-项目结构)
- [常见问题](#-常见问题)
- [许可证](#-许可证)

---

## ✨ 功能特性

- 实时摄像头人脸检测与识别
- 基于 `insightface` 的 `buffalo_l` 模型
- 512 维归一化特征向量 + 余弦相似度 1:N 匹配
- 支持 DirectML GPU 加速，GPU 不可用时自动回退 CPU
- 实时参数调节：检测置信度、识别容差、处理帧率、分辨率
- 多帧注册：15 帧特征融合提高注册稳定性
- 支持本地图片导入并选择目标人脸注册
- 界面基于 CustomTkinter，UI 与 ML 处理线程分离

---

## 💻 环境要求

- Windows 10 / Windows 11
- Python 3.9 或更高
- USB 或内置摄像头

推荐依赖：

```txt
opencv-python>=4.8.0
insightface>=0.7.3
onnxruntime-directml>=1.21.0
customtkinter>=5.2.0
Pillow>=10.0.0
numpy>=1.24.0
```

> 若要启用 Windows DirectML GPU，建议安装 `onnxruntime-directml`；如果只使用 CPU，可改为安装 `onnxruntime`。

---

## 📦 安装指南

```bash
git clone https://github.com/allen902/FaceVision-.git
cd facevision_py
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 可选：安装 DirectML GPU 支持

```bash
pip uninstall onnxruntime onnxruntime-directml -y
pip install onnxruntime-directml==1.21.0
```

---

## 🚀 快速开始

```bash
python main.py
```

1. 点击 `启动摄像头`。
2. 点击 `添加人员`，输入姓名并正对摄像头进行注册。
3. 或点击 `从图片导入`，选择本地照片注册人脸。
4. 已注册人员出现时，画面中会显示姓名与识别置信度。

---

## 🔧 主要功能

### 人脸检测

- 通过 `face_detector.py` 使用 `insightface.app.FaceAnalysis(name='buffalo_l')`。
- 单次推理同时获得检测框、置信度和特征向量。
- 通过 `confidence` 阈值过滤低质量检测结果。

### 人脸识别

- `face_recognizer.py` 使用余弦相似度计算 1:N 匹配。
- 已注册人脸编码与当前检测人脸编码点积比较，超过 `tolerance` 即判定为相同人。

### 人员注册

- 实时注册：连续采集 15 帧人脸特征，取中位数融合，减少异常值影响。
- 图片导入：支持本地图片注册，若图片中存在多张人脸，可选择目标人脸。
- 注册数据保存在 `face_db.json`、`encodings.pkl` 和 `face_photos/`。

### 摄像头管理

- `camera.py` 负责独立线程采集画面并缓存最新帧。
- 支持指定分辨率与目标帧率。
- 同时兼顾采集稳定性与低延迟显示。

### 参数设置

- GPU / CPU 推理设备切换
- 检测置信度调节
- 识别容差调节
- ML 处理帧率调节
- 摄像头分辨率选择

---

## ⚙ 配置文件说明

程序会自动生成并读取 `settings.json`：

```json
{
  "device": "cuda",
  "confidence": 0.25,
  "tolerance": 0.30,
  "cam_width": 640,
  "cam_height": 360,
  "cam_fps": 30,
  "proc_fps": 12
}
```

运行时还会生成：

- `face_db.json`：已注册人员信息
- `encodings.pkl`：人脸特征向量
- `face_photos/`：注册人脸截图

---

## 📁 项目结构

```text
facevision_py/
├── .gitignore
├── main.py
├── config.py
├── camera.py
├── face_detector.py
├── face_recognizer.py
├── face_database.py
├── ui.py
├── settings_dialog.py
├── requirements.txt
├── README.md
├── settings.json       # 运行时生成
├── face_db.json        # 运行时生成
├── encodings.pkl       # 运行时生成
└── face_photos/        # 运行时生成
```

---

## ❓ 常见问题

- **未检测到摄像头**：请确认摄像头已连接并允许访问。
- **GPU 按钮不可用**：说明当前环境没有 `DmlExecutionProvider`，请安装 `onnxruntime-directml` 或改用 CPU。
- **注册失败**：请让人脸正对摄像头并保持稳定，避免遮挡和快速移动。
- **识别结果为“未知”**：可适当降低 `tolerance` 或重新注册更清晰的样本。

---

## 📄 许可证

本项目当前未指定许可证，如需发布请补充 `LICENSE` 文件。