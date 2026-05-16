# FaceVision — 实时人脸识别系统

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue" alt="Python 3.9+"/>
  <img src="https://img.shields.io/badge/GUI-PyQt5_(Win11_Glass)-0078D4" alt="GUI: PyQt5"/>
  <img src="https://img.shields.io/badge/Detection-InsightFace%20RetinaFace-brightgreen" alt="Detection: RetinaFace"/>
  <img src="https://img.shields.io/badge/Recognition-Cosine%20Similarity-success" alt="Recognition: Cosine Similarity"/>
  <img src="https://img.shields.io/badge/GPU-DirectML-orange" alt="GPU: DirectML"/>
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT"/>
</p>

> 一个基于 `insightface` 和 `PyQt5` 的 Windows 实时人脸识别系统，采用 Windows 11 玻璃态（Acrylic/Mica）UI，支持 DirectML GPU 加速与自动 CPU 回退。

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
- [贡献指南](#-贡献指南)
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
- Windows 11 玻璃态仪表盘 UI（PyQt5），支持浅色/深色主题切换
- UI 与 ML 处理线程完全分离，流畅不卡顿
- 无边框对话框支持自由拖动

---

## 🛠 技术栈

| 组件 | 技术 |
|------|------|
| **编程语言** | Python 3.9+ |
| **GUI 框架** | PyQt5 (Windows 11 玻璃态 Acrylic/Mica) |
| **人脸检测** | InsightFace (RetinaFace + MobileFaceNet) |
| **特征提取** | ArcFace (512 维归一化嵌入) |
| **相似度计算** | 余弦相似度 (1:N 匹配) |
| **推理后端** | ONNX Runtime (DirectML / CPU) |
| **图像处理** | OpenCV、Pillow |
| **数字计算** | NumPy |

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
PyQt5>=5.15.9
Pillow>=10.0.0
numpy>=1.24.0
```

> 若要启用 Windows DirectML GPU，建议安装 `onnxruntime-directml`；如果只使用 CPU，可改为安装 `onnxruntime`。

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
├── LICENSE                # MIT 许可证
├── main.py                # 程序入口
├── config.py              # 配置管理
├── camera.py              # 摄像头采集线程
├── face_detector.py       # 人脸检测与特征提取
├── face_recognizer.py     # 人脸识别 (余弦相似度 1:N)
├── face_database.py       # 人脸数据持久化
├── ui.py                  # 原 Tkinter 主界面（已弃用）
├── ui_pyqt.py             # PyQt5 主界面，Windows 11 玻璃态
├── settings_dialog.py     # 设置对话框（已弃用，整合至 ui_pyqt.py）
├── requirements.txt       # Python 依赖
├── README.md              # 项目说明文档
├── settings.json          # 运行时生成的配置文件
├── face_db.json           # 运行时生成的人员数据库
├── encodings.pkl          # 运行时生成的特征向量
└── face_photos/           # 运行时生成的注册照片目录
```

---

## ❓ 常见问题

- **未检测到摄像头**：请确认摄像头已连接并允许访问。
- **GPU 按钮不可用**：说明当前环境没有 `DmlExecutionProvider`，请安装 `onnxruntime-directml` 或改用 CPU。
- **注册失败**：请让人脸正对摄像头并保持稳定，避免遮挡和快速移动。
- **识别结果为"未知"**：可适当降低 `tolerance` 或重新注册更清晰的样本。
- **ONNX Runtime 加载失败**：请确保安装的 `onnxruntime` / `onnxruntime-directml` 版本与 Python 及系统架构匹配。

---

## 🤝 贡献指南

欢迎贡献代码、提交 Issue 或改进建议！

### 贡献流程

1. Fork 本仓库
2. 创建您的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开一个 Pull Request

### 开发建议

- 代码风格请遵循 [PEP 8](https://peps.python.org/pep-0008/)。
- 提交前请确保代码可以正常运行，无语法错误。
- 若添加新功能，请在 PR 中提供相应的使用说明。

---

## 📄 许可证

本项目基于 **MIT 许可证** 开源 — 详见 [LICENSE](LICENSE) 文件。

Copyright (c) 2026 [Allen](https://github.com/allen902)

> 你可以自由地使用、修改和分发本软件，但需保留原始版权声明和许可证声明。本软件按"原样"提供，不提供任何明示或暗示的担保。