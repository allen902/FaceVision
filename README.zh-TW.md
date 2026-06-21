# FaceVision — 即時人臉辨識系統

<p align="center">
  <a href="README.md">English</a> | <a href="README.zh-CN.md">简体中文</a> | <strong>繁體中文</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue" alt="Python 3.9+"/>
  <img src="https://img.shields.io/badge/GUI-PyQt6_(Win11_Mica)-0078D4" alt="GUI: PyQt6"/>
  <img src="https://img.shields.io/badge/Detection-InsightFace%20RetinaFace-brightgreen" alt="Detection: RetinaFace"/>
  <img src="https://img.shields.io/badge/Recognition-Cosine%20Similarity-success" alt="Recognition: Cosine Similarity"/>
  <img src="https://img.shields.io/badge/GPU-DirectML-orange" alt="GPU: DirectML"/>
  <img src="https://img.shields.io/badge/Theme-Dark%20Only-202020" alt="Theme: Dark"/>
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT"/>
</p>

> 基於 `insightface` 與 `PyQt6` 的 Windows 即時人臉辨識系統，深色 Windows 11 Mica 玻璃態 UI，支援 DirectML GPU 加速。

<p align="center">
  <img src="pic/image1.png" alt="FaceVision 主介面" width="80%"/>
</p>

---

## 📋 目錄

- [功能特色](#-功能特色)
- [技術棧](#-技術棧)
- [系統需求](#-系統需求)
- [安裝指南](#-安裝指南)
- [快速開始](#-快速開始)
- [運作原理](#-運作原理)
- [設定檔說明](#-設定檔說明)
- [專案結構](#-專案結構)
- [常見問題](#-常見問題)

---

## ✨ 功能特色

- 即時攝影機人臉偵測與辨識
- `insightface` `buffalo_l` 模型 (RetinaFace + ArcFace)
- 512 維正規化特徵向量 + 餘弦相似度 1:N 比對
- **時序追蹤** — IoU 多目標追蹤 + 滑動視窗身份投票，消除閃爍
- **品質過濾** — 模糊度偵測 + 最小人臉尺寸過濾
- DirectML GPU 加速，GPU 不可用時自動退回 CPU
- 多幀註冊 (15 幀 + 清晰度優選 + 中位數融合)
- 本地圖片匯入並選取目標人臉註冊
- **PyQt6 深色儀表板 UI** — Windows 11 Mica 背景 · 純黑風格 · 全域白色字型
- **無邊框視窗** — 自訂標題列拖曳 · 最小化/關閉按鈕
- **可捲動設定面板** — 10 項可調參數，即時生效
- UI 與 ML 推理執行緒完全分離，流暢不卡頓

---

## 🛠 技術棧

| 組件 | 技術 |
|------|------|
| **程式語言** | Python 3.9+ |
| **GUI 框架** | PyQt6 + Windows 11 Mica |
| **人臉偵測** | InsightFace RetinaFace |
| **特徵提取** | ArcFace (512 維嵌入) |
| **相似度計算** | 餘弦相似度 (1:N) |
| **時序追蹤** | IoU + 滑動視窗投票 |
| **推理後端** | ONNX Runtime (DirectML / CPU) |
| **影像處理** | OpenCV、Pillow |
| **數值計算** | NumPy |

---

## 💻 系統需求

- Windows 10 / Windows 11
- Python 3.9 或以上
- USB 或內建攝影機

---

## 📦 安裝指南

```bash
git clone https://github.com/allen902/FaceVision.git
cd facevision_py
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 可選：安裝 DirectML GPU 支援

```bash
pip uninstall onnxruntime onnxruntime-directml -y
pip install onnxruntime-directml==1.24.0
```

---

## 🚀 快速開始

```bash
python main.py
```

1. 點擊 `啟動攝影機`
2. 點擊 `新增人員`，輸入姓名並正面對著攝影機註冊
3. 或點擊 `從圖片`，選擇本地照片註冊人臉
4. 已註冊人員出現時，畫面中顯示姓名與信心度

---

## 🔧 運作原理

### 人臉偵測

- `face_detector.py` 使用 `insightface.app.FaceAnalysis(name='buffalo_l')`
- 單次推理同時取得偵測框 + 信心度 + 512d 特徵向量
- `confidence` 閾值 + `quality_filter` 模糊過濾 + `min_face_size` 尺寸過濾

### 人臉辨識

- `face_recognizer.py` 餘弦相似度 1:N 比對
- 內建編碼快取 + 版本號機制，避免重複重建

### 時序追蹤

- `face_tracker.py` IoU 比對 + 滑動視窗身份投票
- `track_smooth` 幀內多數投票一致才確認身份，避免閃爍

### 人員註冊

- 即時註冊：15 幀擷取 → 清晰度排序 → 取前 2/3 中位數融合
- 圖片匯入：支援本地圖片註冊，多張人臉可點選目標
- 資料儲存：`face_db.json` + `encodings.pkl` + `face_photos/`

### 參數設定 (10 項)

| 設定項 | 類型 | 範圍 | 說明 |
|--------|------|------|------|
| 推理裝置 | 按鈕切換 | CPU / GPU | DirectML / CUDA 自動偵測 |
| 攝影機解析度 | 下拉選單 | 8 檔 | 320×240 ~ 1280×720 |
| 偵測信心度 | 滑桿 | 0.30 ~ 0.80 | RetinaFace 偵測閾值 |
| 辨識容差 | 滑桿 | 0.30 ~ 0.80 | 餘弦相似度閾值 |
| 處理幀率 | 滑桿 | 5 ~ 60 | ML 推理幀率上限 |
| 偵測模型尺寸 | 下拉選單 | 320 / 480 / 640 | 越小越快 |
| 追蹤平滑幀數 | 滑桿 | 3 ~ 10 | 身份確認所需連續幀數 |
| 品質過濾 | 核取方塊 | 開/關 | 模糊度偵測 |
| 最小人臉尺寸 | 下拉選單 | 60 / 80 / 100 / 120 px | 小於此值過濾 |

<p align="center">
  <img src="pic/image2.png" alt="FaceVision 設定介面" width="80%"/>
</p>

---

## ⚙ 設定檔說明

程式會自動產生 `settings.json`：

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

執行時產生：

- `face_db.json` — 已註冊人員資訊
- `encodings.pkl` — 人臉特徵向量
- `face_photos/` — 註冊人臉截圖

---

## 📁 專案結構

```text
facevision_py/
├── main.py                # 程式進入點
├── config.py              # 設定管理 (settings.json)
├── camera.py              # 攝影機擷取執行緒
├── face_detector.py       # 人臉偵測 + 特徵提取
├── face_recognizer.py     # 人臉辨識 (餘弦相似度)
├── face_database.py       # 資料持續儲存 (JSON + pickle)
├── face_tracker.py        # 時序追蹤 (IoU + 投票)
├── ui_pyqt6.py            # ★ PyQt6 深色 UI (目前主介面)
├── ui_pyqt.py             # PyQt5 舊版 UI (保留參考)
├── requirements.txt       # Python 相依套件
├── CLAUDE.md              # AI Agent 上下文指南
├── README.md              # 專案說明 (英文)
├── README.zh-CN.md        # 專案說明 (簡體中文)
├── README.zh-TW.md        # 專案說明 (繁體中文)
├── settings.json          # 執行時期設定 (自動產生)
├── face_db.json           # 人員資料庫 (自動產生)
├── encodings.pkl          # 特徵向量 (自動產生)
└── face_photos/           # 註冊照片 (自動產生)
```

---

## ❓ 常見問題

- **偵測不到攝影機** — 確認攝影機已連接並允許存取
- **GPU 按鈕無法使用** — 安裝 `onnxruntime-directml` 或改用 CPU
- **註冊失敗** — 正面對著攝影機保持不動，避免遮擋和快速移動
- **辨識為「未知」** — 降低 `tolerance` 或重新註冊更清晰的樣本
- **下拉選單空白** — 已修復，QComboBox 彈出列表需獨立設定 QSS

---

## 🙏 致謝

- 特別感謝 Leon Jane 為本程式的功能測試提供其人臉支援。

---

## 📄 授權條款

MIT License — 詳見 [LICENSE](LICENSE) 檔案。

Copyright (c) 2026 [Allen](https://github.com/allen902)
