"""
FaceVision — 实时人脸识别系统
主入口

架构:
  camera.py         — 摄像头采集线程 (640x360 @ 15fps)
  face_detector.py  — YOLOv8n 人体检测 → 推算面部框
  face_recognizer.py — facenet-pytorch 编码比对
  face_database.py  — JSON + pic
  ui.py             — CustomTkinter UI + 独立处理线程
"""

import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from face_database import FaceDatabase
from face_detector import FaceDetector
from face_recognizer import FaceRecognizer
from camera import CameraThread
from ui import FaceVisionApp, APP_SETTINGS


def main():
    
    print("=" * 50)
    print("  FaceVision — 实时人脸识别系统")
    print("  RetinaFace + insightface + CustomTkinter")
    print("=" * 50)

    device = APP_SETTINGS.get("device", "cpu")
    if device == "cuda":
        print("  >>> 当前设备: GPU (DirectML) <<<")
    else:
        print("  >>> 当前设备: CPU <<<")
    print()

    print("[1/4] 加载人脸数据库…")
    db = FaceDatabase()
    print(f"      已注册 {len(db.get_names())} 人")

    device = APP_SETTINGS.get("device", "cpu")
    print(f"[2/4] 加载人脸检测模型 (RetinaFace + DirectML, device={device})…")
    detector = FaceDetector(
        confidence=APP_SETTINGS.get("confidence", 0.25),
        device=device
    )

    print("[3/4] 初始化人脸识别器…")
    recognizer = FaceRecognizer(
        tolerance=APP_SETTINGS.get("tolerance", 0.5),
        device=device
    )

    print("[4/4] 初始化摄像头…")
    cameras = CameraThread.list_cameras()
    if not cameras:
        print("⚠ 未检测到可用摄像头！")
        print("  请检查摄像头是否已连接，然后重新启动程序。")
        input("按 Enter 退出…")
        sys.exit(1)

    print(f"  可用摄像头: {cameras}")
    cam_fps = APP_SETTINGS.get("cam_fps", 30)
    camera = CameraThread(
        camera_id=cameras[0],
        width=APP_SETTINGS.get("cam_width", 640),
        height=APP_SETTINGS.get("cam_height", 360),
        fps=cam_fps
    )

    print("\n✓ 初始化完成，启动界面…\n")
    app = FaceVisionApp(camera, db, detector, recognizer)
    app.mainloop()


if __name__ == "__main__":
    main()