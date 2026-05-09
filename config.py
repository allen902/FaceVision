"""
FaceVision 全局配置
"""
import json
import os

SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "settings.json")

DEFAULT_SETTINGS = {
    "device": "cuda",         # cpu / cuda
    "confidence": 0.25,
    "tolerance": 0.45,        # buffalo_l 余弦相似度阈值: 0.40=严格, 0.45=推荐, 0.50=宽松
    "cam_width": 640,
    "cam_height": 360,
    "cam_fps": 30,
    "proc_fps": 12,           # 降低处理帧率减轻 GPU/CPU 负载
}


def load_settings():
    """从 settings.json 加载设置，文件不存在则返回默认值"""
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            # 合并默认值，确保新字段不会丢失
            merged = DEFAULT_SETTINGS.copy()
            merged.update(saved)
            return merged
        except (json.JSONDecodeError, IOError):
            pass
    return DEFAULT_SETTINGS.copy()


def save_settings(settings):
    """保存设置到 settings.json"""
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
    except IOError as e:
        print(f"[Config] 保存设置失败: {e}")


APP_SETTINGS = load_settings()

# Windows 11 主题色
ACCENT_COLOR = "#0078D4"
ACCENT_HOVER = "#005A9E"
DARK_BG = "#1C1C1C"
DARK_SURFACE = "#2D2D2D"
DARKER_SURFACE = "#252525"
LIGHT_BG = "#F3F3F3"
LIGHT_SURFACE = "#FAFAFA"
TEXT_PRIMARY = "#FFFFFF"
TEXT_SECONDARY = "#AAAAAA"
BORDER_COLOR = "#404040"
SUCCESS_COLOR = "#10893E"
DANGER_COLOR = "#C42B1C"