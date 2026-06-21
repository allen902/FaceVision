"""
FaceVision i18n — 中英文翻译模块
扁平化 key → {zh, en} 字典，通过 tr() 取当前语言文本
"""

# 稳定内部标记，替代硬编码 "未知"，永不被翻译
UNKNOWN_SENTINEL = "unknown"

_current_lang = "zh"
_change_listeners = []


def init_language():
    """启动时调用一次，从 APP_SETTINGS 读取语言设置"""
    global _current_lang
    from config import APP_SETTINGS
    lang = APP_SETTINGS.get("language", "zh")
    if lang in ("zh", "en"):
        _current_lang = lang


def tr(key, **kwargs):
    """取当前语言翻译文本。kwargs 用于 .format(**kwargs)。"""
    entry = _STRINGS.get(key, {})
    text = entry.get(_current_lang, entry.get("zh", key))
    if kwargs:
        try:
            return text.format(**kwargs)
        except (KeyError, ValueError):
            return text
    return text


def set_language(lang):
    """切换语言，写入 APP_SETTINGS + settings.json，触发全部 listener"""
    global _current_lang
    if lang not in ("zh", "en"):
        return
    if lang == _current_lang:
        return
    _current_lang = lang
    from config import APP_SETTINGS, save_settings
    APP_SETTINGS["language"] = lang
    save_settings(APP_SETTINGS)
    for cb in _change_listeners:
        try:
            cb(lang)
        except Exception:
            pass


def on_language_change(callback):
    """注册语言变更回调 callback(lang)，返回取消注册函数"""
    _change_listeners.append(callback)
    return lambda: _change_listeners.remove(callback) if callback in _change_listeners else None


def current_language():
    return _current_lang


# ═══════════════════════════════════════════════════════════════
# 翻译字符串库
# ═══════════════════════════════════════════════════════════════

_STRINGS = {
    # ── 窗口 & 标题栏 ──
    "window.title":             {"zh": "FaceVision",                  "en": "FaceVision"},
    "title.icon_label":         {"zh": "🔷 FaceVision",               "en": "🔷 FaceVision"},

    # ── 中央面板 ──
    "btn.start_camera":         {"zh": "▶  启动摄像头",               "en": "▶  Start Camera"},
    "btn.stop_camera":          {"zh": "⏹  停止摄像头",              "en": "⏹  Stop Camera"},
    "status.ready":             {"zh": "● 就绪",                      "en": "● Ready"},
    "status.running":           {"zh": "● 运行中",                    "en": "● Running"},
    "status.ready_full":        {"zh": "● 就绪 — 点击「启动摄像头」开始", "en": "● Ready — Click Start Camera to begin"},
    "placeholder.video":        {"zh": "📷 点击「启动摄像头」开始\n\n实时人脸识别", "en": "📷 Click 'Start Camera' to begin\n\nReal-time Face Recognition"},
    "fps.label":                {"zh": "{fps:.1f} FPS",               "en": "{fps:.1f} FPS"},

    # ── 左侧面板 - 人员卡片 ──
    "section.registered_persons": {"zh": "👤 已注册人员",             "en": "👤 Registered"},
    "btn.add":                  {"zh": "添加",                        "en": "Add"},
    "btn.import_image":         {"zh": "从图片",                      "en": "From Image"},
    "btn.delete":               {"zh": "删除",                        "en": "Delete"},
    "placeholder.no_persons":   {"zh": "  暂无注册人员",              "en": "  No registered persons"},

    # ── 左侧面板 - 快速操作 ──
    "section.quick_actions":    {"zh": "⚙ 快速操作",                 "en": "⚙ Quick Actions"},
    "btn.open_settings":        {"zh": "打开完整设置",                "en": "Open Settings"},

    # ── 左侧面板 - 设备信息 ──
    "section.device_info":      {"zh": "ℹ 设备信息",                 "en": "ℹ Device Info"},
    "info.device":              {"zh": "推理设备: {device}",           "en": "Device: {device}"},
    "info.det_size":            {"zh": "检测尺寸: {size}px",          "en": "Detection Size: {size}px"},
    "info.people_count":        {"zh": "已注册: {count} 人",          "en": "Registered: {count}"},

    # ── 设置对话框 ──
    "dialog.settings":          {"zh": "设置",                        "en": "Settings"},
    "dialog.settings_title":    {"zh": "⚙  FaceVision 设置",         "en": "⚙  FaceVision Settings"},
    "section.language":         {"zh": "语言",                        "en": "Language"},
    "lang.zh":                  {"zh": "中文",                        "en": "中文"},
    "lang.en":                  {"zh": "English",                     "en": "English"},
    "section.inference_device": {"zh": "推理设备",                    "en": "Inference Device"},
    "device.cpu":               {"zh": "CPU",                         "en": "CPU"},
    "device.gpu_dml":           {"zh": "GPU (DirectML)",              "en": "GPU (DirectML)"},
    "device.gpu_cuda":          {"zh": "GPU (CUDA)",                  "en": "GPU (CUDA)"},
    "device.gpu_unavailable":   {"zh": "GPU (不可用)",                "en": "GPU (Unavailable)"},
    "section.camera_resolution": {"zh": "摄像头分辨率",               "en": "Camera Resolution"},
    "hint.restart_for_resolution": {"zh": "更改将在下次启动摄像头时生效", "en": "Takes effect on next camera start"},
    "section.detection_confidence": {"zh": "检测置信度",              "en": "Detection Confidence"},
    "section.recognition_tolerance": {"zh": "识别容差",               "en": "Recognition Tolerance"},
    "section.processing_fps":   {"zh": "处理帧率 (FPS)",              "en": "Processing FPS"},
    "section.detection_size":   {"zh": "检测模型尺寸 (越小越快)",     "en": "Detection Model Size (smaller = faster)"},
    "hint.restart_needed":      {"zh": "需重启程序生效",              "en": "Restart required"},
    "section.tracking_smoothness": {"zh": "追踪平滑帧数",             "en": "Tracking Smooth Frames"},
    "smooth.frames_unit":       {"zh": "{v} 帧",                      "en": "{v} frames"},
    "section.quality_filter":   {"zh": "质量过滤",                    "en": "Quality Filter"},
    "quality_filter.label":     {"zh": "启用模糊度过滤",              "en": "Enable blur filtering"},
    "det_size.fast":            {"zh": "320 (快速)",                  "en": "320 (Fast)"},
    "det_size.balanced":        {"zh": "480 (均衡)",                  "en": "480 (Balanced)"},
    "det_size.accurate":        {"zh": "640 (精准)",                  "en": "640 (Accurate)"},
    "min_face.60":              {"zh": "最小 60px",                   "en": "Min 60px"},
    "min_face.80":              {"zh": "最小 80px",                   "en": "Min 80px"},
    "min_face.100":             {"zh": "最小 100px",                  "en": "Min 100px"},
    "min_face.120":             {"zh": "最小 120px",                  "en": "Min 120px"},
    "btn.cancel":               {"zh": "取消",                        "en": "Cancel"},
    "btn.apply":                {"zh": "应用",                        "en": "Apply"},
    "resolution.custom":        {"zh": "{w}×{h} (自定义)",           "en": "{w}×{h} (Custom)"},
    "dialog.resolution_changed": {"zh": "分辨率更改",                 "en": "Resolution Changed"},
    "prompt.resolution_changed_msg": {
        "zh": "摄像头分辨率已设为 {w}×{h}。\n请重启程序使新分辨率生效。",
        "en": "Camera resolution set to {w}×{h}.\nPlease restart for changes to take effect.",
    },
    "status.settings_applied":  {"zh": "✓ 设置已应用",               "en": "✓ Settings Applied"},

    # ── 添加人员对话框 ──
    "dialog.add_person":        {"zh": "添加人员",                    "en": "Add Person"},
    "dialog.register_person":   {"zh": "注册人员",                    "en": "Register Person"},
    "dialog.register_from_image": {"zh": "从图片注册",                "en": "Register from Image"},
    "prompt.enter_name_generic": {"zh": "请输入人员姓名：",           "en": "Enter person name:"},
    "prompt.enter_name_for_camera": {"zh": "请为此人输入姓名：",      "en": "Enter name for this person:"},
    "placeholder.input_name":   {"zh": "输入姓名…",                  "en": "Enter name…"},
    "btn.confirm":              {"zh": "确认",                        "en": "Confirm"},
    "dialog.input_invalid":     {"zh": "输入无效",                    "en": "Invalid Input"},
    "prompt.enter_valid_name":  {"zh": "请输入有效的姓名。",          "en": "Please enter a valid name."},

    # ── 批量删除对话框 ──
    "dialog.batch_delete":      {"zh": "批量删除人员",                "en": "Batch Delete"},
    "btn.select_all":           {"zh": "全选",                        "en": "Select All"},
    "btn.deselect_all":         {"zh": "全不选",                      "en": "Deselect All"},
    "btn.delete_selected":      {"zh": "删除选中",                    "en": "Delete Selected"},
    "prompt.select_to_delete":  {"zh": "勾选要删除的人员：",          "en": "Select persons to delete:"},
    "dialog.no_selection":      {"zh": "未选择",                      "en": "No Selection"},
    "prompt.select_at_least_one": {"zh": "请至少选择一名人员。",      "en": "Please select at least one person."},

    # ── 人脸选择对话框 ──
    "dialog.select_face":       {"zh": "选择要注册的人脸",            "en": "Select Face to Register"},
    "prompt.multi_face":        {"zh": "画面中有多张人脸，请点击要注册的人脸：", "en": "Multiple faces detected. Click the face to register:"},
    "face.card_label":          {"zh": "人脸 #{i}  ({conf:.0%})",    "en": "Face #{i}  ({conf:.0%})"},

    # ── 通用对话框 ──
    "dialog.camera_error":      {"zh": "摄像头错误",                  "en": "Camera Error"},
    "dialog.encoding_failed":   {"zh": "编码失败",                    "en": "Encoding Failed"},
    "prompt.cannot_extract_feature_retry": {"zh": "无法提取人脸特征，请重试。", "en": "Cannot extract face features. Please try again."},
    "dialog.add_failed":        {"zh": "添加失败",                    "en": "Add Failed"},
    "dialog.image_error":       {"zh": "图片错误",                    "en": "Image Error"},
    "prompt.cannot_read_image": {"zh": "无法读取图片文件。",          "en": "Cannot read image file."},
    "dialog.no_face_detected":  {"zh": "未检测到人脸",                "en": "No Face Detected"},
    "prompt.no_clear_face":     {"zh": "所选图片中未检测到清晰人脸。","en": "No clear face detected in the selected image."},
    "prompt.cannot_extract_feature": {"zh": "无法提取人脸特征。",     "en": "Cannot extract face features."},
    "dialog.error":             {"zh": "错误",                        "en": "Error"},
    "prompt.import_failed":     {"zh": "导入失败: {error}",           "en": "Import failed: {error}"},
    "dialog.tip":               {"zh": "提示",                        "en": "Info"},
    "prompt.no_registered":     {"zh": "没有已注册的人员。",          "en": "No registered persons."},
    "dialog.select_image":      {"zh": "选择图片",                    "en": "Select Image"},
    "image_filter":             {"zh": "Images (*.jpg *.jpeg *.png *.bmp *.webp)", "en": "Images (*.jpg *.jpeg *.png *.bmp *.webp)"},

    # ── 状态栏消息 ──
    "status.deleted_count":     {"zh": "✓ 已删除 {count} 人",        "en": "✓ Deleted {count} person(s)"},
    "status.add_success":       {"zh": "✓ 已添加: {name}",           "en": "✓ Added: {name}"},
    "status.add_failed_exists": {"zh": "人员 '{name}' 已存在",       "en": "Person '{name}' already exists"},

    # ── cv2 人脸叠加文字 ──
    "face.unknown_with_sim":    {"zh": "未知 ({sim:.0%})",           "en": "Unknown ({sim:.0%})"},

    # ── 处理线程日志 ──
    "thread.debug_info":        {
        "zh": "[ProcessingThread] 帧#{count}: 检测到 {n_faces} 人脸, {n_valid} 通过质量检查, 追踪数={n_tracks}, 帧shape={shape}",
        "en": "[ProcessingThread] Frame #{count}: detected {n_faces} faces, {n_valid} passed quality check, tracks={n_tracks}, shape={shape}",
    },

    # ── 控制台消息 ──
    "console.no_camera_warn":   {"zh": "⚠ 未检测到可用摄像头！",        "en": "⚠ No camera detected!"},
    "console.check_camera":     {"zh": "  请检查摄像头是否已连接，然后重新启动程序。", "en": "  Please check your camera connection and restart."},
    "console.press_enter_to_exit": {"zh": "按 Enter 退出…",             "en": "Press Enter to exit…"},
}
