# -*- coding: utf-8 -*-
"""ROI 标注绘制模块。

提供跨平台中文字体加载和 ROI 图像标注框/文字绘制功能。
字体搜索优先级：项目捆绑字体 → Windows 系统字体 → Linux 系统字体 → PIL 默认。
"""

import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 项目根目录（Logo_ROI_Post_Processing/ 的上一级）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 项目捆绑字体目录（跨平台首选，随项目分发）
_BUNDLED_FONTS_DIR = _PROJECT_ROOT / "Brand_Resource_Libraries" / "Fonts_Resource"

# 跨平台字体搜索路径（按优先级排列）
_FONT_SEARCH_PATHS = [
    str(_BUNDLED_FONTS_DIR),                      # 项目捆绑字体（跨平台首选）
    "C:/Windows/Fonts",                            # Windows 系统字体
    "/usr/share/fonts",                            # Linux 通用字体
    "/usr/local/share/fonts",                      # Linux 本地安装字体
    os.path.expanduser("~/.fonts"),                # Linux 用户级字体
    "/usr/share/fonts/truetype",                   # Ubuntu 常见字体根
    "/usr/share/fonts/opentype",                   # Linux OpenType 字体
    "/usr/share/fonts/truetype/wqy",               # 文泉驿字体
    "/usr/share/fonts/truetype/noto",              # Noto CJK 字体
]

# 中文字体文件名（按优先级排列，涵盖 Windows 和 Linux 常见中文字体）
_CHINESE_FONT_FILES = [
    "msyh.ttc",                   # 微软雅黑 (Windows / 捆绑)
    "msyh.ttf",                   # 微软雅黑 (备选扩展名)
    "simhei.ttf",                 # 黑体 (Windows / 捆绑)
    "simsun.ttc",                 # 宋体 (Windows / 捆绑)
    "NotoSansCJK-Regular.ttc",    # Noto CJK (Linux 常见)
    "NotoSansSC-Regular.otf",     # Noto 思源黑体
    "wqy-microhei.ttc",           # 文泉驿微米黑
    "wqy-zenhei.ttc",             # 文泉驿正黑
    "DroidSansFallback.ttf",      # Droid 回退字体
    "SourceHanSansSC-Regular.otf",  # 思源黑体
]


def _load_chinese_font(font_size: int):
    """跨平台加载中文字体。

    搜索顺序：
        1. 项目捆绑字体目录 (Brand_Resource_Libraries/Fonts_Resource)
        2. Windows 系统字体目录 (C:/Windows/Fonts)
        3. Linux 系统字体目录 (/usr/share/fonts 等)
        4. PIL 默认字体（最后回退，不支持中文）

    Args:
        font_size: 字体大小（像素）

    Returns:
        PIL ImageFont 对象
    """
    for font_dir in _FONT_SEARCH_PATHS:
        if not font_dir or not Path(font_dir).exists():
            continue
        for font_file in _CHINESE_FONT_FILES:
            font_path = Path(font_dir) / font_file
            if font_path.exists():
                try:
                    return ImageFont.truetype(str(font_path), font_size)
                except Exception:
                    continue

    # 所有字体加载失败，回退到 PIL 默认字体（不支持中文）
    return ImageFont.load_default()


def draw_roi_annotations(roi_image, roi_elements, roi_logo_bbox):
    """
    在 ROI 图像副本上绘制标注框和文字标签（支持中文）。

    标注框线条粗细和文字大小根据 ROI 图像尺寸自适应缩放。
    使用 PIL 绘制文字以支持中文，英文不截断。
    字体加载兼容 Windows 和 Linux/Ubuntu。

    参数:
        roi_image:     ROI 图像 (h, w, 3) BGR
        roi_elements:  [(text, (x1,y1,x2,y2), orientation), ...]
        roi_logo_bbox: (x1, y1, x2, y2)

    返回:
        标注后的图像副本（BGR numpy）
    """
    h, w = roi_image.shape[:2]
    annotated = roi_image.copy()

    # 根据图像尺寸自适应缩放
    scale = max(1.0, min(w, h) / 400.0)
    thickness = max(1, int(scale * 2))
    font_size = max(12, int(scale * 16))

    font = _load_chinese_font(font_size)

    # 先用 PIL 测量文字尺寸，收集所有标签信息
    tmp_pil = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    tmp_draw = ImageDraw.Draw(tmp_pil)

    labels = []  # [(x1, y1, x2, y2, label_text, label_y, tw, th, color), ...]

    # logo 标签
    if roi_logo_bbox:
        x1, y1, x2, y2 = [int(v) for v in roi_logo_bbox]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), thickness)
        label = "csg_logo"
        tb = tmp_draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        label_y = y1 - th - 6 if y1 - th - 6 > 0 else y1 + 6
        cv2.rectangle(annotated, (x1, label_y), (x1 + tw + 8, label_y + th + 6), (0, 255, 0), -1)
        labels.append((x1, label_y, label, (255, 255, 255)))

    # 文字标签
    for text, bbox_coords, ori in roi_elements:
        x1, y1, x2, y2 = [int(v) for v in bbox_coords]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), thickness)

        label = text  # 不截断，完整显示
        tb = tmp_draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        label_y = y1 - th - 6 if y1 - th - 6 > 0 else y1 + 6
        cv2.rectangle(annotated, (x1, label_y), (x1 + tw + 8, label_y + th + 6), (0, 0, 255), -1)
        labels.append((x1, label_y, label, (255, 255, 255)))

    # 所有矩形画完后，转 PIL 统一画文字
    pil_img = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    for x, y, label, color in labels:
        draw.text((x + 4, y + 2), label, fill=color, font=font)

    # 转回 BGR numpy
    annotated = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return annotated
