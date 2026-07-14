# -*- coding: utf-8 -*-
"""
背景检测程序

基于 logo_run_func.py 流程，额外增加背景区域分析：
  - OCR 跳过 logo 检测框内的区域（后置过滤，不修改原图）
  - 保留所有检测框，不筛除重复区域
  - 仅打印结果，不输出 JSON/文件
  - 对 ROI 内排除文字+logo 后的背景区域进行颜色检测与整洁度判别
  - 通过 matplotlib 窗口展示：标注图片、直方图、色号、分析信息
"""

import sys
import re
import json
import base64
import tempfile
import numpy as np
import cv2
import requests
from pathlib import Path
from collections import Counter
from PIL import Image, ImageDraw, ImageFont

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

# 抑制关闭窗口时的 Tkinter TclError
import tkinter as tk
tk.Tk.report_callback_exception = lambda self, *args: None

# =========================== 中文字体配置 ===========================
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from ocr_fun import run_ocr
from OCR_Tuple_Information_Filtering import filter_tuples
from Yolov26_Logo_Object_Detection.yolo26_detect import detect_logo
from Logo_ROI_Post_Processing import process_logo_roi


# =========================== 辅助函数 ===========================

def _poly_to_bbox(poly):
    """多边形坐标 → 外接矩形 (x1, y1, x2, y2)"""
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return (min(xs), min(ys), max(xs), max(ys))


def _bbox_area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _bbox_intersection(b1, b2):
    """两矩形交集面积"""
    x_overlap = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
    y_overlap = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
    return x_overlap * y_overlap


def _bbox_overlap_ratio(b1, b2):
    """交集占较小框面积的比例"""
    inter = _bbox_intersection(b1, b2)
    if inter == 0:
        return 0.0
    return inter / min(_bbox_area(b1), _bbox_area(b2))


def _bbox_iou(b1, b2):
    """IoU = 交集 / 并集"""
    inter = _bbox_intersection(b1, b2)
    if inter == 0:
        return 0.0
    union = _bbox_area(b1) + _bbox_area(b2) - inter
    return inter / union if union > 0 else 0.0


def _deduplicate_logo_detections(detections, iou_thresh=0.5):
    """
    对 logo 检测结果做 NMS 去重，一片区域内只保留置信度最高的一个框。
    返回去重后的列表 [(label, (x1,y1,x2,y2), conf), ...]
    """
    if len(detections) <= 1:
        return detections
    # 按置信度降序
    sorted_dets = sorted(detections, key=lambda x: x[2], reverse=True)
    keep = []
    for det in sorted_dets:
        overlap = any(_bbox_iou(det[1], k[1]) > iou_thresh for k in keep)
        if not overlap:
            keep.append(det)
    return keep


# ---- 中文标注（PIL 绘制，解决 OpenCV putText 不支持中文的问题） ----

def _load_chinese_font(font_size=20):
    """加载中文字体，按优先级尝试"""
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",       # 微软雅黑
        "C:/Windows/Fonts/msyhbd.ttc",     # 微软雅黑 粗体
        "C:/Windows/Fonts/simhei.ttf",     # 黑体
        "C:/Windows/Fonts/simsun.ttc",     # 宋体
        "C:/Windows/Fonts/simkai.ttf",     # 楷体
    ]
    for fp in font_paths:
        try:
            return ImageFont.truetype(fp, font_size)
        except Exception:
            continue
    return ImageFont.load_default()


def _draw_text_pil(img_bgr, text, xy, color_bgr, font_size=16, thickness=None):
    """
    用 PIL 在 BGR 图像上绘制中文文字。
    返回修改后的图像（原地修改）。
    """
    if not text:
        return img_bgr
    font = _load_chinese_font(font_size)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)
    # PIL 用 RGB
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    draw.text(xy, text, fill=color_rgb, font=font)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# =========================== 颜色分析 ===========================

def _kmeans_colors(pixels, k=5):
    """OpenCV K-Means 提取主导色，返回 [(BGR, ratio), ...] 按频率降序"""
    n = len(pixels)
    if n == 0:
        return []
    if n > 10000:  # 采样加速
        idx = np.random.choice(n, 10000, replace=False)
        pixels = pixels[idx]
    pixels_f32 = pixels.astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    k = min(k, len(pixels_f32))
    _, labels, centers = cv2.kmeans(
        pixels_f32, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    centers = centers.astype(np.uint8)
    cnt = Counter(labels.flatten())
    total = sum(cnt.values())
    result = []
    for cluster_id, count in cnt.most_common():
        result.append((tuple(centers[cluster_id].tolist()), count / total))
    return result


def bgr_to_hex(bgr):
    """BGR tuple → '#RRGGBB'"""
    return "#{:02X}{:02X}{:02X}".format(bgr[2], bgr[1], bgr[0])


# =========================== 背景掩码 ===========================

def create_background_mask(h, w, logo_bboxes, text_bboxes):
    """在 ROI 图像上创建背景掩码：背景=1，logo/文字区域=0"""
    mask = np.ones((h, w), dtype=np.uint8)
    for (x1, y1, x2, y2) in list(logo_bboxes) + list(text_bboxes):
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 0
    return mask


# =========================== 大模型分析（分类 + 背景质量，合二为一） ===========================

# 三张模板 logo（固定）
_TEMPLATE_PATHS = [
    "input/A1.04_品牌标识的颜色版本_1_标准彩色版本.png",
    "input/A1.04_品牌标识的颜色版本_2_反白版本.png",
    "input/A1.04_品牌标识的颜色版本_3_墨稿版本.png",
]
_TEMPLATE_LABELS = ["标准彩色版本", "反白版本", "墨稿版本"]

OLLAMA_URL = "http://localhost:12347/api/chat"
OLLAMA_MODEL = "qwen3-vl:8b"


def _img_to_b64(img_bgr):
    """BGR 图像 → base64 字符串"""
    _, buf = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buf).decode()


def _llm_analyze(roi_bgr, dominant_colors):
    """
    一次调用完成：Logo 颜色版本判别 + 背景质量评估。
    发送 3 模板 + ROI + 颜色数据，返回结构化文本，失败返回 None。
    """
    base_dir = Path(__file__).resolve().parent

    # 颜色数据
    color_lines = []
    for bgr, ratio in dominant_colors:
        hex_c = bgr_to_hex(bgr)
        color_lines.append(f"  {hex_c}  占比 {ratio*100:.1f}%")
    color_text = "\n".join(color_lines) if color_lines else "  (无)"

    # images: 3 模板 + 1 ROI
    all_b64 = []
    for tp in _TEMPLATE_PATHS:
        fp = base_dir / tp
        if not fp.exists():
            print(f"  [LLM] 模板不存在: {fp}")
            return None
        all_b64.append(base64.b64encode(fp.read_bytes()).decode())
    all_b64.append(_img_to_b64(roi_bgr))

    prompt = (
        "前面3张是模板logo，按顺序：1标准彩色版本、2反白版本、3墨稿版本。"
        "第4张是待判别的输入ROI图像。\n"
        "\n"
        "墨稿版本logo为黑色；彩色版本logo颜色与反白参考蓝色 #00377A 一致。\n"
        "反白版本参考蓝色: RGB(0,55,122) #00377A（仅作参考色调，不要求严格匹配）\n"
        "\n"
        "背景主导颜色（K-Means聚类，已筛除logo和文字检测框区域，仅统计背景像素）:\n"
        f"{color_text}\n"
        "\n"
        "判断规则:\n"
        "- 根据模板判别输入logo属于哪个颜色版本\n"
        "- 标准彩色/墨稿版本: 要求背景颜色为同一色调即可，可为任意色调，可不为白色（重点）\n"
        "- 反白版本: 背景应为蓝色系（参考 #00377A 色调），此为唯一硬性要求\n"
        "- logo/文字上的瑕疵、脏污、以及背景中的水印、污渍均判定为不干净\n"
        "- logo/文字边缘、正常排版线条、阴影，反光不算杂乱\n"
        "- 色号数据是ROI图像过滤logo与文字检测框后得到的，输出背景色号时，需结合图像中logo附近区域综合作出判断\n"
        "\n"
        "请严格按以下格式逐行输出，不要多余内容:\n"
        "类型: 标准彩色版本/反白版本/墨稿版本\n"
        "背景色号: 色调名称(如蓝色色调)；#XXXXXX RGB(x,x,x)\n"
        "是否符合要求: 是/否\n"
        "是否干净: 是/否\n"
        "描述解释: 详细描述背景情况，包含是否存在反光、阴影、残缺、脏污等问题...\n"
    )

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt, "images": all_b64}],
        "stream": False,
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=None)
        resp.raise_for_status()
        raw = resp.json()["message"]["content"].strip()
        return _parse_llm_output(raw)
    except requests.ConnectionError:
        print("  [LLM] SSH 隧道未连接，跳过大模型分析")
        return None
    except Exception as e:
        print(f"  [LLM] 请求失败: {e}")
        return None


def _parse_llm_output(raw: str) -> dict:
    """解析 LLM 输出的结构化文本，提取为 dict"""
    result = {}
    for line in raw.split("\n"):
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        if key == "类型":
            result["logo_version"] = val
        elif key == "背景色号":
            # "深蓝色 #00377A RGB(0,55,122)" → {tone, hex, rgb}
            m = re.match(r"(.+?)\s+(#[0-9A-Fa-f]{6})\s+RGB\((\d+),\s*(\d+),\s*(\d+)\)", val)
            if m:
                result["bg_color_tone"] = m.group(1).strip()
                result["bg_color_hex"] = m.group(2).upper()
                result["bg_color_rgb"] = [int(m.group(3)), int(m.group(4)), int(m.group(5))]
            else:
                result["bg_color_tone"] = val
                result["bg_color_hex"] = ""
                result["bg_color_rgb"] = []
        elif key == "是否符合要求":
            result["is_compliant"] = val.startswith("是")
        elif key == "是否干净":
            result["is_clean"] = val.startswith("是")
        elif key == "是否包含杂乱背景":
            has = val.startswith("是")
            result["has_clutter"] = has
            result["clutter_location"] = val.split("，", 1)[-1].strip() if "，" in val and has else ("" if not has else val)
        elif key == "描述解释":
            result["description"] = val
    return result if result else None


# =========================== 绘图 ===========================

def draw_histogram(ax, bg_pixels):
    """在 axes 上绘制 RGB 直方图"""
    if len(bg_pixels) == 0:
        ax.text(0.5, 0.5, "无背景像素", ha="center", va="center",
                transform=ax.transAxes)
        return
    for ch, color, label in zip([2, 1, 0], ["red", "green", "blue"], ["R 通道", "G 通道", "B 通道"]):
        hist = cv2.calcHist([bg_pixels[:, ch:ch + 1]], [0], None, [256], [0, 256])
        ax.plot(hist, color=color, linewidth=1, alpha=0.8, label=label)
    ax.set_xlim([0, 256])
    ax.set_xlabel("像素值")
    ax.set_ylabel("像素数量")
    ax.set_title("背景 RGB 直方图")
    ax.legend(fontsize=7)


def draw_swatches(ax, colors):
    """在 axes 上绘制色块"""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, max(len(colors), 1) + 1)
    ax.axis("off")
    ax.set_title("主导颜色", fontsize=11)
    if not colors:
        return
    for i, (bgr, ratio) in enumerate(colors):
        y = len(colors) - i
        rgb = (bgr[2], bgr[1], bgr[0])
        hex_c = bgr_to_hex(bgr)
        rect = plt.Rectangle((0.5, y - 0.35), 2, 0.7,
                             facecolor=tuple(c / 255 for c in rgb),
                             edgecolor="gray", linewidth=0.5)
        ax.add_patch(rect)
        ax.text(3.2, y, f"{hex_c}  {ratio*100:.1f}%", va="center",
                fontsize=9)


# =========================== 主流程 ===========================

def detect_background(input_path: str, conf_thres: float = 0.7):
    """
    背景检测主函数。

    参数:
        input_path: 输入图片路径
        conf_thres: Logo 检测置信率阈值
    """
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"[ERROR] 文件不存在: {input_path}")
        return

    img_bgr = cv2.imdecode(np.fromfile(str(input_file), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"[ERROR] 无法读取图片: {input_path}")
        return

    img_h, img_w = img_bgr.shape[:2]
    print(f"图片尺寸: {img_w} x {img_h}")

    # ==================== Step 1: OCR 文字识别 ====================
    print("\n" + "=" * 50)
    print("[Step 1] OCR 文字识别")
    print("=" * 50)
    # 使用临时目录避免产生持久文件
    with tempfile.TemporaryDirectory() as tmp_dir:
        ocr_result = run_ocr(str(input_file), tmp_dir)
    if not ocr_result.get("success"):
        print(f"OCR 失败: {ocr_result.get('message')}")
        return
    all_text_tuples = ocr_result.get("text_tuples", [])
    print(f"OCR 识别到 {len(all_text_tuples)} 个文字区域")

    # ==================== Step 2: Logo 目标检测 ====================
    print("\n" + "=" * 50)
    print("[Step 2] Logo 目标检测")
    print("=" * 50)
    logo_detections = detect_logo(img_bgr, conf_thres=conf_thres)
    print(f"检测到 {len(logo_detections)} 个 logo（含重复框）")
    # NMS 去重：一片区域仅保留一个
    logo_detections = _deduplicate_logo_detections(logo_detections, iou_thresh=0.5)
    print(f"去重后保留 {len(logo_detections)} 个 logo:")
    for label, (x1, y1, x2, y2), conf in logo_detections:
        print(f"  {label} | ({x1},{y1})-({x2},{y2}) | conf={conf:.3f}")

    # ==================== Step 3: 过滤 + 白名单 ====================
    print("\n" + "=" * 50)
    print("[Step 3] 过滤 logo 检测框内的 OCR 结果 + 白名单过滤")
    print("=" * 50)
    logo_bboxes = [det[1] for det in logo_detections]
    post_logo_tuples = []
    excluded_in_logo = 0
    for tup in all_text_tuples:
        tb = _poly_to_bbox(tup[1])
        inside = any(_bbox_overlap_ratio(tb, lb) > 0.5 for lb in logo_bboxes)
        if inside:
            excluded_in_logo += 1
        else:
            post_logo_tuples.append(tup)
    print(f"排除 {excluded_in_logo} 个在 logo 框内的文字，剩余 {len(post_logo_tuples)} 个")

    # 白名单过滤
    filtered_tuples = filter_tuples(post_logo_tuples)
    print(f"白名单过滤后保留 {len(filtered_tuples)} 个文字区域")

    for text, poly, ori in filtered_tuples:
        bbox = _poly_to_bbox(poly)
        print(f"  [{ori}°] \"{text}\" ({bbox[0]},{bbox[1]})-({bbox[2]},{bbox[3]})")

    # ==================== Step 4: ROI 后处理 ====================
    print("\n" + "=" * 50)
    print("[Step 4] ROI 后处理")
    print("=" * 50)
    # 所有 logo 检测框都传入，不过滤重复
    roi_result = process_logo_roi(filtered_tuples, logo_detections, img_bgr)
    if not roi_result.get("success"):
        print(f"ROI 处理失败: {roi_result.get('message')}")
        return

    max_box = roi_result["max_box"]
    roi_image = roi_result["roi_image"]
    roi_elements = roi_result["roi_elements"]
    roi_logo_bbox = roi_result["roi_logo_bbox"]
    layout_type = roi_result["layout_type"]

    print(f"布局类型: {layout_type}")
    print(f"最大包围盒: {max_box}")
    print(f"ROI 内 logo bbox: {roi_logo_bbox}")
    print(f"ROI 内文字区域: {len(roi_elements)} 个")
    for text, bbox, ori in roi_elements:
        print(f"  [{ori}°] \"{text}\" ({bbox[0]},{bbox[1]})-({bbox[2]},{bbox[3]})")

    # ==================== Step 5: 背景区域提取 ====================
    print("\n" + "=" * 50)
    print("[Step 5] 背景区域提取")
    print("=" * 50)
    text_bboxes_roi = [b[1] for b in roi_elements]
    roi_h, roi_w = roi_image.shape[:2]

    # 背景掩码（排除 logo + 文字）
    bg_mask = create_background_mask(
        roi_h, roi_w,
        [roi_logo_bbox] if roi_logo_bbox else [],
        text_bboxes_roi,
    )
    bg_pixels = roi_image[bg_mask == 1]

    # ==================== Step 6: 颜色检测 ====================
    print("\n" + "=" * 50)
    print("[Step 6] 背景颜色检测")
    print("=" * 50)
    dominant_colors = []
    if len(bg_pixels) > 0:
        dominant_colors = _kmeans_colors(bg_pixels, k=5)
        print("主导颜色:")
        for bgr, ratio in dominant_colors:
            hex_c = bgr_to_hex(bgr)
            print(f"  {hex_c}  (RGB: {bgr[2]},{bgr[1]},{bgr[0]})  -- {ratio*100:.1f}%")
    else:
        print("无背景像素，跳过颜色检测")

    # ==================== Step 7: 可视化（先展示窗口） ====================
    print("\n" + "=" * 50)
    print("[Step 7] 显示结果窗口（关闭窗口后将调用大模型分析）")
    print("=" * 50)

    # logo_version 尚未确定，窗口标题和信息面板标注为待分析
    logo_version = None

    annotated = img_bgr.copy()

    # ROI 最大包围盒（黄色）
    if max_box:
        cv2.rectangle(annotated, (max_box[0], max_box[1]),
                      (max_box[2], max_box[3]), (0, 255, 255), 2)

    # 去重后的 logo 检测框 - 绿色
    for label, (x1, y1, x2, y2), conf in logo_detections:
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        annotated = _draw_text_pil(annotated, f"{label} {conf:.2f}",
                                    (x1, max(y1 - 20, 2)), (0, 255, 0), font_size=14)

    # 所有 OCR 文字区域 - 红色细框
    for text, poly, ori in all_text_tuples:
        bb = _poly_to_bbox(poly)
        cv2.rectangle(annotated, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 1)

    # 白名单过滤保留的文字 - 蓝色粗框（用 PIL 写中文）
    for text, poly, ori in filtered_tuples:
        bb = _poly_to_bbox(poly)
        cv2.rectangle(annotated, (bb[0], bb[1]), (bb[2], bb[3]), (255, 0, 0), 2)
        annotated = _draw_text_pil(annotated, text[:15],
                                    (bb[0], max(bb[1] - 18, 2)), (255, 0, 0), font_size=13)

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    # ---- 屏幕尺寸自适应 ----
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        root.destroy()
    except Exception:
        screen_w, screen_h = 1920, 1080

    dpi = 100
    fig_w = (screen_w / 2) / dpi
    fig_h = (screen_h / 2) / dpi

    # ---- matplotlib 布局 (2 行 × 2 列) ----
    fig = plt.figure(figsize=(fig_w * 1.3, fig_h))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[2, 1], height_ratios=[1.8, 1])

    # ===== 第一行 =====

    # (0,0) 标注图片
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(annotated_rgb)
    ax_img.set_title(
        f"背景检测  |  布局: {layout_type}  |  "
        f"Logo版本: 待LLM分析",
        fontsize=10,
    )
    ax_img.axis("off")
    legend = [
        Patch(facecolor="green", alpha=0.5, label="Logo 检测框"),
        Patch(facecolor="red", alpha=0.5, label="全部 OCR 文字"),
        Patch(facecolor="blue", alpha=0.5, label="白名单保留文字"),
        Patch(facecolor="yellow", alpha=0.3, label="ROI 最大包围盒"),
    ]
    ax_img.legend(handles=legend, loc="lower right", fontsize=7, framealpha=0.8)

    # (0,1) 直方图
    ax_hist = fig.add_subplot(gs[0, 1])
    draw_histogram(ax_hist, bg_pixels)

    # ===== 第二行 =====

    # (1,0) 主导色色块
    ax_color = fig.add_subplot(gs[1, 0])
    draw_swatches(ax_color, dominant_colors)

    # (1,1) 分析信息面板
    ax_info = fig.add_subplot(gs[1, 1])
    ax_info.axis("off")
    info_lines = [
        "=== 背景检测分析 ===",
        "",
        f"Logo 版本 : 待LLM分析",
        "",
        f"图片尺寸 : {img_w} x {img_h}",
        f"布局类型 : {layout_type}",
        f"Logo 框  : {len(logo_detections)}",
        f"OCR 区域 : {len(all_text_tuples)}",
        f"过滤保留 : {len(filtered_tuples)}",
    ]
    ax_info.text(
        0.05, 0.95, "\n".join(info_lines),
        transform=ax_info.transAxes, fontsize=9,
        va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6),
    )

    plt.tight_layout()
    try:
        plt.show()
    except Exception:
        pass

    # ==================== Step 8: 大模型分析（窗口关闭后） ====================
    print("\n" + "=" * 50)
    print("[Step 8] 大模型分析（分类 + 背景质量评估）")
    print("=" * 50)
    llm_result = _llm_analyze(roi_image, dominant_colors)

    if llm_result:
        # 输出到控制台
        print(f"类型: {llm_result.get('logo_version', '未知')}")
        tone = llm_result.get('bg_color_tone', '')
        hex_c = llm_result.get('bg_color_hex', '')
        rgb = llm_result.get('bg_color_rgb', [])
        print(f"背景色号: {tone} {hex_c} RGB({','.join(map(str, rgb))})")
        print(f"是否符合要求: {'是' if llm_result.get('is_compliant') else '否'}")
        print(f"是否干净: {'是' if llm_result.get('is_clean') else '否'}")
        print(f"是否包含杂乱背景: {'是，' + llm_result.get('clutter_location', '') if llm_result.get('has_clutter') else '否'}")
        print(f"描述解释: {llm_result.get('description', '')}")

        # 构建 JSON 并保存（仿照 logo_run_func 风格）
        output_dir = BASE_DIR / "ocr"
        output_dir.mkdir(parents=True, exist_ok=True)
        bg_json = {
            "success": True,
            "logo_version": llm_result.get("logo_version"),
            "bg_color": {
                "tone": llm_result.get("bg_color_tone"),
                "hex": llm_result.get("bg_color_hex"),
                "rgb": llm_result.get("bg_color_rgb"),
            },
            "is_compliant": llm_result.get("is_compliant"),
            "is_clean": llm_result.get("is_clean"),
            "has_clutter": llm_result.get("has_clutter"),
            "clutter_location": llm_result.get("clutter_location"),
            "description": llm_result.get("description"),
        }
        json_path = output_dir / "background_analysis.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(bg_json, f, ensure_ascii=False, indent=2)
        print(f"\n[JSON] 已保存: {json_path}")
    else:
        print(">>> 大模型分析失败或跳过")

    print("\n=== 背景检测完成 ===")


# =========================== 入口 ===========================

if __name__ == "__main__":
    default_input = str(BASE_DIR / "image" /"bad"/ "nanwan_pic_0kyt.jpeg")
    input_path = sys.argv[1] if len(sys.argv) > 1 else default_input
    detect_background(input_path)
