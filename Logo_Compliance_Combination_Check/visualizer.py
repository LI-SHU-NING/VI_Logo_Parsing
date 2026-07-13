# -*- coding: utf-8 -*-
"""
合规检测可视化模块 — 固定面板版。

- 右侧面板固定 420px 宽，信息始终清晰可读
- 左侧图片自适应剩余空间，保持原始宽高比
- 面板仅显示 A 倍数据，不显示像素值
- 窗口自动适配屏幕 ≤ 90%
"""
import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── 渲染参数 ──
PANEL_WIDTH = 480          # 右侧面板固定宽度
MIN_FONT_SIZE = 18
SCREEN_RATIO = 0.90


def _find_font_file():
    for fp in [
        "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/msjh.ttc",
    ]:
        try:
            ImageFont.truetype(fp, 24)
            return fp
        except Exception:
            continue
    return None


FONT_PATH = _find_font_file()


def _load_cjk_font(size=24):
    if FONT_PATH:
        return ImageFont.truetype(FONT_PATH, size)
    return ImageFont.load_default()


# ── 屏幕适配 ──

_screen_w = _screen_h = None


def _get_screen_size():
    global _screen_w, _screen_h
    if _screen_w is None:
        try:
            root = tk.Tk()
            root.withdraw()
            _screen_w = root.winfo_screenwidth()
            _screen_h = root.winfo_screenheight()
            root.destroy()
        except Exception:
            _screen_w = 1920
            _screen_h = 1080
    return _screen_w, _screen_h


# ── 基础绘图 ──

def _draw_rect(canvas, bbox, color, thick):
    if bbox is None:
        return
    x1, y1, x2, y2 = bbox
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thick, cv2.LINE_AA)


def _draw_line(canvas, pt1, pt2, color, thick):
    cv2.line(canvas, pt1, pt2, color, thick, cv2.LINE_AA)


def _draw_circle(canvas, center, radius, color, fill=-1):
    cv2.circle(canvas, center, radius, color, fill, cv2.LINE_AA)


# ── 条件解析 ──

def _parse_req(cond):
    """从条件字符串提取可读范围"""
    import re
    nums = [float(m) for m in re.findall(r'([\d.]+)\s*\*\s*A', cond)]
    ops = re.findall(r'([><]=?)\s*[\d.]+\s*\*\s*A', cond)
    if not nums:
        return ""
    lo, hi = None, None
    for op, val in zip(ops, nums):
        if op in ('<', '<='):
            hi = val if hi is None else min(hi, val)
        else:
            lo = val if lo is None else max(lo, val)
    if lo is not None and hi is not None and lo > hi:
        lo, hi = hi, lo
    if lo is not None and hi is not None:
        return f"{lo:.2f}-{hi:.2f}A"
    if lo is not None:
        return f">{lo:.2f}A"
    if hi is not None:
        return f"<{hi:.2f}A"
    return ""


class _Panel:
    def __init__(self, violations=None):
        self.boxes = []
        self.gaps = []
        self.violations = violations or []

    def add_box(self, label, color, bbox, A):
        if bbox is None:
            return
        w_A = (bbox[2] - bbox[0]) / A
        h_A = (bbox[3] - bbox[1]) / A
        rgb = (color[2], color[1], color[0]) if len(color) == 3 else color
        self.boxes.append((label, rgb, w_A, h_A))

    def add_gap(self, label, color, px, A):
        rgb = (color[2], color[1], color[0]) if len(color) == 3 else color
        self.gaps.append((label, rgb, px / A))

    def render(self, draw, x0, y0, font, font_sm, img_h):
        pad = int(font_sm.size * 0.8)
        indent = pad + 6
        row_h = int(font_sm.size * 1.8)
        cy = y0 + pad

        # ── 框体尺寸 ──
        if self.boxes:
            draw.text((x0 + pad, cy), "框体尺寸:", fill=(80, 80, 80), font=font)
            cy += row_h
            draw.text((x0 + indent, cy), "名称", fill=(130, 130, 130), font=font_sm)
            draw.text((x0 + 170, cy), "W(A)", fill=(130, 130, 130), font=font_sm)
            draw.text((x0 + 290, cy), "H(A)", fill=(130, 130, 130), font=font_sm)
            cy += row_h

            for label, rgb, w_A, h_A in self.boxes:
                if cy + row_h > img_h:
                    break
                sq = max(8, font_sm.size - 4)
                draw.rectangle([x0 + indent, cy + 4, x0 + indent + sq, cy + 4 + sq],
                               fill=rgb, outline=(180, 180, 180))
                draw.text((x0 + indent + sq + 4, cy), label, fill=(50, 50, 50), font=font_sm)
                draw.text((x0 + 170, cy), f"{w_A:.2f}", fill=(50, 50, 50), font=font_sm)
                draw.text((x0 + 290, cy), f"{h_A:.2f}", fill=(50, 50, 50), font=font_sm)
                cy += row_h

        # ── 间距测量 ──
        if self.gaps:
            cy += pad * 2
            draw.text((x0 + pad, cy), "间距测量:", fill=(80, 80, 80), font=font)
            cy += row_h
            draw.text((x0 + indent, cy), "名称", fill=(130, 130, 130), font=font_sm)
            draw.text((x0 + 280, cy), "A倍", fill=(130, 130, 130), font=font_sm)
            cy += row_h

            for label, rgb, a_val in self.gaps:
                if cy + row_h > img_h:
                    break
                sq = max(8, font_sm.size - 4)
                draw.rectangle([x0 + indent, cy + 4, x0 + indent + sq, cy + 4 + sq],
                               fill=rgb, outline=(180, 180, 180))
                draw.text((x0 + indent + sq + 4, cy), label, fill=(50, 50, 50), font=font_sm)
                draw.text((x0 + 280, cy), f"{a_val:.2f}", fill=(50, 50, 50), font=font_sm)
                cy += row_h

        # ── 违规详情 ──
        if self.violations:
            cy += pad * 2
            draw.text((x0 + pad, cy), "违规项:", fill=(200, 40, 40), font=font)
            cy += row_h
            for v in self.violations:
                if cy + row_h > img_h:
                    break
                name = v.get("name", "")
                cond = v.get("condition", "")
                req = _parse_req(cond)
                line = f"! {name}"
                if req:
                    line += f"  应: {req}"
                draw.text((x0 + indent, cy), line, fill=(200, 40, 40), font=font_sm)
                cy += row_h


# ── 渲染输出 ──

def _render_panel_window(img_canvas, img_w, img_h, texts, panel, info, font, font_sm,
                          file_name, A_raw):
    """组装 [图像 + 480px 面板] 并显示。
    违规时图像区域叠加红色边框警告。
    """
    MIN_WIN_H = 700
    total_w = img_w + PANEL_WIDTH
    total_h = max(img_h, MIN_WIN_H)

    canvas = np.full((total_h, total_w, 3), 255, dtype=np.uint8)

    # 图像居中
    y_off = max(0, (total_h - img_h) // 2)
    canvas[y_off:y_off + img_h, :img_w] = img_canvas

    # 违规时：图像区域外红色边框
    has_violations = bool(panel.violations)
    if has_violations:
        border = 4
        cv2.rectangle(canvas,
                      (y_off, y_off),
                      (img_w - y_off - 1, y_off + img_h - 1),
                      (50, 50, 255), border, cv2.LINE_AA)
        # 顶部警告条
        cv2.rectangle(canvas, (0, y_off), (img_w - 1, y_off + 30),
                      (50, 50, 255), -1, cv2.LINE_AA)
        # 用 PIL 写文字
        pil_tmp = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        tmp_draw = ImageDraw.Draw(pil_tmp)
        tmp_draw.text((img_w // 2 - 40, y_off + 4), "FAIL", fill=(255, 255, 255), font=font)
        canvas = cv2.cvtColor(np.array(pil_tmp), cv2.COLOR_RGB2BGR)

    # 分隔线
    cv2.line(canvas, (img_w, 0), (img_w, total_h - 1), (200, 200, 200), 1, cv2.LINE_AA)

    # 超屏缩放
    sw, sh = _get_screen_size()
    max_w = int(sw * SCREEN_RATIO)
    max_h = int(sh * SCREEN_RATIO)
    if total_w > max_w or total_h > max_h:
        scale = min(max_w / total_w, max_h / total_h)
        new_total_w = int(total_w * scale)
        new_total_h = int(total_h * scale)
        canvas = cv2.resize(canvas, (new_total_w, new_total_h), interpolation=cv2.INTER_AREA)
        total_w, total_h = new_total_w, new_total_h
        new_img_w = int(img_w * scale)
        img_w = new_img_w
        y_off = int(y_off * scale)

    pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    draw.text((10, 10), info["text"], fill=info.get("fill", (255, 255, 255)), font=font)

    for x, y, content, color in texts:
        if x < img_w:
            draw.text((x, y + y_off), content, fill=color, font=font)

    status = "PASS" if not has_violations else "FAIL"
    sc = (0, 180, 0) if not has_violations else (200, 40, 40)
    draw.text((img_w + 10, 10), f"A = {A_raw:.0f} px  {status}", fill=sc, font=font)

    panel.render(draw, img_w, 36, font, font_sm, total_h)

    canvas = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    cv2.namedWindow(file_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(file_name, total_w, total_h)
    cv2.imshow(file_name, canvas)
    print(f"  按任意键关闭 {file_name} 窗口...")
    cv2.waitKey(0)
    cv2.destroyWindow(file_name)


# ════════════════════════════════════════════════════════════
#  中心参考窗口
# ════════════════════════════════════════════════════════════

def show_centers_window(result: dict):
    img = result.get("_img")
    if img is None:
        return

    result_mut = dict(result)
    img_h, img_w = img.shape[:2]
    logo_bbox = result_mut.get("_logo_bbox")
    A = result_mut.get("A", 1)

    # 左侧字体根据图像短边；面板字体独立
    base_dim = min(img_w, img_h)
    font_size = max(MIN_FONT_SIZE, int(base_dim / 45))
    font = _load_cjk_font(font_size)
    panel_font_size = max(MIN_FONT_SIZE, int(PANEL_WIDTH / 20))
    font_sm = _load_cjk_font(panel_font_size)

    thick = max(2, int(base_dim / 300))
    texts = []
    panel = _Panel(result_mut.get("violations"))

    canvas = img.copy()

    # === 所有 OCR 原始文字框（浅灰虚线框） ===
    all_boxes = result_mut.get("_all_text_boxes", [])
    for bbox in all_boxes:
        if bbox:
            x1, y1, x2, y2 = bbox
            for i in range(0, 10, 3):
                cv2.line(canvas, (x1 + i, y1), (min(x1 + i + 2, x2), y1), (180, 180, 180), 1)
                cv2.line(canvas, (x1 + i, y2), (min(x1 + i + 2, x2), y2), (180, 180, 180), 1)
                cv2.line(canvas, (x1, y1 + i), (x1, min(y1 + i + 2, y2)), (180, 180, 180), 1)
                cv2.line(canvas, (x2, y1 + i), (x2, min(y1 + i + 2, y2)), (180, 180, 180), 1)

    # === Logo 基准十字线 ===
    if logo_bbox:
        lx1, ly1, lx2, ly2 = logo_bbox
        _draw_rect(canvas, logo_bbox, (0, 255, 0), thick)
        panel.add_box("Logo", (0, 255, 0), logo_bbox, A)
        lcx = (lx1 + lx2) // 2
        lcy = (ly1 + ly2) // 2

        v_up = max(0, int(lcy - 0.5 * A))
        v_down = min(img_h - 1, int(lcy + 0.5 * A))
        _draw_line(canvas, (lcx, v_up), (lcx, v_down), (0, 220, 255), thick)
        _draw_circle(canvas, (lcx, v_up), max(5, thick + 1), (0, 220, 255))
        _draw_circle(canvas, (lcx, v_down), max(5, thick + 1), (0, 220, 255))
        texts.append((lcx + 6, (v_up + v_down) // 2, "0.5A", (255, 220, 0)))

        h_left = max(0, int(lcx - 0.65 * A))
        h_right = min(img_w - 1, int(lcx + 0.65 * A))
        _draw_line(canvas, (h_left, lcy), (h_right, lcy), (255, 150, 0), thick)
        _draw_circle(canvas, (h_left, lcy), max(5, thick + 1), (255, 150, 0))
        _draw_circle(canvas, (h_right, lcy), max(5, thick + 1), (255, 150, 0))
        texts.append(((h_left + h_right) // 2, lcy - 16, "0.65A", (0, 150, 255)))

    # === 文字框十字线 ===
    _add_crosshair(canvas, result_mut.get("_cn_bbox"), (255, 0, 0), (255, 120, 120),
                    A, thick, panel, "主文字(cn)")
    _add_crosshair(canvas, result_mut.get("_en_bbox"), (220, 220, 0), (220, 220, 120),
                    A, thick, panel, "英文品牌标识")
    _add_crosshair(canvas, result_mut.get("_slogan_cn_bbox"), (220, 0, 220), (255, 160, 255),
                    A, thick, panel, "中文口号")
    _add_crosshair(canvas, result_mut.get("_slogan_en_bbox"), (180, 80, 180), (255, 200, 255),
                    A, thick, panel, "英文口号")
    _add_crosshair(canvas, result_mut.get("_subsidiary_bbox"), (60, 200, 60), (150, 255, 150),
                    A, thick, panel, "子公司")
    _add_crosshair(canvas, result_mut.get("_sub_left_bbox"), (100, 150, 220), (150, 200, 255),
                    A, thick, panel, "子-左框")
    _add_crosshair(canvas, result_mut.get("_sub_right_bbox"), (100, 150, 220), (150, 200, 255),
                    A, thick, panel, "子-右框")
    _add_crosshair(canvas, result_mut.get("_sub_upper_bbox"), (60, 200, 60), (150, 255, 180),
                    A, thick, panel, "子-上框")
    _add_crosshair(canvas, result_mut.get("_sub_lower_bbox"), (40, 180, 40), (130, 230, 130),
                    A, thick, panel, "子-下框")

    texts = _dedup_texts(texts, font_size * 2)
    info = {
        "text": f"{result_mut.get('rule_id','')}-{result_mut.get('layout_type','')}",
        "fill": (0, 230, 0) if result.get("passed") else (255, 40, 40),
        "passed": result.get("passed"),
    }
    _render_panel_window(canvas, img_w, img_h, texts, panel, info, font, font_sm,
                          f"中心参考 - {result.get('file', '')}", A)


def _add_crosshair(canvas, bbox, box_color, line_color, A, thick, panel, label):
    if bbox is None:
        return
    x1, y1, x2, y2 = bbox
    _draw_rect(canvas, bbox, box_color, thick)
    panel.add_box(label, box_color, bbox, A)

    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    _draw_line(canvas, (cx, y1), (cx, y2), line_color, max(1, thick - 1))
    _draw_circle(canvas, (cx, y1), max(4, thick), line_color)
    _draw_circle(canvas, (cx, y2), max(4, thick), line_color)
    _draw_line(canvas, (x1, cy), (x2, cy), line_color, max(1, thick - 1))
    _draw_circle(canvas, (x1, cy), max(4, thick), line_color)
    _draw_circle(canvas, (x2, cy), max(4, thick), line_color)


def _dedup_texts(texts, min_dist=30):
    texts.sort(key=lambda t: (t[1], t[0]))
    kept = []
    for t in texts:
        too_close = any(abs(t[0] - k[0]) < min_dist and abs(t[1] - k[1]) < min_dist for k in kept)
        if not too_close:
            kept.append(t)
    return kept


# ════════════════════════════════════════════════════════════
#  间距参考窗口
# ════════════════════════════════════════════════════════════

def _build_gaps_result(result: dict):
    """构建间距参考画布，返回 (_render_panel_window 所需参数)"""
    img = result.get("_img")
    if img is None:
        return None

    result_mut = dict(result)
    rule_id = result_mut.get("rule_id", "")
    layout = result_mut.get("layout_type", "")
    is_horiz = (layout == "横式")

    img_h, img_w = img.shape[:2]
    logo_bbox = result_mut.get("_logo_bbox")
    cn_bbox   = result_mut.get("_cn_bbox")
    en_bbox   = result_mut.get("_en_bbox")
    A = result_mut.get("A", 1)

    base_dim = min(img_w, img_h)
    font_size = max(MIN_FONT_SIZE, int(base_dim / 45))
    font = _load_cjk_font(font_size)
    panel_font_size = max(MIN_FONT_SIZE, int(PANEL_WIDTH / 20))
    font_sm = _load_cjk_font(panel_font_size)
    thick = max(2, int(base_dim / 300))
    texts = []
    panel = _Panel(result_mut.get("violations"))

    if logo_bbox is not None and cn_bbox is not None:
        canvas = img.copy()
        lx1, ly1, lx2, ly2 = logo_bbox
        lcx = (lx1 + lx2) // 2
        lcy = (ly1 + ly2) // 2

        _draw_rect(canvas, logo_bbox, (0, 255, 0), thick)
        panel.add_box("Logo", (0, 255, 0), logo_bbox, A)
        _draw_rect(canvas, cn_bbox, (255, 90, 90), max(2, thick // 2))
        panel.add_box("主文字(cn)", (255, 90, 90), cn_bbox, A)
        if rule_id not in ("R06", "R07") and en_bbox is not None:
            _draw_rect(canvas, en_bbox, (210, 210, 60), max(2, thick // 2))
            panel.add_box("英文品牌标识", (210, 210, 60), en_bbox, A)

        if is_horiz:
            h_right = max(0, int(lcx + 0.65 * A))
            text_lm = cn_bbox[0]
            if en_bbox is not None and en_bbox[0] < text_lm:
                text_lm = en_bbox[0]
            _draw_hgap(canvas, h_right, text_lm, lcy, A, (0, 230, 230), thick + 1, texts)
            panel.add_gap("主横向间距", (0, 230, 230), text_lm - h_right, A)
        else:
            v_down = min(img_h - 1, int(lcy + 0.5 * A))
            _draw_vgap(canvas, lcx, v_down, cn_bbox[1], A, (0, 230, 230), thick + 1, texts)
            panel.add_gap("主竖向间距", (0, 230, 230), cn_bbox[1] - v_down, A)

        if rule_id == "R02":
            if is_horiz:
                sub = result_mut.get("_subsidiary_bbox")
                if sub is not None:
                    _draw_rect(canvas, sub, (100, 230, 100), max(2, thick // 2))
                    panel.add_box("子公司", (100, 230, 100), sub, A)
                    ref_bot = en_bbox[3] if en_bbox is not None else cn_bbox[3]
                    _draw_vgap(canvas, (cn_bbox[0] + cn_bbox[2]) // 2, ref_bot, sub[1], A, (150, 255, 150), thick, texts)
                    panel.add_gap("en/cn→子公司", (150, 255, 150), sub[1] - ref_bot, A)
                    offset = abs(sub[0] - cn_bbox[0])
                    s_cy = (sub[1] + sub[3]) // 2
                    _draw_hgap(canvas, min(cn_bbox[0], sub[0]), max(cn_bbox[0], sub[0]), s_cy, A, (150, 255, 150), thick, texts)
                    panel.add_gap("子公司左偏移", (150, 255, 150), offset, A)
            elif layout == "竖式":
                sl = result_mut.get("_sub_left_bbox")
                sr = result_mut.get("_sub_right_bbox")
                ref_bot = en_bbox[3] if en_bbox is not None else cn_bbox[3]
                if sl is not None:
                    _draw_rect(canvas, sl, (100, 190, 255), max(2, thick // 2))
                    panel.add_box("子-左框", (100, 190, 255), sl, A)
                    _draw_vgap(canvas, (sl[0] + sl[2]) // 2, ref_bot, sl[1], A, (150, 210, 255), thick, texts)
                    panel.add_gap("en→子-左框", (150, 210, 255), sl[1] - ref_bot, A)
                if sr is not None:
                    _draw_rect(canvas, sr, (100, 190, 255), max(2, thick // 2))
                    panel.add_box("子-右框", (100, 190, 255), sr, A)
                    _draw_vgap(canvas, (sr[0] + sr[2]) // 2, ref_bot, sr[1], A, (150, 210, 255), thick, texts)
                    panel.add_gap("en→子-右框", (150, 210, 255), sr[1] - ref_bot, A)
                if sl is not None and sr is not None:
                    top_y = int((sl[1] + sr[1]) / 2)
                    _draw_hgap(canvas, sl[0], sr[2], top_y - 12, A, (150, 210, 255), thick, texts)
                    panel.add_gap("子公司总宽", (150, 210, 255), sr[2] - sl[0], A)
            else:
                su = result_mut.get("_sub_upper_bbox")
                sd = result_mut.get("_sub_lower_bbox")
                ref_bot = en_bbox[3] if en_bbox is not None else cn_bbox[3]
                if su is not None:
                    _draw_rect(canvas, su, (100, 230, 100), max(2, thick // 2))
                    panel.add_box("子-上框", (100, 230, 100), su, A)
                    _draw_vgap(canvas, (cn_bbox[0] + cn_bbox[2]) // 2, ref_bot, su[1], A, (150, 255, 150), thick, texts)
                    panel.add_gap("en→子-上框", (150, 255, 150), su[1] - ref_bot, A)
                if sd is not None:
                    _draw_rect(canvas, sd, (80, 210, 80), max(2, thick // 2))
                    panel.add_box("子-下框", (80, 210, 80), sd, A)
        elif rule_id in ("R04", "R05"):
            sc = result_mut.get("_slogan_cn_bbox")
            se = result_mut.get("_slogan_en_bbox")
            if sc is not None:
                _draw_rect(canvas, sc, (255, 90, 255), max(2, thick // 2))
                panel.add_box("中文口号", (255, 90, 255), sc, A)
                scy = (sc[1] + sc[3]) // 2
                _draw_vgap(canvas, lcx, lcy, scy, A, (255, 180, 0), thick, texts)
                panel.add_gap("Logo→口号中心", (255, 180, 0), abs(scy - lcy), A)
            if se is not None:
                _draw_rect(canvas, se, (255, 170, 255), max(2, thick // 2))
                panel.add_box("英文口号", (255, 170, 255), se, A)
                if sc is not None:
                    scy = (sc[1] + sc[3]) // 2
                    sey = (se[1] + se[3]) // 2
                    _draw_vgap(canvas, (sc[0] + sc[2]) // 2, scy, sey, A, (200, 130, 255), thick, texts)
                    panel.add_gap("中→英口号间距", (200, 130, 255), abs(sey - scy), A)

        texts = _dedup_texts(texts, font_size * 2)
    else:
        canvas = img.copy()
        texts = []

    info = {
        "text": f"{rule_id}-{layout}",
        "fill": (0, 230, 0) if result.get("passed") else (255, 40, 40),
        "passed": result.get("passed"),
    }
    return canvas, img_w, img_h, texts, panel, info, font, font_sm, A

def show_gaps_window(result: dict):
    data = _build_gaps_result(result)
    if data is None:
        return
    canvas, img_w, img_h, texts, panel, info, font, font_sm, A = data
    _render_panel_window(canvas, img_w, img_h, texts, panel, info, font, font_sm,
                          f"间距参考 - {result.get('file', '')}", A)


def save_gaps_image(result: dict, output_path: str):
    """保存间距参考图为 PNG 文件（含右侧信息面板）。
    如果总宽度超过 2400px，等比压缩至 2400px 以内，保证面板不被挤压。
    """
    MAX_SAVE_W = 2400
    MIN_WIN_H = 700
    data = _build_gaps_result(result)
    if data is None:
        return None
    canvas, img_w, img_h, texts, panel, info, font, font_sm, A = data

    total_w = img_w + PANEL_WIDTH
    total_h = max(img_h, MIN_WIN_H)

    # 超宽缩放
    if total_w > MAX_SAVE_W:
        scale = MAX_SAVE_W / total_w
        canvas = cv2.resize(canvas,
                            (int(img_w * scale), int(img_h * scale)),
                            interpolation=cv2.INTER_AREA)
        img_w = int(img_w * scale)
        img_h = int(img_h * scale)
        total_w = int(total_w * scale)
        total_h = max(int(total_h * scale), MIN_WIN_H)
        font = _load_cjk_font(max(MIN_FONT_SIZE, int(font.size * scale)))
        font_sm = _load_cjk_font(max(MIN_FONT_SIZE, int(font_sm.size * scale)))
        # 缩放 texts 坐标
        texts = [(int(x * scale), int(y * scale), t, c) for x, y, t, c in texts]
        A = A * scale

    out = np.full((total_h, total_w, 3), 255, dtype=np.uint8)
    y_off = max(0, (total_h - img_h) // 2)
    out[y_off:y_off + img_h, :img_w] = canvas
    cv2.line(out, (img_w, 0), (img_w, total_h - 1), (200, 200, 200), 1, cv2.LINE_AA)

    pil_img = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    draw.text((10, 10), info["text"], fill=info.get("fill", (255, 255, 255)), font=font)
    for x, y, content, color in texts:
        if x < img_w:
            draw.text((x, y + y_off), content, fill=color, font=font)
    status = "PASS" if not panel.violations else "FAIL"
    sc = (0, 180, 0) if not panel.violations else (200, 40, 40)
    draw.text((img_w + 10, 10), f"A = {A:.0f} px  {status}", fill=sc, font=font)
    panel.render(draw, img_w, 36, font, font_sm, total_h)

    out = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    cv2.imencode('.png', out)[1].tofile(output_path)
    return output_path


# ── 间距线 ──

def _draw_vgap(canvas, x, y_top, y_bot, A, color, thick, texts):
    y1, y2 = int(y_top), int(y_bot)
    if y2 <= y1:
        return
    _draw_line(canvas, (x, y1), (x, y2), color, thick)
    r = max(4, thick)
    _draw_circle(canvas, (x, y1), r, color)
    _draw_circle(canvas, (x, y2), r, color)
    texts.append((x + 8, (y1 + y2) // 2, f"{(y2-y1)/A:.2f}A", (color[2], color[1], color[0])))


def _draw_hgap(canvas, x_left, x_right, y, A, color, thick, texts):
    x1, x2 = int(x_left), int(x_right)
    if x2 <= x1:
        return
    _draw_line(canvas, (x1, y), (x2, y), color, thick)
    r = max(4, thick)
    _draw_circle(canvas, (x1, y), r, color)
    _draw_circle(canvas, (x2, y), r, color)
    texts.append(((x1 + x2) // 2, y - 18, f"{(x2-x1)/A:.2f}A", (color[2], color[1], color[0])))


# ════════════════════════════════════════════════════════════

def show_windows(result: dict):
    show_centers_window(result)
    show_gaps_window(result)
