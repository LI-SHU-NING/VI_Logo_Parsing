# -*- coding: utf-8 -*-
"""
品牌标识 ROI 后处理

功能:
    1. 从多个 logo 检测结果中选出最佳 logo（置信率阈值过滤 + 面积/距离综合打分）
    2. 过滤出与 logo 邻近的关键文字元组
    3. 聚合 logo + 邻近文字得到最大 box，裁剪 ROI 图像
    4. ROI 内各元素坐标重算
    5. 判断布局类型（横式/竖式/中置式）
"""
import numpy as np


def _poly_to_bbox(poly):
    """多边形坐标 → 外接矩形 (x1, y1, x2, y2)"""
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return (min(xs), min(ys), max(xs), max(ys))


def _bbox_area(bbox):
    """矩形面积"""
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def _bbox_center(bbox):
    """矩形中心点"""
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _bbox_distance(bbox1, bbox2):
    """两个矩形之间的最小距离（边到边，重叠则为0）"""
    dx = max(0, max(bbox1[0], bbox2[0]) - min(bbox1[2], bbox2[2]))
    dy = max(0, max(bbox1[1], bbox2[1]) - min(bbox1[3], bbox2[3]))
    return (dx * dx + dy * dy) ** 0.5


def _select_best_logo(logo_detections, text_bboxes, conf_threshold=0.5,
                      area_weight=0.4, distance_weight=0.6):
    """
    从多个 logo 检测结果中选出最佳 logo。

    策略: 先过滤置信率低于阈值的，再在剩余中按 (面积得分 * area_weight + 距离得分 * distance_weight) 综合打分。

    参数:
        logo_detections: [(label, (x1,y1,x2,y2), conf), ...]
        text_bboxes:     [(x1,y1,x2,y2), ...] 关键文字的外接矩形列表
        conf_threshold:  置信率过滤阈值
        area_weight:     面积得分权重
        distance_weight: 距离得分权重

    返回:
        最佳 logo 元组 (label, (x1,y1,x2,y2), conf)，无合格 logo 时返回 None
    """
    if not logo_detections:
        return None

    # 过滤低置信率
    candidates = [logo for logo in logo_detections if logo[2] >= conf_threshold]
    if not candidates:
        # 全部低于阈值时降级取置信率最高的
        candidates = [max(logo_detections, key=lambda x: x[2])]
        return candidates[0]

    if len(candidates) == 1:
        return candidates[0]

    # 计算文字群中心
    if text_bboxes:
        all_x = [b[0] for b in text_bboxes] + [b[2] for b in text_bboxes]
        all_y = [b[1] for b in text_bboxes] + [b[3] for b in text_bboxes]
        text_center = (sum(all_x) / len(all_x), sum(all_y) / len(all_y))
    else:
        text_center = None

    # 归一化面积和距离
    areas = [_bbox_area(logo[1]) for logo in candidates]
    max_area = max(areas) if max(areas) > 0 else 1

    distances = []
    for logo in candidates:
        if text_center:
            logo_center = _bbox_center(logo[1])
            d = ((logo_center[0] - text_center[0]) ** 2 + (logo_center[1] - text_center[1]) ** 2) ** 0.5
            distances.append(d)
        else:
            distances.append(0)
    max_dist = max(distances) if max(distances) > 0 else 1

    # 综合打分：面积越大越好，距离越小越好
    best_score = -1
    best_logo = None
    for i, logo in enumerate(candidates):
        area_score = areas[i] / max_area
        dist_score = 1 - (distances[i] / max_dist)
        score = area_score * area_weight + dist_score * distance_weight
        if score > best_score:
            best_score = score
            best_logo = logo

    return best_logo


def _filter_nearby_text(text_tuples, logo_bbox, dist_factor=3.0):
    """
    过滤出与 logo 邻近的文字元组。

    判断依据: 文字框与 logo 框的距离 < logo 对角线长度 * dist_factor

    参数:
        text_tuples: [(text, poly, orientation), ...]
        logo_bbox:   (x1, y1, x2, y2)
        dist_factor: 距离阈值因子（相对于 logo 对角线长度）

    返回:
        邻近文字元组列表
    """
    if not text_tuples:
        return []

    logo_w = logo_bbox[2] - logo_bbox[0]
    logo_h = logo_bbox[3] - logo_bbox[1]
    logo_diag = (logo_w * logo_w + logo_h * logo_h) ** 0.5
    threshold = logo_diag * dist_factor

    nearby = []
    for tup in text_tuples:
        text_bbox = _poly_to_bbox(tup[1])
        d = _bbox_distance(text_bbox, logo_bbox)
        if d <= threshold:
            nearby.append(tup)

    return nearby


def _compute_max_box(logo_bbox, text_bboxes, margin_ratio=0.05):
    """
    计算 logo + 文字的外接最大 box，并加一定边距。

    参数:
        logo_bbox:    (x1, y1, x2, y2)
        text_bboxes:  [(x1,y1,x2,y2), ...]
        margin_ratio: 边距占 box 尺寸的比例

    返回:
        (x1, y1, x2, y2) 最大 box
    """
    all_x1 = [logo_bbox[0]]
    all_y1 = [logo_bbox[1]]
    all_x2 = [logo_bbox[2]]
    all_y2 = [logo_bbox[3]]

    for b in text_bboxes:
        all_x1.append(b[0])
        all_y1.append(b[1])
        all_x2.append(b[2])
        all_y2.append(b[3])

    x1, y1, x2, y2 = min(all_x1), min(all_y1), max(all_x2), max(all_y2)
    w, h = x2 - x1, y2 - y1
    mx, my = w * margin_ratio, h * margin_ratio

    return (int(x1 - mx), int(y1 - my), int(x2 + mx), int(y2 + my))


def _determine_layout(logo_bbox, text_bboxes, max_box, text_orientations=None):
    """
    判断布局类型：横式 / 竖式 / 中置式。

    判断规则:
        1. 无文字时按 max_box 宽高比：宽≥高→横式，否则→竖式
        2. 文字朝向存在 90° → 竖式
        3. logo 垂直中心在文字顶部上方（logo 在文字上面）→ 中置式
        4. 否则（logo 与文字并排）→ 横式
    """
    if not text_bboxes:
        w = max_box[2] - max_box[0]
        h = max_box[3] - max_box[1]
        return "横式" if w >= h else "竖式"

    has_90 = any(o == 90 for o in (text_orientations or []))
    if has_90:
        return "竖式"

    text_y1 = min(b[1] for b in text_bboxes)
    logo_cy = (logo_bbox[1] + logo_bbox[3]) / 2
    if logo_cy < text_y1:
        return "中置式"
    return "横式"


def process_logo_roi(text_tuples, logo_detections, img_bgr,
                     conf_threshold=0.5, dist_factor=3.0, margin_ratio=0.05,
                     area_weight=0.4, distance_weight=0.6):
    """
    品牌标识 ROI 后处理主函数。

    参数:
        text_tuples:       关键文字元组 [(text, poly, orientation), ...]
        logo_detections:   logo 检测结果 [(label, (x1,y1,x2,y2), conf), ...]
        img_bgr:           原始 BGR 图像 (h, w, 3)
        conf_threshold:    logo 置信率过滤阈值
        dist_factor:       邻近文字过滤的距离因子（相对 logo 对角线）
        margin_ratio:      最大 box 边距比例
        area_weight:       logo 选择的面大的权重
        distance_weight:   logo 选择的距离得分权重

    返回:
        dict: {
            "success": bool,
            "best_logo": tuple | None,       # (label, (x1,y1,x2,y2), conf)
            "max_box": tuple | None,         # (x1, y1, x2, y2) 原图坐标
            "roi_image": np.ndarray | None,  # 裁剪的 ROI 图像
            "roi_elements": list,            # ROI 内重算坐标的元素 [(text, (x1,y1,x2,y2), orientation)]
            "layout_type": str | None,       # "横式" | "竖式" | "中置式"
            "message": str,
        }
    """
    if img_bgr is None:
        return {"success": False, "message": "输入图像为空"}

    img_h, img_w = img_bgr.shape[:2]

    # 文字元组转 bbox
    text_bboxes = [_poly_to_bbox(t[1]) for t in text_tuples] if text_tuples else []

    # Step 1: 选择最佳 logo
    best_logo = _select_best_logo(logo_detections, text_bboxes,
                                  conf_threshold=conf_threshold,
                                  area_weight=area_weight,
                                  distance_weight=distance_weight)
    if best_logo is None:
        return {"success": False, "message": "未检测到 logo"}

    logo_bbox = best_logo[1]

    # Step 2: 过滤邻近文字
    nearby_text = _filter_nearby_text(text_tuples, logo_bbox, dist_factor=dist_factor)
    nearby_bboxes = [_poly_to_bbox(t[1]) for t in nearby_text]

    # Step 3: 计算最大 box
    max_box = _compute_max_box(logo_bbox, nearby_bboxes, margin_ratio=margin_ratio)

    # 裁剪到图像范围内
    max_box = (
        max(0, max_box[0]),
        max(0, max_box[1]),
        min(img_w, max_box[2]),
        min(img_h, max_box[3]),
    )

    # Step 4: 裁剪 ROI 图像
    roi_image = img_bgr[max_box[1]:max_box[3], max_box[0]:max_box[2]].copy()

    # Step 5: ROI 内坐标重算
    ox, oy = max_box[0], max_box[1]
    roi_elements = []
    for tup in nearby_text:
        text = tup[0]
        poly = tup[1]
        ori = tup[2]
        new_bbox = _poly_to_bbox(poly)
        new_bbox = (new_bbox[0] - ox, new_bbox[1] - oy,
                    new_bbox[2] - ox, new_bbox[3] - oy)
        roi_elements.append((text, new_bbox, ori))

    # logo 坐标也重算
    roi_logo_bbox = (logo_bbox[0] - ox, logo_bbox[1] - oy,
                     logo_bbox[2] - ox, logo_bbox[3] - oy)

    # Step 6: 判断布局类型
    text_orientations = [t[2] for t in nearby_text]
    layout_type = _determine_layout(logo_bbox, nearby_bboxes, max_box, text_orientations)

    return {
        "success": True,
        "best_logo": best_logo,
        "max_box": max_box,
        "roi_image": roi_image,
        "roi_elements": roi_elements,
        "roi_logo_bbox": roi_logo_bbox,
        "layout_type": layout_type,
        "message": "Success",
    }
