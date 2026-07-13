# -*- coding: utf-8 -*-
import os
import sys
import json
import logging
import traceback
from pathlib import Path

# OCR_Parsing 模块内的文件，BASE_DIR 指向项目根目录（parent.parent）
BASE_DIR = Path(__file__).resolve().parent.parent


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("WordParsing")

# 同模块相对导入（原为 import layout_parsing_batch_img_API）
from . import layout_parsing_batch_img_API



def extract_text_tuples(json_path: str) -> list[tuple]:
    """
    从 OCR 结果 JSON 文件中提取 (文本, 多边形坐标, 朝向角度) 元组列表。

    朝向角度由 rec_polys 的宽高比推算:
        高 > 宽 * 1.5 → 90°（竖排）
        宽 > 高 * 1.5 → 0°（横排）
        其他          → 0°（默认横排）

    参数:
        json_path: *_res.json 文件的完整路径

    返回:
        list[tuple]: [(text, rec_poly, orientation), ...]
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ocr_res = data.get("overall_ocr_res", {})
    rec_texts = ocr_res.get("rec_texts", [])
    rec_polys = ocr_res.get("rec_polys", [])

    results = []
    for i, text in enumerate(rec_texts):
        poly = rec_polys[i] if i < len(rec_polys) else []

        # 根据多边形坐标推算朝向
        if len(poly) == 4:
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            w = max(xs) - min(xs)
            h = max(ys) - min(ys)
            orientation = 90 if h > w * 1.5 else 0
        else:
            orientation = 0

        results.append((text, poly, orientation))

    return results


def _poly_bounds(poly):
    """计算多边形的外接矩形 [x_min, y_min, x_max, y_max]"""
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return [min(xs), min(ys), max(xs), max(ys)]


def _overlap_ratio_1d(a_min, a_max, b_min, b_max):
    """计算两个区间在某个轴上的重叠率（重叠长度 / 较短区间长度）"""
    overlap = max(0, min(a_max, b_max) - max(a_min, b_min))
    shorter = min(a_max - a_min, b_max - b_min)
    return overlap / shorter if shorter > 0 else 0


def _merge_polys(poly1, poly2):
    """合并两个多边形，取外接矩形对应的四点多边形"""
    b1 = _poly_bounds(poly1)
    b2 = _poly_bounds(poly2)
    x_min = min(b1[0], b2[0])
    y_min = min(b1[1], b2[1])
    x_max = max(b1[2], b2[2])
    y_max = max(b1[3], b2[3])
    return [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]


def _edges_aligned(bounds1, bounds2, ori, edge_threshold=0.08):
    """
    检查两个块的边缘是否对齐。

    横排(0°)：检查上y和下y是否接近（|y_min差| 和 |y_max差| 都小于平均高度 * edge_threshold）
    竖排(90°)：检查左x和右x是否接近（|x_min差| 和 |x_max差| 都小于平均宽度 * edge_threshold）

    参数:
        bounds1, bounds2: [x_min, y_min, x_max, y_max]
        ori: 0 或 90
        edge_threshold: 边缘差异阈值占平均尺寸的比例（默认 0.08）
    """
    if ori == 90:
        avg_width = ((bounds1[2] - bounds1[0]) + (bounds2[2] - bounds2[0])) / 2
        if avg_width == 0:
            return False
        left_diff = abs(bounds1[0] - bounds2[0])
        right_diff = abs(bounds1[2] - bounds2[2])
        return left_diff < avg_width * edge_threshold and right_diff < avg_width * edge_threshold
    else:
        avg_height = ((bounds1[3] - bounds1[1]) + (bounds2[3] - bounds2[1])) / 2
        if avg_height == 0:
            return False
        top_diff = abs(bounds1[1] - bounds2[1])
        bottom_diff = abs(bounds1[3] - bounds2[3])
        return top_diff < avg_height * edge_threshold and bottom_diff < avg_height * edge_threshold


def merge_text_tuples(text_tuples: list[tuple], overlap_threshold=0.9, gap_ratio=0.03, edge_threshold=0.08) -> list[tuple]:
    """
    合并同行/同列的文本元组。

    规则:
        90°（竖排）：
            - x 重叠率 > overlap_threshold → 同列候选
            - 左x和右x边缘对齐（差异 < 平均宽度 * edge_threshold）→ 同列确认
            - y 间距 < 框高度 * gap_ratio → 合并
        0°（横排）：
            - y 重叠率 > overlap_threshold → 同行候选
            - 上y和下y边缘对齐（差异 < 平均高度 * edge_threshold）→ 同行确认
            - x 间距 < 框宽度 * gap_ratio → 合并

    参数:
        text_tuples: extract_text_tuples 的输出
        overlap_threshold: 同列/同行的重叠率阈值（默认 0.9）
        gap_ratio: 间距阈值占框尺寸的比例（默认 0.03）
        edge_threshold: 边缘对齐阈值占平均尺寸的比例（默认 0.08）

    返回:
        list[tuple]: 合并后的 [(text, rec_poly, orientation), ...]
    """
    if not text_tuples:
        return []

    # 按 orientation 分组
    groups = {}
    for item in text_tuples:
        ori = item[2]
        groups.setdefault(ori, []).append(item)

    merged = []
    for ori, items in groups.items():
        # 先按列/行分组，再在组内合并
        lines = []  # 每个元素是一个"列"或"行"的列表
        for item in items:
            bounds = _poly_bounds(item[1])
            placed = False
            for line in lines:
                # 检查与该行/列中第一个元素的重叠率 + 边缘对齐
                first_bounds = _poly_bounds(line[0][1])
                if ori == 90:
                    overlap = _overlap_ratio_1d(bounds[0], bounds[2], first_bounds[0], first_bounds[2])
                else:
                    overlap = _overlap_ratio_1d(bounds[1], bounds[3], first_bounds[1], first_bounds[3])
                if overlap > overlap_threshold and _edges_aligned(bounds, first_bounds, ori, edge_threshold):
                    line.append(item)
                    placed = True
                    break
            if not placed:
                lines.append([item])

        # 在每个列/行内，按阅读顺序排序并贪心合并
        for line in lines:
            if ori == 90:
                line.sort(key=lambda t: _poly_bounds(t[1])[1])  # 按 y 升序
            else:
                line.sort(key=lambda t: _poly_bounds(t[1])[0])  # 按 x 升序

            result = [line[0]]
            for i in range(1, len(line)):
                prev = result[-1]
                curr = line[i]
                prev_bounds = _poly_bounds(prev[1])
                curr_bounds = _poly_bounds(curr[1])

                should_merge = False
                # 边缘对齐 + 间距检查
                if _edges_aligned(prev_bounds, curr_bounds, ori, edge_threshold):
                    if ori == 90:
                        gap = curr_bounds[1] - prev_bounds[3]  # curr.y_min - prev.y_max
                        ref_height = max(prev_bounds[3] - prev_bounds[1], curr_bounds[3] - curr_bounds[1])
                        if 0 <= gap < ref_height * gap_ratio:
                            should_merge = True
                    else:
                        gap = curr_bounds[0] - prev_bounds[2]  # curr.x_min - prev.x_max
                        ref_width = max(prev_bounds[2] - prev_bounds[0], curr_bounds[2] - curr_bounds[0])
                        if 0 <= gap < ref_width * gap_ratio:
                            should_merge = True

                if should_merge:
                    new_text = prev[0] + " " + curr[0]
                    new_poly = _merge_polys(prev[1], curr[1])
                    result[-1] = (new_text, new_poly, ori)
                else:
                    result.append(curr)

            merged.extend(result)

    return merged


def run_ocr(input_path: str, output_path: str) -> dict:
    """
    执行 OCR 文档解析，输出 (文本, 多边形坐标, 朝向角度) 元组。

    参数:
        input_path:  输入文件的完整路径（含文件名）
        output_path: 输出目录的路径

    返回:
        dict: {
            "success": bool,
            "file_path": str | None,
            "message": str,
            "text_tuples": list,   # OCR 文本元组
        }
    """
    input_file = Path(input_path)
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_name_without_ext = input_file.stem

    # 禁用 PIR，解决 PaddlePaddle 3.x + oneDNN 兼容性问题
    engine_config = {"enable_new_ir": False, "run_mode": "paddle"}

    try:
        logger.info(f"OCR 文本抽取开始: {input_file}")

        out = output_dir / file_name_without_ext
        layout_parsing_batch_img_API.process_images(input_file, out, engine_config=engine_config)

        # 二次整理：从 _res.json 提取 (文本, 多边形坐标, 朝向角度) 元组
        json_file = out / f"{file_name_without_ext}_res.json"
        text_tuples = extract_text_tuples(str(json_file))

        # 后处理：合并同列/同行的文本
        merged_tuples = merge_text_tuples(text_tuples)

        logger.info("OCR 处理成功完成")
        return {"success": True, "file_path": str(output_dir), "message": "Success", "text_tuples": merged_tuples}

    except Exception as e:
        logger.error(f"执行出错:\n{traceback.format_exc()}")
        return {"success": False, "file_path": None, "message": f"处理异常: {str(e)}"}


if __name__ == "__main__":
    result = run_ocr(
        input_path=str(BASE_DIR / "input" / "1.png"),
        output_path=str(BASE_DIR / "ocr"),
    )
    print(result)
