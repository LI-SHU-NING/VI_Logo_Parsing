# -*- coding: utf-8 -*-
"""
品牌标识合规检测核心逻辑。

提供:
    - check_compliance(): 基于已有 pipeline 结果执行合规规则评估
    - 注解管理: load_annotations / save_annotation / get_annotation
    - 报告输出: print_report()
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent

ANNOTATIONS_PATH = BASE_DIR / "Brand_Resource_Libraries" / "image_annotations.json"
TEMPLATE_PATH = BASE_DIR / "Brand_Resource_Libraries" / "Logo_Compliance_Rule_Template.json"

RATIO_MAP = {
    ("R01", "横式"):   ("x", 4.02),
    ("R01", "中置式"): ("y", 0.98),
    ("R02", "横式"):   ("x", 2.95),
    ("R02", "竖式"):   ("y", 0.752),
    ("R02", "中置式"): ("y", 0.752),
    ("R03", "横式"):   ("x", 2.95),
    ("R03", "竖式"):   ("y", 3.25),
    ("R03", "中置式"): ("y", 0.752),
    ("R04", "横式"):   ("x", 2.95),
    ("R05", "横式"):   ("x", 2.95),
    ("R06", "横式"):   ("x", 2.18),
    ("R06", "中置式"): ("y", 0.865),
    ("R07", "横式"):   ("x", 7.21),
    ("R07", "竖式"):   ("y", 3.85),
}


# ── 注解管理 ──────────────────────────────────────────────

def load_annotations() -> dict:
    if ANNOTATIONS_PATH.exists():
        with open(ANNOTATIONS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"images": {}}


def save_annotation(filename: str, rule_id: str, layout_type: str, brand_type: str = ""):
    ann = load_annotations()
    ann["images"][filename] = {
        "rule_id": rule_id,
        "layout_type": layout_type,
        "brand_type": brand_type,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    ANNOTATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ANNOTATIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(ann, f, ensure_ascii=False, indent=2)
    print(f"  注解已保存: {filename} -> {rule_id}-{layout_type}")


def get_annotation(filename: str) -> Optional[dict]:
    ann = load_annotations()
    return ann.get("images", {}).get(filename)


# ── 内部辅助函数 ──────────────────────────────────────────

def _poly_to_bbox(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return (min(xs), min(ys), max(xs), max(ys))


# 品牌英文关键词（用于识别主品牌英文文字框）
_BRAND_EN = {"CHINA", "SOUTHERN", "POWER", "GRID"}
# 子公司英文关键词（用于排除污染 — 命中这些词的框不参与品牌英文合并）
_SUBSIDIARY_EN = {
    "CO.", "LTD", "FINANCE", "ENERGY", "SUPPLY", "STORAGE",
    "CO,", "COMPANY", "LIMITED", "CORP", "LTD.",
}


def _bbox_near(a, b, tol=5):
    return (abs(a[0] - b[0]) < tol and abs(a[1] - b[1]) < tol
            and abs(a[2] - b[2]) < tol and abs(a[3] - b[3]) < tol)


def _is_contained(inner, outer, tol=3):
    """检查 inner 框是否被 outer 框包含（带容差）"""
    return (inner[0] >= outer[0] - tol and inner[1] >= outer[1] - tol and
            inner[2] <= outer[2] + tol and inner[3] <= outer[3] + tol)


def _find_bbox(filtered_tuples, keyword):
    for tup in filtered_tuples:
        text = tup[0].replace(" ", "")
        if keyword == "中国南方电网":
            if text == keyword:
                return _poly_to_bbox(tup[1])
        elif keyword in text:
            return _poly_to_bbox(tup[1])
    return None


def _find_en_bbox(filtered_tuples, cn_bbox=None):
    """查找品牌英文文字框（CHINA SOUTHERN POWER GRID 系列）。

    收集所有包含品牌英文关键词的元组，但仅限在 cn_bbox 附近的（垂直距离
    不超过 cn_bbox 高度的 3 倍），避免将远处子公司英文误并入品牌区域。
    """
    ref_y_center = (cn_bbox[1] + cn_bbox[3]) / 2 if cn_bbox else None
    cn_h = (cn_bbox[3] - cn_bbox[1]) if cn_bbox else 9999

    brand_tuples = []
    for tup in filtered_tuples:
        text_upper = tup[0].upper()
        if not any(kw in text_upper for kw in _BRAND_EN):
            continue
        # 空间约束：必须在 cn_bbox 附近
        if ref_y_center is not None:
            b = _poly_to_bbox(tup[1])
            b_cy = (b[1] + b[3]) / 2
            if abs(b_cy - ref_y_center) > cn_h * 3:
                continue
        brand_tuples.append(tup)

    if not brand_tuples:
        return None

    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')
    for tup in brand_tuples:
        b = _poly_to_bbox(tup[1])
        min_x, min_y = min(min_x, b[0]), min(min_y, b[1])
        max_x, max_y = max(max_x, b[2]), max(max_y, b[3])
    return (min_x, min_y, max_x, max_y)


def _find_en_slogan(filtered_tuples):
    for tup in filtered_tuples:
        text_upper = tup[0].upper()
        if "LIGHT UP" in text_upper or "WARM UP" in text_upper:
            return _poly_to_bbox(tup[1])
    return None


def _find_subsidiary_text(filtered_tuples, cn_bbox=None, en_bbox=None):
    """查找子公司文字框（横式布局 — 品牌文字下方或右侧的独立文本）。
    候选框须在品牌区域下方（>= ref_bottom-5）或右侧（>= ref_right-5）。
    """
    # 确定品牌文字区域的下边界和右边界
    ref_bottom = 0
    ref_right = 0
    if en_bbox is not None:
        ref_bottom = en_bbox[3]
        ref_right = en_bbox[2]
    if cn_bbox is not None:
        ref_bottom = max(ref_bottom, cn_bbox[3])
        ref_right = max(ref_right, cn_bbox[2])

    candidates = []
    for tup in filtered_tuples:
        bbox = _poly_to_bbox(tup[1])
        if cn_bbox and _bbox_near(bbox, cn_bbox):
            continue
        if en_bbox and _is_contained(bbox, en_bbox):
            continue
        text_upper = tup[0].upper()
        if "万家灯火" in tup[0] or "LIGHT UP" in text_upper:
            continue
        # 必须在品牌文字下方或右侧
        is_below = bbox[1] >= ref_bottom - 5
        is_right = bbox[0] >= ref_right - 5
        if not is_below and not is_right:
            continue
        candidates.append(bbox)

    if not candidates:
        return None

    # 综合排序：y1 最大（最下方）优先，面积大者优先
    candidates.sort(key=lambda b: (b[1], -(b[2] - b[0]) * (b[3] - b[1])))
    return candidates[-1]


def _find_subsidiary_pair(filtered_tuples, cn_bbox=None, en_bbox=None):
    candidates = []
    for tup in filtered_tuples:
        bbox = _poly_to_bbox(tup[1])
        if cn_bbox and _bbox_near(bbox, cn_bbox):
            continue
        if en_bbox and _is_contained(bbox, en_bbox):
            continue
        text_upper = tup[0].upper()
        if "万家灯火" in tup[0] or "LIGHT UP" in text_upper:
            continue
        candidates.append((bbox, tup[0]))
    if len(candidates) < 2:
        return None, None, False
    candidates.sort(key=lambda x: x[0][0])
    left_text = candidates[0][1]
    left_is_en = all(ord(c) < 128 for c in left_text)
    return candidates[0][0], candidates[-1][0], left_is_en


def _find_subsidiary_stack(filtered_tuples, cn_bbox=None, en_bbox=None):
    candidates = []
    for tup in filtered_tuples:
        bbox = _poly_to_bbox(tup[1])
        if cn_bbox and _bbox_near(bbox, cn_bbox):
            continue
        if en_bbox and _is_contained(bbox, en_bbox):
            continue
        text_upper = tup[0].upper()
        if "万家灯火" in tup[0] or "LIGHT UP" in text_upper:
            continue
        candidates.append((bbox, tup[0]))
    if not candidates:
        return None, None, False
    candidates.sort(key=lambda x: x[0][1])
    has_en = any(all(ord(c) < 128 for c in text) for _, text in candidates)
    if len(candidates) == 1:
        return candidates[0][0], None, has_en
    return candidates[0][0], candidates[-1][0], has_en


# ── 核心合规检测 ──────────────────────────────────────────

def check_compliance(
    filtered_tuples: list,
    best_logo: tuple,
    layout_type: str,
    rule_id: str,
    brand_type: str = "",
    img_bgr=None,
    template_path: Optional[str] = None,
    all_tuples: list = None,
) -> dict:
    """
    对已检测出的品牌标识元素执行合规规则评估。

    参数:
        filtered_tuples: 过滤后的文字元组 [(text, poly, orientation), ...]
        best_logo:       最佳 logo 元组 (label, (x1,y1,x2,y2), conf)
        layout_type:     布局类型 ("横式" / "竖式" / "中置式")
        rule_id:         品牌规则ID (R01~R07)
        brand_type:      品牌类型名称（可选，用于报告）
        img_bgr:         原始 BGR 图像（可选，用于可视化）
        template_path:   规则模板 JSON 路径（可选）

    返回:
        dict: {
            "file": str,
            "brand_type": str,
            "rule_id": str,
            "layout_type": str,
            "A": float,
            "violations": list,
            "passed": bool,
            "_img": np.ndarray | None,
            "_logo_bbox": tuple,
            "_cn_bbox": tuple,
            "_en_bbox": tuple | None,
            "_slogan_cn_bbox": tuple | None,
            "_slogan_en_bbox": tuple | None,
            "_subsidiary_bbox": tuple | None,
            "_sub_left_bbox": tuple | None,
            "_sub_right_bbox": tuple | None,
            "_sub_upper_bbox": tuple | None,
            "_sub_lower_bbox": tuple | None,
            "_eval_env": dict,
        }
    """
    logo_bbox = best_logo[1]

    # 选择主文字
    if rule_id == "R06":
        cn_bbox = _find_bbox(filtered_tuples, "CSG")
        cn_label = "CSG"
    elif rule_id == "R07":
        # 英文全称可能被 OCR 拆分为多个单词框，用合并逻辑
        cn_bbox = _find_en_bbox(filtered_tuples)
        cn_label = "英文全称"
    else:
        cn_bbox = _find_bbox(filtered_tuples, "中国南方电网")
        cn_label = "中国南方电网"

    if cn_bbox is None:
        return {
            "file": "",
            "brand_type": brand_type,
            "rule_id": rule_id,
            "layout_type": layout_type,
            "error": f"未找到'{cn_label}'文字框",
        }

    en_bbox = _find_en_bbox(filtered_tuples, cn_bbox) if rule_id not in ("R06", "R07") else None

    # 计算基准距离 A
    key = (rule_id, layout_type)
    if key not in RATIO_MAP:
        return {
            "file": "",
            "brand_type": brand_type,
            "rule_id": rule_id,
            "layout_type": layout_type,
            "error": f"未定义 {rule_id}-{layout_type} 的基准比值",
        }
    axis, ratio = RATIO_MAP[key]
    if axis == "y":
        distance = abs((cn_bbox[1] + cn_bbox[3]) / 2 - (logo_bbox[1] + logo_bbox[3]) / 2)
    else:
        distance = abs((cn_bbox[0] + cn_bbox[2]) / 2 - (logo_bbox[0] + logo_bbox[2]) / 2)
    A = distance / ratio

    # 构建 eval 环境
    logo_top = logo_bbox[1]
    logo_bottom_raw = logo_bbox[3]
    logo_height = logo_bbox[3] - logo_bbox[1]
    logo_width = logo_bbox[2] - logo_bbox[0]

    if layout_type == "横式":
        logo_right_eval = (logo_bbox[0] + logo_bbox[2]) / 2 + 0.65 * A
        logo_bottom_eval = logo_bbox[3]
    else:
        logo_right_eval = logo_bbox[2]
        logo_bottom_eval = (logo_bbox[1] + logo_bbox[3]) / 2 + 0.5 * A

    cn_text_width = cn_bbox[2] - cn_bbox[0]
    cn_text_top = cn_bbox[1]
    cn_text_bottom = cn_bbox[3]
    cn_text_height = cn_bbox[3] - cn_bbox[1]
    text_leftmost = cn_bbox[0]
    text_rightmost = cn_bbox[2]
    if en_bbox is not None:
        text_leftmost = min(text_leftmost, en_bbox[0])
        text_rightmost = max(text_rightmost, en_bbox[2])
    text_width_all = text_rightmost - text_leftmost

    slogan_cn = _find_bbox(filtered_tuples, "万家灯火")
    slogan_en = _find_en_slogan(filtered_tuples)
    subsidiary = _find_subsidiary_text(filtered_tuples, cn_bbox, en_bbox)
    sub_left, sub_right, sub_left_is_en = _find_subsidiary_pair(filtered_tuples, cn_bbox, en_bbox)
    sub_upper, sub_lower, sub_has_en = _find_subsidiary_stack(filtered_tuples, cn_bbox, en_bbox)

    eval_env = {
        "A": A,
        "logo_right": logo_right_eval,
        "logo_top": logo_top,
        "logo_bottom": logo_bottom_eval,
        "logo_height": logo_height,
        "logo_width": logo_width,
        "logo_center_y": (logo_top + logo_bottom_raw) / 2,
        "text_leftmost": text_leftmost,
        "text_rightmost": text_rightmost,
        "text_width_all": text_width_all,
        "cn_text_width": cn_text_width,
        "cn_text_top": cn_text_top,
        "cn_text_bottom": cn_text_bottom,
        "cn_text_height": cn_text_height,
        "cn_text_left": cn_bbox[0],
        "slogan_cn_center_y": ((slogan_cn[1] + slogan_cn[3]) / 2) if slogan_cn else 0,
        "slogan_cn_width": (slogan_cn[2] - slogan_cn[0]) if slogan_cn else 0,
        "slogan_cn_height": (slogan_cn[3] - slogan_cn[1]) if slogan_cn else 0,
        "slogan_en_center_y": ((slogan_en[1] + slogan_en[3]) / 2) if slogan_en else 0,
        "subsidiary_top": subsidiary[1] if subsidiary else 0,
        "subsidiary_height": (subsidiary[3] - subsidiary[1]) if subsidiary else 0,
        "subsidiary_left": subsidiary[0] if subsidiary else 0,
        "en_text_bottom": en_bbox[3] if en_bbox else 0,
        "en_text_center_y": ((en_bbox[1] + en_bbox[3]) / 2) if en_bbox else 0,
        "subsidiary_center_y": ((subsidiary[1] + subsidiary[3]) / 2) if subsidiary else 0,
        "sub_left_width": (sub_left[2] - sub_left[0]) if sub_left else 0,
        "sub_right_width": (sub_right[2] - sub_right[0]) if sub_right else 0,
        "sub_left_top": sub_left[1] if sub_left else 0,
        "sub_right_top": sub_right[1] if sub_right else 0,
        "sub_pair_range": (sub_right[2] - sub_left[0]) if sub_left and sub_right else 0,
        "sub_left_is_en": 1 if sub_left_is_en else 0,
        "sub_upper_top": sub_upper[1] if sub_upper else 0,
        "sub_upper_height": (sub_upper[3] - sub_upper[1]) if sub_upper else 0,
        "sub_upper_width": (sub_upper[2] - sub_upper[0]) if sub_upper else 0,
        "sub_lower_height": (sub_lower[3] - sub_lower[1]) if sub_lower else 0,
        "sub_lower_width": (sub_lower[2] - sub_lower[0]) if sub_lower else 0,
        "sub_has_en": 1 if sub_has_en else 0,
        "sub_missing_lower": 1 if (sub_upper is not None and sub_lower is None) else 0,
    }

    # 加载规则模板
    tp = Path(template_path) if template_path else TEMPLATE_PATH
    if not tp.exists():
        return {"file": "", "error": "规则模板文件未找到"}
    with open(tp, "r", encoding="utf-8") as f:
        template = json.load(f)

    # 匹配规则
    matched_rule = None
    if rule_id == "R02" and layout_type == "中置式":
        variant_target = "中英文全称" if sub_has_en else "具体单位"
        for rule in template.get("rules", []):
            if rule.get("brand_rule_id") == rule_id and rule.get("layout_type") == layout_type \
                    and rule.get("variant", "") == variant_target:
                matched_rule = rule
                break
    if matched_rule is None:
        for rule in template.get("rules", []):
            if rule.get("brand_rule_id") == rule_id and rule.get("layout_type") == layout_type \
                    and "variant" not in rule:
                matched_rule = rule
                break

    if matched_rule is None:
        return {
            "file": "",
            "brand_type": brand_type,
            "rule_id": rule_id,
            "layout_type": layout_type,
            "A": A,
            "error": "未找到匹配规则",
        }

    # 逐条评估，跳过缺少元素的检查
    # 构建跳过列表：哪些变量因元素缺失不可用
    skip_vars = set()
    if subsidiary is None:
        skip_vars.update(["subsidiary_top","subsidiary_height","subsidiary_left",
                          "subsidiary_center_y"])
    if sub_left is None or sub_right is None:
        skip_vars.update(["sub_left_width","sub_right_width","sub_left_top",
                          "sub_right_top","sub_pair_range","sub_left_is_en"])
    if sub_upper is None and sub_lower is None:
        skip_vars.update(["sub_upper_top","sub_upper_height","sub_upper_width",
                          "sub_lower_height","sub_lower_width"])
    if sub_upper is None:
        skip_vars.update(["sub_upper_top","sub_upper_height","sub_upper_width"])
    if sub_lower is None:
        skip_vars.update(["sub_lower_height","sub_lower_width"])
    if slogan_cn is None:
        skip_vars.update(["slogan_cn_center_y","slogan_cn_width","slogan_cn_height"])
    if slogan_en is None:
        skip_vars.update(["slogan_en_center_y"])
    if en_bbox is None:
        skip_vars.update(["en_text_bottom", "en_text_center_y"])

    violations = []
    for check in matched_rule.get("checks", []):
        cond = check.get("condition", "")
        if any(v in cond for v in skip_vars):
            continue
        # 规范化逻辑运算符：JSON 中为大写 OR/AND，Python eval 需要小写
        cond = cond.replace(" OR ", " or ").replace(" AND ", " and ")
        try:
            violated = eval(cond, {"__builtins__": {}}, eval_env)
        except Exception:
            violated = False
        if violated:
            violations.append({
                "name": check["name"],
                "description": check.get("description", ""),
                "condition": check.get("condition", ""),
                "severity": check.get("severity", "warning"),
            })

    return {
        "file": "",
        "brand_type": brand_type,
        "rule_id": rule_id,
        "layout_type": layout_type,
        "A": round(A, 1),
        "violations": violations,
        "passed": len(violations) == 0,
        "_img": img_bgr,
        "_logo_bbox": logo_bbox,
        "_cn_bbox": cn_bbox,
        "_en_bbox": en_bbox,
        "_slogan_cn_bbox": slogan_cn,
        "_slogan_en_bbox": slogan_en,
        "_subsidiary_bbox": subsidiary,
        "_sub_left_bbox": sub_left,
        "_sub_right_bbox": sub_right,
        "_sub_upper_bbox": sub_upper,
        "_sub_lower_bbox": sub_lower,
        "_eval_env": eval_env,
        "_matched_checks": matched_rule.get("checks", []),
        "_all_text_boxes": [_poly_to_bbox(t[1]) for t in (all_tuples or [])],
    }


# ── 报告输出 ──────────────────────────────────────────────

def print_report(result: dict):
    """打印单张图片的合规报告"""
    if "error" in result:
        print(f"\n  X {result.get('file', '')}: {result['error']}")
        return

    status = "PASS" if result["passed"] else "FAIL"
    print(f"\n{'─' * 60}")
    print(f"  文件: {result.get('file', '')}")
    print(f"  类别: {result['brand_type']} ({result['rule_id']})")
    print(f"  布局: {result['layout_type']}")
    print(f"  基准距离 A = {result['A']:.0f}px")
    print(f"  结果: {status}")

    if result["violations"]:
        print(f"\n  违规项 ({len(result['violations'])}):")
        for v in result["violations"]:
            tag = "!!" if v["severity"] == "error" else " !"
            print(f"    {tag} [{v['severity']}] {v['name']}: {v['description']}")
    print(f"{'─' * 60}")
