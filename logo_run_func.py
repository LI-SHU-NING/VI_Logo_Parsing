# -*- coding: utf-8 -*-
"""
品牌标识识别入口程序（主流程）

流程:
    0. 调用 blur_detection 进行图像模糊检测 → 不通过则终止流程（is_clear=False）
    1. 调用 OCR_Parsing 进行 OCR 定位识别 → 输出 (文本, 坐标, 朝向) 元组
    2. 调用 OCR_Tuple_Information_Filtering 过滤无关元组 → 保留品牌相关信息
    3. 调用 Yolov26_Logo_Object_Detection 进行 Logo 目标检测 → 输出 logo 元组
    4. 调用 Logo_ROI_Post_Processing 进行 ROI 后处理 → 聚合最大box + 裁剪ROI + 判断布局
    5. 调用 Brand_Logo_Norm_Classification 进行品牌标识类型识别
    6. 调用 Background_Color_Verify 进行背景颜色校验 → 背景掩码 + K-Means + MLLM分析
    7. 调用 Logo_Font_Verify 进行字体校验 → 并发对比中文/英文全称/英文简称字体样例图
    8. 调用 Logo_Compliance_Combination_Check 进行合规间距/比例检测 → 规则模板评估 + 可视化
    9. 输出完整结果（含 blur_detection 标志位 + 各模块耗时统计 + 日志路径）
"""

import sys
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from OCR_Parsing import run_ocr
from blur_detection import check_clarity, ClarityResult
from OCR_Tuple_Information_Filtering import filter_tuples
from Yolov26_Logo_Object_Detection.yolo26_detect import detect_logo
from Logo_ROI_Post_Processing import process_logo_roi, draw_roi_annotations
from Brand_Logo_Norm_Classification import classify_brand
from Background_Color_Verify import BackgroundVerifier
from Logo_Font_Verify import FontVerifier
from Logo_Compliance_Combination_Check import check_compliance, print_report, save_gaps_image

LOGS_DIR = BASE_DIR / "Logs"


def _write_run_log(logs_dir: Path, input_stem: str, timing: dict, result: dict) -> str:
    """将本次运行耗时与关键结果写入日志文件，返回日志文件路径。"""
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"{input_stem}_{timestamp}.log"

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("品牌标识识别运行日志\n")
        f.write(f"输入文件: {input_stem}\n")
        f.write(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"运行状态: {'成功' if result.get('success') else '失败'}\n")
        f.write(f"消息: {result.get('message', '')}\n")
        f.write("=" * 60 + "\n\n")

        f.write("【模块耗时统计】\n")
        for module, elapsed in timing.items():
            if elapsed is None:
                f.write(f"  {module:25s}: 跳过(未执行)\n")
            else:
                f.write(f"  {module:25s}: {elapsed:.3f} 秒\n")

        f.write("\n【关键结果摘要】\n")
        bd = result.get("blur_detection")
        if bd:
            f.write(f"  模糊检测: is_clear={bd.get('is_clear')}, "
                    f"combined_score={bd.get('combined_score')}\n")
        ocr = result.get("ocr_result")
        if ocr:
            f.write(f"  OCR 文本数: {len(ocr.get('text_tuples', []))}\n")
        f.write(f"  过滤后元组数: {len(result.get('filtered_tuples', []))}\n")
        f.write(f"  Logo 检测数: {len(result.get('logo_detections', []))}\n")
        roi = result.get("logo_roi")
        if roi and isinstance(roi, dict):
            f.write(f"  ROI 布局类型: {roi.get('layout_type')}\n")
        brand = result.get("brand_classification")
        if brand:
            f.write(f"  品牌类型: {brand.get('brand_type')} ({brand.get('rule_id')})\n")
        bg = result.get("background_verification")
        if bg:
            f.write(f"  背景校验: success={bg.get('success')}, "
                    f"logo版本={bg.get('logo_version')}, "
                    f"符合要求={bg.get('is_compliant')}, "
                    f"干净={bg.get('is_clean')}\n")
        cv = result.get("compliance_verification")
        if cv:
            if "error" in cv:
                f.write(f"  合规检测: 跳过 ({cv.get('error')})\n")
            else:
                f.write(f"  合规检测: passed={cv.get('passed')}, "
                        f"A={cv.get('A')}, "
                        f"违规项={len(cv.get('violations', []))}\n")
        fv = result.get("font_verification")
        if fv:
            f.write(f"  字体校验: success={fv.get('success')}\n")
            for k, v in (fv.get("results") or {}).items():
                if v:
                    f.write(f"    {k}: {v.get('conforms_flag')} "
                            f"(置信度={v.get('confidence')})\n")

    return str(log_path)


def run_logo_recognition(
    input_path: str,
    output_path: str,
    conf_thres=0.55,
    # === 模糊检测参数（暴露给外部调用方，默认值已经过测试验证，效果较好） ===
    # 调整说明：
    #   blur_check: 是否启用模糊检测。关闭(False)后跳过检测，恢复原流程行为。
    #       影响：关闭后 blur_detection 字段返回 None，流程不受清晰度约束。
    #   blur_threshold: 清晰度判定阈值（归一化模式下）。
    #       调大 → 判定更严格（更多图被判为模糊）；调小 → 更宽松。
    #       归一化模式下清晰图 combined 通常 > 1.5，模糊图 < 1.0。
    #   blur_laplacian_weight / blur_sobel_weight: 两种梯度特征的加权占比，和应为 1.0。
    #       Laplacian 对离焦模糊更敏感；Sobel(Tenengrad) 对边缘梯度更敏感。
    #   blur_normalize: 是否用图像纹理方差归一化。
    #       开启(True)可消除分辨率差异影响；关闭后 threshold 需切换到 50-500 尺度。
    #   整体影响：模糊检测不通过时，OCR/Logo 检测/ROI/分类流程全部跳过，直接返回，
    #       blur_detection.is_clear=False 作为标志位。
    blur_check: bool = True,
    blur_threshold: float = 1.0,
    blur_laplacian_weight: float = 0.5,
    blur_sobel_weight: float = 0.5,
    blur_normalize: bool = True,
) -> dict:
    """
    执行品牌标识识别的完整流程。

    参数:
        input_path:  输入文件的完整路径（含文件名）
        output_path: 输出目录的路径
        conf_thres:  Logo 检测置信率阈值
        blur_check:  是否启用图像模糊检测（默认 True）
        blur_threshold:         模糊检测清晰度阈值（默认 1.0，归一化模式）
        blur_laplacian_weight:  Laplacian 方差权重（默认 0.5）
        blur_sobel_weight:      Sobel 梯度方差权重（默认 0.5）
        blur_normalize:         是否纹理归一化（默认 True）

    返回:
        dict: {
            "success": bool,
            "blur_detection": dict | None,  # 模糊检测结果，含 is_clear 标志位
            "ocr_result": dict,             # OCR 原始结果
            "filtered_tuples": list,        # 过滤后的文字元组
            "logo_detections": list,        # Logo 检测结果
            "logo_roi": dict,               # ROI 后处理结果
            "brand_classification": dict,   # 品牌标识分类结果
            "background_verification": dict,# 背景颜色校验结果
            "font_verification": dict,      # 字体校验结果
            "compliance_verification": dict|None, # 合规间距/比例检测结果
            "timing": dict,                 # 各模块耗时统计（秒）
            "log_path": str,                # 日志文件路径
            "message": str,
        }

    blur_detection 字段结构:
        {
            "is_clear": bool,           # 标志位：True=清晰通过，False=模糊未通过
            "laplacian_var": float,     # Laplacian 方差原始值
            "sobel_var": float,         # Sobel 梯度方差原始值
            "combined_score": float,    # 综合清晰度得分
            "threshold": float,         # 使用的阈值
        }

    timing 字段结构（提前终止时未执行的模块不会出现）:
        {
            "image_load": float,            # 图像加载耗时
            "blur_detection": float|None,   # 模糊检测耗时（未启用则为 None）
            "ocr": float,                   # OCR 耗时
            "tuple_filtering": float,       # 元组过滤耗时
            "logo_detection": float,        # Logo 检测耗时
            "roi_processing": float,        # ROI 后处理耗时
            "roi_annotation": float|None,   # ROI 标注与保存耗时（无 ROI 则为 None）
            "brand_classification": float,  # 品牌分类耗时
            "background_verification": float|None, # 背景颜色校验耗时
            "font_verification": float|None,      # 字体校验耗时
            "compliance_check": float|None,       # 合规检测耗时
            "total": float,                 # 总耗时
        }
    """
    input_file = Path(input_path)
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    timing = {}
    overall_start = time.perf_counter()

    def _finalize(result: dict) -> dict:
        """汇总总耗时、写入日志、附加 timing 与 log_path 后返回。"""
        timing["total"] = time.perf_counter() - overall_start
        log_path = _write_run_log(LOGS_DIR, input_file.stem, timing, result)
        result["timing"] = dict(timing)
        result["log_path"] = log_path
        return result

    # Step 0: 图像加载 + 模糊检测（在 OCR 之前执行，不通过则终止整个流程）
    # 图像在此处加载一次，后续 Step 3 Logo 检测复用，避免重复读取
    t = time.perf_counter()
    img_bgr = cv2.imdecode(np.fromfile(str(input_file), dtype=np.uint8), cv2.IMREAD_COLOR)
    timing["image_load"] = time.perf_counter() - t

    blur_result = None
    if blur_check:
        if img_bgr is None:
            return _finalize({
                "success": False,
                "blur_detection": None,
                "ocr_result": None,
                "filtered_tuples": [],
                "logo_detections": [],
                "logo_roi": None,
                "brand_classification": None,
                "background_verification": None,
                "font_verification": None,
                "compliance_verification": None,
                "message": "无法读取图像文件，模糊检测无法执行",
            })

        t = time.perf_counter()
        blur_result = check_clarity(
            img_bgr,
            threshold=blur_threshold,
            laplacian_weight=blur_laplacian_weight,
            sobel_weight=blur_sobel_weight,
            normalize=blur_normalize,
        )
        timing["blur_detection"] = time.perf_counter() - t

        if not blur_result.is_clear:
            # 模糊检测未通过 → 终止流程，返回带 is_clear=False 标志位的结果
            return _finalize({
                "success": False,
                "blur_detection": {
                    "is_clear": False,
                    "laplacian_var": blur_result.laplacian_var,
                    "sobel_var": blur_result.sobel_var,
                    "combined_score": blur_result.combined_score,
                    "threshold": blur_result.threshold,
                },
                "ocr_result": None,
                "filtered_tuples": [],
                "logo_detections": [],
                "logo_roi": None,
                "brand_classification": None,
                "background_verification": None,
                "font_verification": None,
                "compliance_verification": None,
                "message": (
                    f"图像模糊检测未通过: combined_score={blur_result.combined_score:.2f} "
                    f"< threshold={blur_result.threshold:.2f}"
                ),
            })
    else:
        timing["blur_detection"] = None

    # Step 1: OCR 定位识别
    t = time.perf_counter()
    ocr_result = run_ocr(input_path, output_path)
    timing["ocr"] = time.perf_counter() - t

    if not ocr_result.get("success"):
        return _finalize({
            "success": False,
            "blur_detection": {
                "is_clear": True,
                "laplacian_var": blur_result.laplacian_var,
                "sobel_var": blur_result.sobel_var,
                "combined_score": blur_result.combined_score,
                "threshold": blur_result.threshold,
            } if blur_result else None,
            "ocr_result": ocr_result,
            "filtered_tuples": [],
            "logo_detections": [],
            "logo_roi": None,
            "brand_classification": None,
            "background_verification": None,
            "font_verification": None,
            "compliance_verification": None,
            "message": f"OCR 处理失败: {ocr_result.get('message', '')}",
        })

    text_tuples = ocr_result.get("text_tuples", [])

    # Step 2: 过滤无关元组，只保留品牌相关信息
    t = time.perf_counter()
    filtered_tuples = filter_tuples(text_tuples)
    timing["tuple_filtering"] = time.perf_counter() - t

    # Step 3: Logo 目标检测（img_bgr 已在 Step 0 加载，此处复用）
    t = time.perf_counter()
    logo_detections = detect_logo(img_bgr, conf_thres=conf_thres) if img_bgr is not None else []
    timing["logo_detection"] = time.perf_counter() - t

    # Step 4: 品牌标识 ROI 后处理 + 判断布局类型
    t = time.perf_counter()
    roi_result = process_logo_roi(filtered_tuples, logo_detections, img_bgr)
    timing["roi_processing"] = time.perf_counter() - t

    # 保存 ROI 图像 + 标注图 + 元组JSON
    if roi_result.get("success") and roi_result.get("roi_image") is not None:
        t = time.perf_counter()
        roi_dir = output_dir / input_file.stem
        roi_dir.mkdir(parents=True, exist_ok=True)
        stem = input_file.stem
        roi_image = roi_result["roi_image"]

        # 1. 原始 ROI 图
        roi_img_path = roi_dir / f"{stem}_roi.png"
        cv2.imencode('.png', roi_image)[1].tofile(str(roi_img_path))
        roi_result["roi_image_path"] = str(roi_img_path)

        # 2. 标注 ROI 图（框 + 文字标签，自适应缩放）
        roi_elements = roi_result.get("roi_elements", [])
        roi_logo_bbox = roi_result.get("roi_logo_bbox")
        annotated = draw_roi_annotations(roi_image, roi_elements, roi_logo_bbox)
        roi_annotated_path = roi_dir / f"{stem}_roi_annotated.png"
        cv2.imencode('.png', annotated)[1].tofile(str(roi_annotated_path))
        roi_result["roi_annotated_path"] = str(roi_annotated_path)

        # 3. ROI 重算坐标元组 JSON
        roi_json = {
            "max_box": roi_result.get("max_box"),
            "layout_type": roi_result.get("layout_type"),
            "best_logo": list(roi_result.get("best_logo", [])),
            "roi_logo_bbox": list(roi_logo_bbox) if roi_logo_bbox else None,
            "roi_elements": [
                {"text": t_text, "bbox": list(b), "orientation": o}
                for t_text, b, o in roi_elements
            ],
        }
        roi_json_path = roi_dir / f"{stem}_roi.json"
        with open(roi_json_path, "w", encoding="utf-8") as f:
            json.dump(roi_json, f, ensure_ascii=False, indent=2)
        roi_result["roi_json_path"] = str(roi_json_path)

        timing["roi_annotation"] = time.perf_counter() - t
    else:
        timing["roi_annotation"] = None

    # Step 5: 品牌标识类型识别
    t = time.perf_counter()
    brand_result = classify_brand(filtered_tuples)
    timing["brand_classification"] = time.perf_counter() - t

    # Step 6: 背景颜色校验
    background_result = None
    if roi_result.get("success") and roi_result.get("roi_image") is not None:
        t = time.perf_counter()
        bg_verifier = BackgroundVerifier()
        background_result = bg_verifier.verify_background(
            roi_image=roi_result["roi_image"],
            roi_elements=roi_result.get("roi_elements", []),
            roi_logo_bbox=roi_result.get("roi_logo_bbox"),
        )
        timing["background_verification"] = time.perf_counter() - t

        if background_result.get("success"):
            print(f"\n[背景校验] logo版本={background_result.get('logo_version')} "
                  f"符合要求={background_result.get('is_compliant')} "
                  f"干净={background_result.get('is_clean')}")
        else:
            print(f"\n[背景校验] 失败: {background_result.get('error')}")
    else:
        timing["background_verification"] = None

    # Step 7: 字体校验（使用 ROI 图像，并发对比中文/英文全称/英文简称字体样例图）
    font_result = None
    roi_img_path = roi_result.get("roi_image_path")
    if roi_result.get("success") and roi_img_path and Path(roi_img_path).exists():
        t = time.perf_counter()
        font_verifier = FontVerifier()
        font_result = font_verifier.verify_brand_font(user_image_path=roi_img_path)
        timing["font_verification"] = time.perf_counter() - t

        if font_result.get("success"):
            print("\n[字体校验]")
            for k, v in (font_result.get("results") or {}).items():
                if v:
                    print(f"  {k}: {v.get('conforms_flag')} (置信度={v.get('confidence')})")
        else:
            print(f"\n[字体校验] 失败: {font_result.get('error')}")
    else:
        timing["font_verification"] = None

    # Step 8: 合规间距/比例检测（品牌分类未识别到类型时跳过）
    compliance_result = None
    if (roi_result.get("success") and roi_result.get("best_logo")
            and roi_result.get("roi_logo_bbox")
            and brand_result.get("rule_id")):
        t = time.perf_counter()
        # 使用 ROI 局部坐标系：roi_elements 的 bbox 转为多边形格式供 _poly_to_bbox 使用
        roi_tuples = [
            (text, [(b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3])], orientation)
            for text, b, orientation in roi_result.get("roi_elements", [])
        ]
        best_logo_roi = (
            roi_result["best_logo"][0],
            roi_result["roi_logo_bbox"],
            roi_result["best_logo"][2],
        )
        compliance_result = check_compliance(
            filtered_tuples=roi_tuples,
            best_logo=best_logo_roi,
            layout_type=roi_result["layout_type"],
            rule_id=brand_result["rule_id"],
            brand_type=brand_result.get("brand_type", ""),
            img_bgr=roi_result["roi_image"],
        )
        compliance_result["file"] = input_file.name
        timing["compliance_check"] = time.perf_counter() - t

        print_report(compliance_result)

        # 保存合规间距可视化图（坐标已是 ROI 局部坐标系，无需转换）
        if "error" not in compliance_result:
            roi_dir = output_dir / input_file.stem
            roi_dir.mkdir(parents=True, exist_ok=True)
            compliance_img_path = str(roi_dir / f"{input_file.stem}_compliance.png")
            save_gaps_image(compliance_result, compliance_img_path)
            compliance_result["compliance_image_path"] = compliance_img_path
    else:
        timing["compliance_check"] = None
        if not brand_result.get("rule_id"):
            print("\n[合规检测] 跳过: 未识别到品牌类型 (rule_id=None)")

    # Step 9: 打印 ROI 内重算坐标
    if roi_result.get("success"):
        print("\n========== ROI 内坐标（重算后） ==========")
        print(f"布局类型: {roi_result.get('layout_type')}")
        print(f"最大box(原图坐标): {roi_result.get('max_box')}")
        logo_bbox = roi_result.get("roi_logo_bbox")
        if logo_bbox:
            print(f"Logo (ROI内坐标): {list(logo_bbox)}")
        print("文字元组 (ROI内坐标):")
        for t_text, b, o in roi_result.get("roi_elements", []):
            print(f"  [{o}°] \"{t_text}\" bbox={list(b)}")
        print("==========================================\n")

    # Step 10: 汇总输出
    return _finalize({
        "success": True,
        "blur_detection": {
            "is_clear": True,
            "laplacian_var": blur_result.laplacian_var,
            "sobel_var": blur_result.sobel_var,
            "combined_score": blur_result.combined_score,
            "threshold": blur_result.threshold,
        } if blur_result else None,
        "ocr_result": {
            "file_path": ocr_result.get("file_path"),
            "text_tuples": text_tuples,
        },
        "filtered_tuples": filtered_tuples,
        "logo_detections": logo_detections,
        "logo_roi": roi_result,
        "brand_classification": brand_result,
        "background_verification": background_result,
        "font_verification": font_result,
        "compliance_verification": compliance_result,
        "message": "Success",
    })


if __name__ == "__main__":
    result = run_logo_recognition(
        #input_path=str(BASE_DIR / "input" / """A1.10-2不同组合形式品牌标识(海外版)_3_竖式.png"),书宁整理的颜色测试图image\bad
        input_path=str(BASE_DIR / "input" / "书宁整理的颜色测试图image" / "bad" / "nanwan_pic_0kyt.jpeg"),
        output_path=str(BASE_DIR / "ocr"),
    )

    # 打印关键结果（不含图像数据）
    print_result = {k: v for k, v in result.items()}
    if print_result.get("logo_roi") and isinstance(print_result["logo_roi"], dict):
        roi = print_result["logo_roi"]
        print_result["logo_roi"] = {
            k: v for k, v in roi.items()
            if k != "roi_image"
        }
    print(json.dumps(print_result, ensure_ascii=False, indent=2, default=str))
