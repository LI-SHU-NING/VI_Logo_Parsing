# -*- coding: utf-8 -*-
"""
Logo 字体验证模块。

功能：
    南网品牌标识字体规范校验（verify_brand_font）— 使用南网标准字体样例图校验：
    - 若图中有"中国南方电网"字样 → 对比中文字体标准样例图
    - 若图中有"CHINA SOUTHERN POWER GRID"字样 → 对比英文全称字体标准样例图
    - 若图中有"CSG"字样 → 对比英文简称字体标准样例图
    不识别具体字体类型，只判断是否符合南网字体规范，以 JSON 格式输出结果。

实现逻辑：
    南网标准字体样例图(Brand_Resource_Libraries/Fonts_Image/) + 用户图 →
    3 次 MLLM 调用（中文、英文全称、英文简称各一次）→
    对比字体是否一致（中文黑体、英文Arial Bold），检查文字残缺/模糊
    + 1 次 MLLM 调用检测图片中其他文字（非标准字样）的完整性
    JSON 结果（文字是否存在 + 文字是否完整 + 文字是否清晰 + 是否符合规范 + 标志位 + 置信度 + 详细说明）
    每次只对比一种字样，结果更聚焦可靠。
    默认串行执行，可通过 parallel=True 切换为并发执行以缩短总耗时。
    图片中无对应字样时标记为"不适用"；文字残缺或模糊时标记为"不是"（硬性不合格）。

    使用标准样例图对比比凭经验识别更准确，因为 MLLM 可以直接视觉比较
    用户图片与标准样例的笔画粗细、字形结构、衬线特征。
"""

import json
import sys
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from MLLM_LLM_Component_Tools import MLLMClient

# 南网品牌标准字体样例图目录
BRAND_FONT_IMAGE_DIR = _PROJECT_ROOT / "Brand_Resource_Libraries" / "Fonts_Image"

# 南网品牌标准字体样例图配置
# 每项：目标文字 + 样例图路径 + 样例描述
BRAND_FONT_TEMPLATES = {
    "chinese": {
        "text": "中国南方电网",
        "template_path": BRAND_FONT_IMAGE_DIR / "中国南方电网" / "中国南方电网.jpg",
        "description": "南网中文字体标准",
    },
    "english_full": {
        "text": "CHINA SOUTHERN POWER GRID",
        "template_path": BRAND_FONT_IMAGE_DIR / "CHINA_SOUTHERN_POWER_GRID" / "CHINA SOUTHERN POWER GRID.jpg",
        "description": "南网英文全称字体标准",
    },
    "csg": {
        "text": "CSG",
        "template_path": BRAND_FONT_IMAGE_DIR / "CSG" / "CSG.jpg",
        "description": "南网英文简称字体标准",
    },
}


class FontVerifier:
    """Logo 字体验证器。

    依赖：
        - MLLM_LLM_Component_Tools.MLLMClient（多模态大模型）
    """

    def __init__(self):
        self._mllm = MLLMClient()

    # ------------------------------------------------------------------
    #  南网品牌标识字体规范校验
    # ------------------------------------------------------------------

    def verify_brand_font(
        self,
        user_image_path: str,
        temperature: float = 0.3,
        parallel: bool = False,
    ) -> Dict[str, Any]:
        """南网品牌标识字体规范校验（3 次 MLLM 调用）。

        使用南网品牌标准字体样例图作为参考，通过 MLLM 视觉对比判断用户图片中
        的字体是否符合南网字体规范。不识别具体字体类型，只判断是否符合标准。

        3 项模板校验（中文、英文全称、英文简称各一次 MLLM 调用），
        每次只对比一种字样，结果更聚焦可靠。
        另有 1 项其他文字完整性检测（无模板，只查残缺/模糊）。
        默认串行执行（parallel=False），可通过 parallel=True 切换为并发执行。

        校验项（每项独立判断，图片中无对应字样时标记为"不适用"）：
            1. 中文"中国南方电网" → 对比中文字体标准样例图
            2. 英文"CHINA SOUTHERN POWER GRID" → 对比英文全称字体标准样例图
            3. 英文"CSG" → 对比英文简称字体标准样例图
            4. 其他文字 → 检测图片中非标准字样文字（无论中英文）的完整性

        每项输出 JSON 格式结果：
            - text_present: 图片中是否包含该字样（true/false）
            - text_complete: 文字是否完整无残缺（true/false/null，字样不存在时为null）
            - text_clear: 文字是否清晰不模糊（true/false/null，字样不存在时为null）
            - conforms: 字体是否符合品牌标准（true/false/null，字样不存在时为null）
            - conforms_flag: "是"=符合 / "不是"=不符合（含文字残缺/模糊/字体不符）/ "不适用"=字样不存在
            - confidence: "高"/"中"/"低"/"不适用"
            - detail: 详细说明（残缺时需指明具体哪个字、何种残缺类型）

        Args:
            user_image_path: 用户图片路径
            temperature:     采样温度
            parallel:        是否并发执行（默认 False 串行，True 为并发）

        Returns:
            {
                "success": bool,
                "results": {
                    "chinese": {text, text_present, text_complete, text_clear, conforms, conforms_flag, confidence, detail},
                    "english_full": {...},
                    "csg": {...},
                    "other_text": {...},
                },
                "raw_responses": {"chinese": dict, "english_full": dict, "csg": dict},
                "error": str|None,
            }
        """
        user_path = Path(user_image_path)
        if not user_path.exists():
            return {
                "success": False,
                "error": f"用户图片不存在: {user_image_path}",
            }

        # 分离存在/缺失的样例图
        available = []
        missing = []
        for key, info in BRAND_FONT_TEMPLATES.items():
            if info["template_path"].exists():
                available.append((key, info))
            else:
                missing.append((key, info))

        # 缺失样例图直接标记不适用
        results: Dict[str, Any] = {}
        raw_responses: Dict[str, Any] = {}
        errors = []
        for key, info in missing:
            results[key] = {
                "text": info["text"],
                "text_present": None,
                "text_complete": None,
                "text_clear": None,
                "conforms": None,
                "conforms_flag": "不适用",
                "confidence": "不适用",
                "detail": f"品牌标准字体样例图不存在: {info['template_path']}",
            }
            raw_responses[key] = None
            errors.append(f"{key}: 样例图不存在 ({info['template_path']})")

        if not available:
            return {
                "success": False,
                "error": "所有品牌标准字体样例图均不存在",
                "results": results,
                "raw_responses": raw_responses,
            }

        # 额外检测项：图片中其他文字（非标准字样）的完整性，无模板
        standard_texts = [info["text"] for _, info in available]
        available.append(("other_text", {
            "text": "其他文字",
            "description": "非标准字样文字完整性检测",
        }))

        # 逐项校验：串行或并行
        def _check(item):
            key, info = item
            if "template_path" not in info:
                # 其他文字完整性检测（无模板，只查残缺/模糊）
                return key, self._verify_other_text(
                    user_image_path=str(user_path),
                    standard_texts=standard_texts,
                    temperature=temperature,
                )
            return key, self._verify_with_template(
                user_image_path=str(user_path),
                template_image_path=str(info["template_path"]),
                target_text=info["text"],
                template_description=info["description"],
                temperature=temperature,
            )

        if parallel:
            # 并发执行
            with ThreadPoolExecutor(max_workers=len(available)) as executor:
                futures = {executor.submit(_check, item): item[0] for item in available}
                for future in as_completed(futures):
                    key = futures[future]
                    try:
                        _, check_result = future.result()
                        results[key] = check_result.get("result")
                        raw_responses[key] = check_result.get("raw_response")
                        if not check_result.get("success"):
                            errors.append(f"{key}: {check_result.get('error')}")
                    except Exception as e:
                        results[key] = None
                        raw_responses[key] = None
                        errors.append(f"{key}: {str(e)}")
        else:
            # 串行执行
            for item in available:
                key, check_result = _check(item)
                results[key] = check_result.get("result")
                raw_responses[key] = check_result.get("raw_response")
                if not check_result.get("success"):
                    errors.append(f"{key}: {check_result.get('error')}")

        success = len(errors) == 0
        return {
            "success": success,
            "results": results,
            "raw_responses": raw_responses,
            "error": "; ".join(errors) if errors else None,
        }

    def _verify_with_template(
        self,
        user_image_path: str,
        template_image_path: str,
        target_text: str,
        template_description: str,
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        """用品牌标准字体样例图校验用户图片中的字体（内部方法）。

        将品牌标准字体样例图和用户图片一起送入 MLLM，对比字体是否一致
        （中文黑体、英文 Arial Bold），同时检查文字残缺/模糊。

        判断规则：文字残缺/模糊/字体不一致 → 不合格（conforms=false）。

        MLLM 以 JSON 格式返回结果，解析失败时用正则兜底。

        Args:
            user_image_path:    用户图片路径
            template_image_path: 品牌标准字体样例图路径
            target_text:        需要校验的目标文字
            template_description: 样例图描述（如"南网中文字体标准"）
            temperature:        采样温度

        Returns:
            {
                "success": bool,
                "result": {text, text_present, text_complete, text_clear, conforms, conforms_flag, confidence, detail},
                "raw_response": dict|None,
                "error": str|None,
            }
        """
        prompt = f"""第一张是品牌标准字体样例图（{template_description}），文字为"{target_text}"。
第二张是用户待校验图片。

请对比判断用户图片中"{target_text}"字样的字体是否与样例图一致（中文应为黑体，英文应为Arial Bold）。

同时检查文字是否有残缺（笔画缺失/断裂/截断）或模糊（无法辨认）。

规则：
- 文字不存在 → conforms_flag="不适用"
- 文字残缺或模糊 → conforms_flag="不是"
- 字体与样例不一致 → conforms_flag="不是"
- 字体与样例一致且文字完整清晰 → conforms_flag="是"

仅输出JSON：
{{"text_present": true或false, "text_complete": true或false或null, "text_clear": true或false或null, "conforms": true或false或null, "conforms_flag": "是"或"不是"或"不适用", "confidence": "高"或"中"或"低"或"不适用", "detail": "说明"}}"""

        try:
            result = self._mllm.chat(
                prompt=prompt,
                image_paths=[template_image_path, user_image_path],
                temperature=temperature,
                stream=False,
            )

            content = self._extract_content(result)
            parsed = self._parse_json_response(content, target_text)

            return {
                "success": True,
                "result": parsed,
                "raw_response": result,
                "error": None,
            }

        except Exception as e:
            return {
                "success": False,
                "result": None,
                "raw_response": None,
                "error": f"字体校验失败 ({target_text}): {str(e)}",
            }

    def _verify_other_text(
        self,
        user_image_path: str,
        standard_texts: list,
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        """检测图片中非标准字样文字的完整性（内部方法）。

        不对比字体类型（无模板），只检测图片中除了标准字样之外的其他文字
        （无论中文还是英文）是否完整无残缺、是否清晰不模糊。

        Args:
            user_image_path: 用户图片路径
            standard_texts:  标准字样文字列表（这些字样不纳入"其他文字"检测）
            temperature:     采样温度

        Returns:
            {
                "success": bool,
                "result": {text, text_present, text_complete, text_clear, conforms, conforms_flag, confidence, detail},
                "raw_response": dict|None,
                "error": str|None,
            }
        """
        standard_list = "、".join(f'"{t}"' for t in standard_texts)
        prompt = f"""请检测图片中除了{standard_list}之外的所有其他文字（无论中文还是英文），判断这些文字是否完整无残缺。

只检测文字完整性，不检测字体类型。检查是否有笔画缺失、断裂、截断或模糊（无法辨认）。

规则：
- 除标准字样外无其他文字 → conforms_flag="不适用"
- 其他文字有残缺或模糊 → conforms_flag="不是"
- 其他文字完整清晰 → conforms_flag="是"

仅输出JSON：
{{"text_present": true或false, "text_complete": true或false或null, "text_clear": true或false或null, "conforms": true或false或null, "conforms_flag": "是"或"不是"或"不适用", "confidence": "高"或"中"或"低"或"不适用", "detail": "说明"}}"""

        try:
            result = self._mllm.chat(
                prompt=prompt,
                image_paths=[user_image_path],
                temperature=temperature,
                stream=False,
            )

            content = self._extract_content(result)
            parsed = self._parse_json_response(content, "其他文字")

            return {
                "success": True,
                "result": parsed,
                "raw_response": result,
                "error": None,
            }

        except Exception as e:
            return {
                "success": False,
                "result": None,
                "raw_response": None,
                "error": f"其他文字完整性检测失败: {str(e)}",
            }

    @staticmethod
    def _parse_json_response(content: str, target_text: str) -> Dict[str, Any]:
        """从 MLLM 响应中解析 JSON 结果。

        优先尝试直接解析 JSON，失败时用正则提取各字段。

        Args:
            content:    MLLM 响应文本
            target_text: 目标文字（用于填充结果的 text 字段）

        Returns:
            {text, text_present, text_complete, text_clear, conforms, conforms_flag, confidence, detail}
        """
        # 尝试从内容中提取并解析 JSON
        parsed = None
        json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                parsed = None

        if isinstance(parsed, dict):
            return {
                "text": target_text,
                "text_present": parsed.get("text_present"),
                "text_complete": parsed.get("text_complete"),
                "text_clear": parsed.get("text_clear"),
                "conforms": parsed.get("conforms"),
                "conforms_flag": parsed.get("conforms_flag", "不适用"),
                "confidence": parsed.get("confidence", "不适用"),
                "detail": parsed.get("detail", ""),
            }

        # 正则兜底解析
        text_present = None
        text_complete = None
        text_clear = None
        conforms = None
        conforms_flag = "不适用"
        confidence = "不适用"
        detail = content

        # text_present
        tp_match = re.search(r'text_present\s*[：:]\s*(true|false)', content, re.IGNORECASE)
        if tp_match:
            text_present = tp_match.group(1).lower() == "true"

        # text_complete
        tc_match = re.search(r'text_complete\s*[：:]\s*(true|false|null)', content, re.IGNORECASE)
        if tc_match:
            val = tc_match.group(1).lower()
            text_complete = None if val == "null" else (val == "true")

        # text_clear
        tcl_match = re.search(r'text_clear\s*[：:]\s*(true|false|null)', content, re.IGNORECASE)
        if tcl_match:
            val = tcl_match.group(1).lower()
            text_clear = None if val == "null" else (val == "true")

        # conforms
        conf_match = re.search(r'conforms\s*[：:]\s*(true|false|null)', content, re.IGNORECASE)
        if conf_match:
            val = conf_match.group(1).lower()
            conforms = None if val == "null" else (val == "true")

        # conforms_flag
        flag_match = re.search(r'conforms_flag\s*[：:]\s*["\']?(是|不是|不适用)["\']?', content)
        if flag_match:
            conforms_flag = flag_match.group(1)

        # confidence
        conf_val_match = re.search(r'confidence\s*[：:]\s*["\']?(高|中|低|不适用)["\']?', content)
        if conf_val_match:
            confidence = conf_val_match.group(1)

        # detail
        detail_match = re.search(r'detail\s*[：:]\s*["\']?(.+?)["\']?\s*$', content, re.DOTALL)
        if detail_match:
            detail = detail_match.group(1).strip()

        return {
            "text": target_text,
            "text_present": text_present,
            "text_complete": text_complete,
            "text_clear": text_clear,
            "conforms": conforms,
            "conforms_flag": conforms_flag,
            "confidence": confidence,
            "detail": detail,
        }

    # ------------------------------------------------------------------
    #  内部方法
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_content(result: Dict[str, Any]) -> str:
        """从 MLLM 响应中提取文本内容（兼容 OpenAI 格式）。"""
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0].get("message", {}).get("content", "")
        return str(result)
