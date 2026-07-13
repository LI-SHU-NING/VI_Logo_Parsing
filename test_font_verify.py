# -*- coding: utf-8 -*-
"""Logo_Font_Verify 模块测试脚本。

用法：
    conda run -n python_11 python test_font_verify.py [图片路径]

不传图片路径时，默认使用 input 目录下的测试图片。
"""
import sys
import time
import json
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from Logo_Font_Verify import FontVerifier


def print_separator(title=""):
    print("\n" + "=" * 60)
    if title:
        print(f"  {title}")
        print("=" * 60)


def print_item_result(label, item):
    """打印单项校验结果。"""
    print(f"\n[{label}]")
    print(f"  字样是否存在:   {item.get('text_present')}")
    print(f"  是否符合规范:   {item.get('conforms')}")
    print(f"  标志位:         {item.get('conforms_flag')}")
    print(f"  置信度:         {item.get('confidence')}")
    detail = item.get('detail', '')
    if len(detail) > 150:
        detail = detail[:150] + "..."
    print(f"  详细说明:       {detail}")


def main():
    # 选择测试图片
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # 默认测试图片（横式品牌标识，含中文+英文全称）
        image_path = str(_PROJECT_ROOT / "input" / "错误字体3.png")

    image_file = Path(image_path)
    print_separator("Logo_Font_Verify 模块测试")
    print(f"测试图片: {image_file.name}")
    print(f"完整路径: {image_file}")
    print(f"文件大小: {image_file.stat().st_size} bytes")

    if not image_file.exists():
        print(f"\n[错误] 图片不存在: {image_path}")
        print(f"可用测试图片 (input 目录):")
        input_dir = _PROJECT_ROOT / "input"
        if input_dir.exists():
            for f in sorted(input_dir.iterdir()):
                if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp'):
                    print(f"  {f.name}")
        return

    # 初始化 FontVerifier
    print_separator("初始化")
    verifier = FontVerifier()
    print(f"可用方法: {[m for m in dir(verifier) if not m.startswith('_')]}")

    # === 测试 verify_brand_font（南网品牌标识字体规范校验）===
    print_separator("测试 verify_brand_font（3次MLLM并发调用）")
    print("校验项:")
    print("  1. 中文'中国南方电网' → 对比中文字体标准样例图")
    print("  2. 英文'CHINA SOUTHERN POWER GRID' → 对比英文全称字体标准样例图")
    print("  3. 英文'CSG' → 对比英文简称字体标准样例图")
    print("\n开始校验（请耐心等待 MLLM 响应）...")

    start_time = time.time()
    result = verifier.verify_brand_font(image_path, temperature=0.3)
    elapsed = time.time() - start_time

    print(f"\n总耗时: {elapsed:.1f}s")
    print(f"success: {result.get('success')}")
    if result.get('error'):
        print(f"error: {result.get('error')}")

    # 打印逐项结果
    results = result.get("results", {})
    print_separator("校验结果")
    print_item_result("中文 '中国南方电网'", results.get("chinese", {}))
    print_item_result("英文 'CHINA SOUTHERN POWER GRID'", results.get("english_full", {}))
    print_item_result("英文 'CSG'", results.get("csg", {}))

    # 打印 JSON 格式结果
    print_separator("完整 JSON 输出")
    print(json.dumps(result.get("results", {}), ensure_ascii=False, indent=2))

    # 汇总
    print_separator("汇总")
    chinese_ok = results.get("chinese", {}).get("conforms_flag") == "是"
    english_ok = results.get("english_full", {}).get("conforms_flag") == "是"
    csg_flag = results.get("csg", {}).get("conforms_flag")
    csg_ok = csg_flag in ("是", "不适用")  # 不适用视为通过

    print(f"中文字体规范:   {'✓ 符合' if chinese_ok else '✗ 不符合'}")
    print(f"英文全称规范:   {'✓ 符合' if english_ok else '✗ 不符合'}")
    print(f"英文简称规范:   {'✓ 符合' if csg_ok == '是' else ('- 不适用' if csg_flag == '不适用' else '✗ 不符合')}")
    print(f"总体:           {'✓ 全部通过' if (chinese_ok and english_ok and csg_ok) else '✗ 存在不符项'}")

    print_separator("测试完成")


if __name__ == "__main__":
    main()
