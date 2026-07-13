# -*- coding: utf-8 -*-
"""测试错误字体识别能力。"""
import sys
import time
import json
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from Logo_Font_Verify import FontVerifier

ERROR_FONTS = [
    "input/错误字体.png",
    "input/错误字体2.png",
    "input/错误字体3.png",
]
CORRECT_FONT = "input/A1.05_1_不同组合形式品牌标识_3_横式.png"

verifier = FontVerifier()

# 测试正确字体
print("=" * 60)
print("  正确字体（对照组）")
print("=" * 60)
img = str(_PROJECT_ROOT / CORRECT_FONT)
print(f"图片: {Path(img).name} ({Path(img).stat().st_size} bytes)")
start = time.time()
result = verifier.verify_brand_font(img, temperature=0.3)
elapsed = time.time() - start
print(f"耗时: {elapsed:.1f}s")
for key, label in [("chinese", "中文"), ("english_full", "英文全称"), ("csg", "CSG")]:
    item = result.get("results", {}).get(key, {})
    print(f"  [{label}] flag={item.get('conforms_flag')}  conf={item.get('confidence')}  detail={item.get('detail', '')[:100]}")

# 测试错误字体
for font_path in ERROR_FONTS:
    img = str(_PROJECT_ROOT / font_path)
    if not Path(img).exists():
        print(f"\n[跳过] 图片不存在: {font_path}")
        continue
    print("\n" + "=" * 60)
    print(f"  错误字体: {Path(img).name}")
    print("=" * 60)
    print(f"图片: {Path(img).name} ({Path(img).stat().st_size} bytes)")
    start = time.time()
    result = verifier.verify_brand_font(img, temperature=0.3)
    elapsed = time.time() - start
    print(f"耗时: {elapsed:.1f}s")
    for key, label in [("chinese", "中文"), ("english_full", "英文全称"), ("csg", "CSG")]:
        item = result.get("results", {}).get(key, {})
        print(f"  [{label}] flag={item.get('conforms_flag')}  conf={item.get('confidence')}  detail={item.get('detail', '')[:100]}")

print("\n" + "=" * 60)
print("  测试完成")
print("=" * 60)
