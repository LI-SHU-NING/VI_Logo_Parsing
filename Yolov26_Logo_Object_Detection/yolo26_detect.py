# -*- coding: utf-8 -*-
"""
南方电网 Logo 目标检测（YOLO26-nano + ONNX Runtime，纯 CPU）

使用 csg_logo_v5.onnx 模型检测图片中的 csg_logo 目标，
返回 (标签内容, 坐标, 置信率) 元组列表。
"""
import time
import numpy as np
import onnxruntime as ort
import cv2
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / "csg_logo_v5.onnx"
CLASS_NAMES = ["csg_logo"]


def _letterbox(img, new_shape=640):
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw / 2, dh / 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, r, (dw, dh)


def _preprocess(img_bgr, img_size=640):
    img_lb, ratio, pad = _letterbox(img_bgr, (img_size, img_size))
    img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_nchw = np.transpose(img_norm, (2, 0, 1))[np.newaxis]
    return img_nchw, ratio, pad


def detect_logo(img_bgr, conf_thres=0.25, img_size=640, model_path=None):
    """
    检测图片中的南方电网 Logo。

    参数:
        img_bgr:    BGR 格式的 numpy 图像 (h, w, 3)
        conf_thres: 置信率阈值，低于此值的结果被过滤
        img_size:   推理图像尺寸
        model_path: ONNX 模型路径，默认使用同目录下 csg_logo_v5.onnx

    返回:
        list[tuple]: [(标签内容, 坐标(x1,y1,x2,y2), 置信率), ...]
    """
    path = model_path or str(MODEL_PATH)
    session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    orig_shape = img_bgr.shape[:2]
    img_input, ratio, pad = _preprocess(img_bgr, img_size)

    start = time.perf_counter()
    outputs = session.run([output_name], {input_name: img_input})
    infer_ms = (time.perf_counter() - start) * 1000

    # 后处理：(1, 300, 6) → 过滤低置信度 → 还原坐标
    dets = outputs[0][0]
    mask = dets[:, 4] > conf_thres
    dets = dets[mask]

    results = []
    for det in dets:
        x1, y1, x2, y2, conf, cls_id = det
        x1 = int(max(0, min((x1 - pad[0]) / ratio, orig_shape[1])))
        y1 = int(max(0, min((y1 - pad[1]) / ratio, orig_shape[0])))
        x2 = int(max(0, min((x2 - pad[0]) / ratio, orig_shape[1])))
        y2 = int(max(0, min((y2 - pad[1]) / ratio, orig_shape[0])))
        label = CLASS_NAMES[int(cls_id)] if int(cls_id) < len(CLASS_NAMES) else str(int(cls_id))
        results.append((label, (x1, y1, x2, y2), float(conf)))

    return results


if __name__ == "__main__":
    import sys
    img_path = sys.argv[1] if len(sys.argv) > 1 else r"c:\Users\12908\Desktop\vi_project\Paddle_OCR_Parsing\input\南方电网财务有限公司.png"
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is not None:
        dets = detect_logo(img, conf_thres=0.25)
        print(f"检测到 {len(dets)} 个目标:")
        for label, (x1, y1, x2, y2), conf in dets:
            print(f"  {label} | 坐标({x1},{y1},{x2},{y2}) | 置信率={conf:.3f}")
    else:
        print(f"无法读取: {img_path}")
