"""图像清晰度检测模块。

使用 Laplacian 方差 + Sobel/Tenengrad 梯度能量，对整张图像进行模糊程度评估。
"""

import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


@dataclass
class ClarityResult:
    """清晰度检测结果。"""

    is_clear: bool
    laplacian_var: float
    sobel_var: float
    combined_score: float
    threshold: float

    def __str__(self) -> str:
        status = "CLEAR" if self.is_clear else "BLURRY"
        return (
            f"[{status}] combined={self.combined_score:.2f} "
            f"laplacian_var={self.laplacian_var:.2f} "
            f"sobel_var={self.sobel_var:.2f} "
            f"threshold={self.threshold:.2f}"
        )


def _load_gray(image_input: Union[str, Path, np.ndarray]) -> Optional[np.ndarray]:
    """加载灰度图，支持文件路径或 numpy 数组。"""
    if isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 2:
            return image_input.astype(np.float64)
        if len(image_input.shape) == 3:
            return cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY).astype(np.float64)
        return None

    path = str(image_input)
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    return gray.astype(np.float64)


def compute_laplacian_var(gray: np.ndarray) -> float:
    """计算 Laplacian 方差（对离焦模糊最敏感）。"""
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    return float(lap.var())


def compute_sobel_var(gray: np.ndarray) -> float:
    """计算 Sobel 梯度幅值的方差（Tenengrad 思路，对边缘梯度敏感）。"""
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.hypot(sx, sy)
    return float(gradient_mag.var())


def check_clarity(
    image_input: Union[str, Path, np.ndarray],
    threshold: float = 1.0,
    laplacian_weight: float = 0.5,
    sobel_weight: float = 0.5,
    normalize: bool = True,
) -> ClarityResult:
    """检测图像清晰度。

    Args:
        image_input: 图片文件路径或 BGR/灰度 numpy 数组。
        threshold: 清晰度阈值，低于此值判定为模糊。
            - 归一化模式（normalize=True）：阈值范围 0.5-2.0，默认 1.0
              清晰图 combined 通常 > 1.5，模糊图 < 1.0
            - 原始模式（normalize=False）：阈值范围 50-500，默认 100
              清晰图 combined 通常 > 200，模糊图 < 100
        laplacian_weight: Laplacian 方差权重，默认 0.5。
        sobel_weight: Sobel 梯度方差权重，默认 0.5。
        normalize: 是否用图像纹理方差归一化。推荐开启，可消除分辨率差异影响。

    Returns:
        ClarityResult 包含各项分数和判定结果。
    """
    gray = _load_gray(image_input)
    if gray is None:
        return ClarityResult(
            is_clear=False,
            laplacian_var=0.0,
            sobel_var=0.0,
            combined_score=0.0,
            threshold=threshold,
        )

    lap_var = compute_laplacian_var(gray)
    sob_var = compute_sobel_var(gray)

    if normalize:
        texture_var = float(np.var(gray)) + 1e-6
        lap_norm = lap_var / texture_var
        sob_norm = sob_var / texture_var
    else:
        lap_norm = lap_var
        sob_norm = sob_var

    combined = laplacian_weight * lap_norm + sobel_weight * sob_norm

    return ClarityResult(
        is_clear=combined >= threshold,
        laplacian_var=lap_var,
        sobel_var=sob_var,
        combined_score=combined,
        threshold=threshold,
    )


def is_image_clear(
    image_input: Union[str, Path, np.ndarray],
    threshold: float = 1.0,
) -> bool:
    """判断图像是否清晰，供项目其他模块直接调用。

    Args:
        image_input: 图片文件路径或 BGR/灰度 numpy 数组。
        threshold: 清晰度阈值（归一化模式默认 1.0）。
                   归一化模式下: 清晰图通常 > 1.5，模糊图 < 1.0。
                   可根据业务图片样本微调。

    Returns:
        True 表示清晰，False 表示模糊。

    用法::

        from blur_detection import is_image_clear

        if not is_image_clear("photo.jpg"):
            print("图片模糊，停止处理")
            return
    """
    return check_clarity(image_input, threshold=threshold).is_clear
