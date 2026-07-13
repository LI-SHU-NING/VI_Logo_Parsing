"""Logo 分析 API 路由。

提供 Logo 存在性检测、颜色分析、字体分析等接口。
"""

import sys
import uuid
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel

# 将项目根目录加入 sys.path，以便导入 Logo_Font_Verify
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from .analyzer import LogoAnalyzer
from Logo_Font_Verify import FontVerifier

router = APIRouter(tags=["Logo分析"])

# 创建全局分析器实例
analyzer = LogoAnalyzer()
# 字体识别与验证器实例
font_verifier = FontVerifier()

# 项目临时目录
TEMP_DIR = Path(__file__).parent.parent.parent / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)


# --- 请求模型 ---

class CheckLogoRequest(BaseModel):
    """检查 logo 是否存在的请求"""
    image_path: str
    temperature: float = 0.3


class AnalyzeRequest(BaseModel):
    """通用分析请求"""
    image_path: str
    temperature: float = 0.3


class VerifyFontRequest(BaseModel):
    """字体验证请求"""
    image_path: str
    expected_font: str = "黑体"  # 期望的品牌标准字体：黑体/微软雅黑/宋体
    reference_text: str = "中国南方电网 CHINA SOUTHERN POWER GRID"
    temperature: float = 0.3


# --- 响应模型 ---

class CheckLogoResponse(BaseModel):
    """检查 logo 是否存在的响应"""
    success: bool
    logo_exists: bool = False
    confidence: str = "未知"
    error: str = ""


class AnalyzeResponse(BaseModel):
    """通用分析响应"""
    success: bool
    color: str = ""
    font_analysis: str = ""
    error: str = ""


# --- API 接口 ---

def _save_upload_file(file: UploadFile) -> Path:
    """
    保存上传的文件到项目临时目录。

    Args:
        file: 上传的文件对象

    Returns:
        保存后的文件路径
    """
    # 生成唯一文件名
    file_ext = Path(file.filename).suffix
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    file_path = TEMP_DIR / unique_filename

    # 保存文件
    contents = file.file.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    return file_path


def _cleanup_file(file_path: Path):
    """
    清理临时文件。

    Args:
        file_path: 要删除的文件路径
    """
    try:
        if file_path.exists():
            file_path.unlink()
    except Exception:
        pass  # 忽略删除失败


@router.post("/check-logo", summary="检查图片中是否存在指定 logo")
def check_logo_exists(
    file: UploadFile = File(...),
    temperature: float = 0.3,
):
    """
    上传图片，检查是否包含与参考图片相同的 logo。

    参考图片默认为 data/logo/logo1.png。

    Args:
        file: 上传的图片文件
        temperature: 采样温度，默认 0.3

    Returns:
        包含判断结果的响应
    """
    file_path = None
    try:
        # 保存上传的文件到项目临时目录
        file_path = _save_upload_file(file)

        # 调用分析器
        result = analyzer.check_logo_exists(
            user_image_path=str(file_path),
            temperature=temperature,
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")
    finally:
        # 清理临时文件
        if file_path:
            _cleanup_file(file_path)


@router.post("/check-logo-path", summary="检查服务器路径图片中是否存在指定 logo")
def check_logo_exists_by_path(request: CheckLogoRequest):
    """
    通过服务器路径检查图片是否包含与参考图片相同的 logo。

    Args:
        request: 包含图片路径和温度参数的请求

    Returns:
        包含判断结果的响应
    """
    try:
        result = analyzer.check_logo_exists(
            user_image_path=request.image_path,
            temperature=request.temperature,
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")


@router.post("/analyze-color", summary="分析 logo 颜色")
def analyze_logo_color(
    file: UploadFile = File(...),
    temperature: float = 0.3,
):
    """
    上传图片，分析其中 logo 的颜色。

    Args:
        file: 上传的图片文件
        temperature: 采样温度，默认 0.3

    Returns:
        包含颜色分析结果的响应
    """
    file_path = None
    try:
        # 保存上传的文件到项目临时目录
        file_path = _save_upload_file(file)

        # 调用分析器
        result = analyzer.analyze_logo_color(
            user_image_path=str(file_path),
            temperature=temperature,
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"颜色分析失败: {str(e)}")
    finally:
        # 清理临时文件
        if file_path:
            _cleanup_file(file_path)


@router.post("/analyze-color-path", summary="分析服务器路径图片的 logo 颜色")
def analyze_logo_color_by_path(request: AnalyzeRequest):
    """
    通过服务器路径分析图片中 logo 的颜色。

    Args:
        request: 包含图片路径和温度参数的请求

    Returns:
        包含颜色分析结果的响应
    """
    try:
        result = analyzer.analyze_logo_color(
            user_image_path=request.image_path,
            temperature=request.temperature,
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"颜色分析失败: {str(e)}")


@router.post("/analyze-font", summary="分析 logo 字体")
def analyze_logo_font(
    file: UploadFile = File(...),
    temperature: float = 0.3,
):
    """
    上传图片，分析其中 logo 的字体（单图识别模式）。

    Args:
        file: 上传的图片文件
        temperature: 采样温度，默认 0.3

    Returns:
        包含字体分析结果的响应
    """
    file_path = None
    try:
        # 保存上传的文件到项目临时目录
        file_path = _save_upload_file(file)

        # 调用 FontVerifier 进行字体识别
        result = font_verifier.analyze_font(
            user_image_path=str(file_path),
            temperature=temperature,
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"字体分析失败: {str(e)}")
    finally:
        # 清理临时文件
        if file_path:
            _cleanup_file(file_path)


@router.post("/analyze-font-path", summary="分析服务器路径图片的 logo 字体")
def analyze_logo_font_by_path(request: AnalyzeRequest):
    """
    通过服务器路径分析图片中 logo 的字体（单图识别模式）。

    Args:
        request: 包含图片路径和温度参数的请求

    Returns:
        包含字体分析结果的响应
    """
    try:
        result = font_verifier.analyze_font(
            user_image_path=request.image_path,
            temperature=request.temperature,
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"字体分析失败: {str(e)}")


@router.post("/verify-font", summary="验证 logo 字体是否符合品牌标准")
def verify_logo_font(
    file: UploadFile = File(...),
    expected_font: str = "黑体",
    reference_text: str = "中国南方电网 CHINA SOUTHERN POWER GRID",
    temperature: float = 0.3,
):
    """
    上传图片，验证其中 logo 的字体是否符合品牌标准（双图对比模式）。

    使用品牌标准字体文件渲染参考文本图片，再与用户图片一起送入 MLLM 做视觉对比。

    Args:
        file: 上传的图片文件
        expected_font: 期望的品牌标准字体（黑体/微软雅黑/宋体），默认黑体
        reference_text: 渲染参考图使用的文本
        temperature: 采样温度，默认 0.3

    Returns:
        包含字体验证结果的响应
    """
    file_path = None
    try:
        file_path = _save_upload_file(file)

        result = font_verifier.verify_font(
            user_image_path=str(file_path),
            expected_font=expected_font,
            reference_text=reference_text,
            temperature=temperature,
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"字体验证失败: {str(e)}")
    finally:
        if file_path:
            _cleanup_file(file_path)


@router.post("/verify-font-path", summary="验证服务器路径图片的 logo 字体")
def verify_logo_font_by_path(request: VerifyFontRequest):
    """
    通过服务器路径验证图片中 logo 的字体是否符合品牌标准（双图对比模式）。

    Args:
        request: 包含图片路径、期望字体、参考文本和温度参数的请求

    Returns:
        包含字体验证结果的响应
    """
    try:
        result = font_verifier.verify_font(
            user_image_path=request.image_path,
            expected_font=request.expected_font,
            reference_text=request.reference_text,
            temperature=request.temperature,
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"字体验证失败: {str(e)}")
