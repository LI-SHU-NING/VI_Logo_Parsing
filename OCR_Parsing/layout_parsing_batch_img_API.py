import os
from pathlib import Path
from paddlex import create_pipeline


def process_images(in_path: Path, output_path: Path, pp_option=None, engine_config=None):
    """
    处理图片文件夹或单个PDF/图片文件，使用 PPStructureV3 进行结构化文本识别，
    并保存结果为多种格式（图片、Markdown、JSON、Excel等）。

    参数:
    in_path (Path): 输入路径（支持文件夹路径 / 单个PDF文件路径 / 单个图片文件路径）
    output_path (Path): 输出路径，保存处理后的文件。
    """
    # 确保输出路径存在
    output_path.mkdir(parents=True, exist_ok=True)

    # 创建 layout_parsing 实例（使用默认配置，不启用 textline_orientation 模型）
    pipeline = create_pipeline(pipeline="layout_parsing", pp_option=pp_option, engine_config=engine_config)

    # 支持的文件格式（图片 + PDF）
    support_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".pdf"]
    # 单独拆分图片格式（用于文件夹遍历）
    image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

    # ------------------- 核心修改：兼容文件夹/单个文件 -------------------
    file_list = []
    if in_path.is_dir():
        # 场景1：输入是文件夹 → 遍历所有图片文件
        file_list = sorted([
            f for f in in_path.iterdir()
            if f.suffix.lower() in image_exts and f.is_file()
        ])
        if not file_list:
            raise ValueError(f"文件夹 {in_path} 中没有找到支持的图片文件！")
    elif in_path.is_file():
        # 场景2：输入是单个文件 → 判断是否为支持的格式（PDF/图片）
        if in_path.suffix.lower() not in support_exts:
            raise ValueError(
                f"不支持的文件格式：{in_path.suffix}！\n"
                f"仅支持：{', '.join(support_exts)}"
            )
        file_list = [in_path]  # 转为列表，统一后续处理逻辑
    else:
        # 场景3：输入既不是文件夹也不是文件（路径无效）
        raise FileNotFoundError(f"路径不存在或无效：{in_path}")

    # ------------------- 保持原有处理逻辑（统一遍历文件列表） -------------------
    markdown_list = []
    markdown_images = []

    # 逐个处理文件（支持单个/多个文件）
    for file_path in file_list:
        file_type = "PDF文件" if file_path.suffix.lower() == ".pdf" else "图片文件"
        print(f"正在处理 {file_type}：{file_path.name}")

        # 调用 pipeline 预测（支持 PDF/图片输入，直接传文件路径字符串）
        output = pipeline.predict(
            input=str(file_path),  # PPStructureV3 支持 PDF 和图片输入
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            use_seal_recognition=False,
            use_table_recognition = False,
            layout_merge_bboxes_mode="small",
            layout_nms=True
        )

        # 保存结果（每个文件的结果保存到输出目录下）
        for res in output:
            res.print()  # 打印预测的结构化输出
            res.save_to_img(save_path=output_path)  # 保存可视化图像
            res.save_to_json(save_path=output_path)  # 保存JSON结构
            res.save_to_xlsx(save_path=output_path)  # 保存表格Excel
            res.save_to_html(save_path=output_path)  # 保存HTML表格

    print(f"[OK] 全部文件处理完成！输出目录：{output_path}")

'''
# ------------------- 使用示例（三种场景都支持） -------------------
if __name__ == "__main__":
    # 示例1：处理文件夹（原有场景）
    # in_path = Path("./remove_red_seal_batch")
    # output_path = Path("./PP_V3_canshu_output_folder")
    # process_images(in_path, output_path)

    # 示例2：处理单个PDF文件（新增场景，你的报错场景）
    in_path = Path("pdf_input/2/XA_certificate.pdf")
    output_path = Path("./PP_V3_canshu_output_pdf")
    process_images(in_path, output_path)

    # 示例3：处理单个图片文件（新增场景）
    # in_path = Path("./test_images/single_img.jpg")
    # output_path = Path("./PP_V3_canshu_output_single_img")
    # process_images(in_path, output_path)
'''
