---
name: "paddle-ocr-env"
description: "Runs PaddleOCR project scripts in the local conda python_11 virtual environment. Invoke when user wants to run any Python script in this project or install dependencies."
---

# PaddleOCR Project Environment

All Python programs in this project must be executed in the local conda virtual environment **python_11**.

## How to Run

Always activate the conda environment before running any Python script:

```powershell
conda activate python_11
python <script_path>
```

## How to Install Dependencies

When installing any Python packages, always use the python_11 environment:

```powershell
conda activate python_11
pip install <package_name>
```

## Project Location

- Project root: `c:\Users\12908\Desktop\vi_project\Paddle_OCR_Parsing`
- Entry script: `fun.py`
- Key modules: `layout_parsing_batch_img_API.py`, `json_feature_extraction_api.py`

## Important Notes

- Never use the system default Python or other conda environments.
- Always verify the environment is `python_11` before running scripts.
- The project depends on PaddlePaddle and PaddleX, which are installed in python_11.
