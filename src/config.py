"""
config.py
==========================
项目全局配置
统一管理所有路径
==========================
"""

from pathlib import Path

# ----------------------------
# 项目根目录
# ----------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent

# ----------------------------
# 数据目录
# ----------------------------
DATA_DIR = ROOT_DIR / "data"

# ----------------------------
# 输出目录
# ----------------------------
OUTPUT_DIR = ROOT_DIR / "output"

FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
REPORT_DIR = OUTPUT_DIR / "reports"
MODEL_DIR = OUTPUT_DIR / "models"

# ----------------------------
# 自动创建目录
# ----------------------------
for folder in [
    FIGURE_DIR,
    TABLE_DIR,
    REPORT_DIR,
    MODEL_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)