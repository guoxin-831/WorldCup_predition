"""
config.py
==========================
项目全局配置
统一管理所有路径和模型参数
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

# ----------------------------
# 随机种子配置
# ----------------------------
RANDOM_SEED = 42

# ----------------------------
# 数据划分配置
# ----------------------------
TRAIN_TEST_SPLIT = 0.2

# ----------------------------
# 并行训练配置
# ----------------------------
PARALLEL_TRAINING = True
MAX_WORKERS = 4

# ----------------------------
# 模型参数配置
# ----------------------------
MODEL_PARAMS = {
    "逻辑回归": {
        "solver": "lbfgs",
        "max_iter": 1000,
        "class_weight": "balanced",
        "random_state": RANDOM_SEED
    },
    "随机森林": {
        "n_estimators": 150,
        "max_depth": 10,
        "min_samples_split": 4,
        "min_samples_leaf": 1,
        "class_weight": "balanced",
        "random_state": RANDOM_SEED
    },
    "梯度提升": {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.1,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "random_state": RANDOM_SEED
    },
    "XGBoost": {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,
        "scale_pos_weight": 1,
        "random_state": RANDOM_SEED,
        "verbosity": 0
    },
    "LightGBM": {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,
        "class_weight": "balanced",
        "random_state": RANDOM_SEED,
        "verbose": -1
    },
    "CatBoost": {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.1,
        "class_weights": [1, 1, 1],
        "random_state": RANDOM_SEED,
        "verbose": 0
    },
    "K近邻": {
        "n_neighbors": 7,
        "weights": "distance"
    },
    "支持向量机": {
        "kernel": "rbf",
        "class_weight": "balanced",
        "random_state": RANDOM_SEED,
        "probability": True
    }
}

# ----------------------------
# 超参数调优配置
# ----------------------------
HYPERPARAM_GRIDS = {
    "逻辑回归": {
        "C": [0.1, 1, 10, 100],
        "penalty": ["l2"],
        "max_iter": [500, 1000, 2000]
    },
    "随机森林": {
        "n_estimators": [50, 100, 150, 200],
        "max_depth": [3, 5, 7, 10],
        "min_samples_split": [2, 4, 6],
        "min_samples_leaf": [1, 2, 3]
    },
    "梯度提升": {
        "n_estimators": [50, 100, 150],
        "max_depth": [2, 3, 5],
        "learning_rate": [0.05, 0.1, 0.2],
        "min_samples_split": [5, 10, 15]
    },
    "XGBoost": {
        "n_estimators": [50, 100, 150],
        "max_depth": [2, 3, 5],
        "learning_rate": [0.05, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0]
    },
    "LightGBM": {
        "n_estimators": [50, 100, 150],
        "max_depth": [2, 3, 5],
        "learning_rate": [0.05, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0]
    },
    "CatBoost": {
        "n_estimators": [50, 100, 150],
        "max_depth": [2, 3, 5],
        "learning_rate": [0.05, 0.1, 0.2]
    },
    "K近邻": {
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"]
    },
    "支持向量机": {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", "auto", 0.1, 1],
        "kernel": ["rbf"]
    },
    "Voting集成": {},
    "Stacking集成": {}
}

# ----------------------------
# 需要标准化的模型
# ----------------------------
MODELS_NEED_SCALING = ["逻辑回归", "K近邻", "支持向量机"]

# ----------------------------
# 需要整数标签的模型
# ----------------------------
MODELS_NEED_INT_LABELS = ["XGBoost", "CatBoost"]

# ----------------------------
# 标签映射
# ----------------------------
LABEL_MAPPING = {"主队胜": 0, "平局": 1, "客队胜": 2}
NUM_TO_LABEL = {0: "主队胜", 1: "平局", 2: "客队胜"}

# ----------------------------
# 评估指标配置
# ----------------------------
MIN_ACCURACY_THRESHOLD = 0.35
