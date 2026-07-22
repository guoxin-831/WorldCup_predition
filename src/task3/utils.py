"""
utils.py
==========================
Task3 工具模块
==========================
功能：
1. 日志配置和管理
2. 文件操作工具
3. 数据验证和处理
4. 类型提示和工具函数
==========================
"""

import logging
import os
import json
import random
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np


def setup_logger(name: str = "WorldCupPrediction", level: int = logging.INFO) -> logging.Logger:
    """
    配置日志系统

    Args:
        name: 日志器名称
        level: 日志级别

    Returns:
        配置好的日志器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    log_dir = os.path.join(os.path.dirname(__file__), "..", "..", "logs")
    os.makedirs(log_dir, exist_ok=True)

    file_handler = logging.FileHandler(
        os.path.join(log_dir, "task3.log"),
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def save_to_json(data: Dict[str, Any], file_path: str) -> bool:
    """
    保存数据到JSON文件

    Args:
        data: 要保存的数据
        file_path: 文件路径

    Returns:
        是否保存成功
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger = setup_logger()
        logger.error(f"保存JSON文件失败: {file_path}, 错误: {e}")
        return False


def load_from_json(file_path: str) -> Optional[Dict[str, Any]]:
    """
    从JSON文件加载数据

    Args:
        file_path: 文件路径

    Returns:
        加载的数据，如果失败返回None
    """
    try:
        if not os.path.exists(file_path):
            return None
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger = setup_logger()
        logger.error(f"加载JSON文件失败: {file_path}, 错误: {e}")
        return None


def validate_team_name(team_name: str, valid_teams: List[str]) -> bool:
    """
    验证队伍名称是否有效

    Args:
        team_name: 队伍名称
        valid_teams: 有效队伍列表

    Returns:
        是否有效
    """
    if not isinstance(team_name, str):
        return False
    if team_name.strip() == "":
        return False
    return team_name.strip() in valid_teams


def validate_feature_data(feature_data: Dict[str, Any], required_features: List[str]) -> bool:
    """
    验证特征数据是否完整

    Args:
        feature_data: 特征数据
        required_features: 必需特征列表

    Returns:
        是否完整
    """
    if not isinstance(feature_data, dict):
        return False
    for feature in required_features:
        if feature not in feature_data:
            return False
        if feature_data[feature] is None:
            return False
        if isinstance(feature_data[feature], float) and np.isnan(feature_data[feature]):
            return False
    return True


def safe_divide(a: Union[int, float], b: Union[int, float], default: float = 0.0) -> float:
    """
    安全除法，避免除零错误

    Args:
        a: 被除数
        b: 除数
        default: 除零时的默认值

    Returns:
        除法结果或默认值
    """
    try:
        if b == 0:
            return default
        return float(a) / float(b)
    except (TypeError, ValueError):
        return default


def normalize_probabilities(probs: List[float]) -> List[float]:
    """
    归一化概率列表，确保总和为1

    Args:
        probs: 概率列表

    Returns:
        归一化后的概率列表
    """
    try:
        total = sum(probs)
        if total == 0:
            return [1.0 / len(probs)] * len(probs)
        return [p / total for p in probs]
    except (TypeError, ValueError):
        return [0.0] * len(probs)


def format_probability(prob: float) -> str:
    """
    格式化概率为百分比字符串

    Args:
        prob: 概率值

    Returns:
        格式化后的字符串
    """
    try:
        return f"{prob * 100:.2f}%"
    except (TypeError, ValueError):
        return "0.00%"


def get_team_list(df: pd.DataFrame) -> List[str]:
    """
    从数据框中获取所有队伍名称

    Args:
        df: 包含队伍信息的数据框

    Returns:
        队伍名称列表
    """
    home_teams = df["主队名称"].unique().tolist()
    away_teams = df["客队名称"].unique().tolist()
    all_teams = list(set(home_teams + away_teams))
    return sorted(all_teams)


def calculate_time_diff(func):
    """
    装饰器：计算函数执行时间

    Args:
        func: 被装饰的函数

    Returns:
        包装后的函数
    """
    import time

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger = setup_logger()
        logger.debug(f"{func.__name__} 执行时间: {end_time - start_time:.2f}秒")
        return result

    return wrapper


def handle_exceptions(default_return=None):
    """
    装饰器：捕获异常并返回默认值

    Args:
        default_return: 异常发生时的默认返回值

    Returns:
        包装后的函数
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = setup_logger()
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"{func.__name__} 执行失败: {e}")
                return default_return

        return wrapper

    return decorator


def setup_random_seed(seed: int = 42) -> None:
    """
    设置随机种子，保证实验可重复性

    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)


def setup_chinese_font() -> None:
    """
    设置中文显示字体，解决matplotlib中文乱码问题
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.rcParams["font.sans-serif"] = [
            "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"
        ]
        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams["font.family"] = "sans-serif"
        sns.set_theme(style="whitegrid", font="Microsoft YaHei")
    except ImportError:
        logger = setup_logger()
        logger.warning("matplotlib或seaborn未安装，无法设置中文字体")
