"""
predictor.py
==========================
Task3 预测模块
==========================
功能：
1. 加载训练好的模型和标准化器
2. 构建比赛特征向量（无数据泄漏）
3. 预测比赛结果（胜负平）
4. 输出预测概率和结果解释
5. 批量预测支持
==========================
"""

import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Optional, Any
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from src.config import MODEL_DIR
from .utils import setup_logger, validate_team_name, format_probability, normalize_probabilities

logger = setup_logger(__name__)


class Predictor:
    """
    比赛结果预测器

    Attributes:
        model: 训练好的分类模型
        scaler: 标准化器
        team_history: 队伍历史数据
        feature_columns: 特征列名列表
        label_mapping: 标签映射
        num_to_label: 数值到标签的映射
    """

    def __init__(
        self,
        model: Optional[BaseEstimator] = None,
        scaler: Optional[StandardScaler] = None,
        team_history: Optional[pd.DataFrame] = None,
        feature_columns: Optional[List[str]] = None,
        home_advantage_factor: float = 0.85
    ):
        """
        初始化预测器

        Args:
            model: 训练好的分类模型
            scaler: 标准化器
            team_history: 队伍历史数据
            feature_columns: 特征列名列表
            home_advantage_factor: 主队优势因子，用于调整主队获胜概率。
                                   值越小，主队获胜概率降低越多。默认0.85。
        """
        self.model = model
        self.scaler = scaler
        self.team_history = team_history.set_index("队伍名称") if team_history is not None else None
        self.home_advantage_factor = home_advantage_factor

        self.feature_columns = feature_columns or [
            "阶段类型",
            "参赛次数差", "比赛场次差", "场均进球差", "场均失球差",
            "净胜球差", "成绩排名差", "近3届场均进球差", "近3届胜率差",
            "场均净胜球差", "淘汰赛胜率差", "小组赛胜率差",
            "场均半场进球差", "半场胜率差",
            "交锋胜场", "交锋平局", "交锋负场", "交锋净胜球",
            "交锋总场次", "交锋胜率"
        ]

        self.label_mapping = {"主队胜": 0, "平局": 1, "客队胜": 2}
        self.num_to_label = {0: "主队胜", 1: "平局", 2: "客队胜"}

        logger.info(f"预测器初始化完成，主队优势因子: {home_advantage_factor}")

    def load_model(self, model_name: str = "CatBoost") -> bool:
        """
        加载训练好的模型和标准化器

        Args:
            model_name: 模型名称

        Returns:
            是否加载成功
        """
        logger.info(f"开始加载模型: {model_name}")

        model_path = MODEL_DIR / f"task3_{model_name}_model.pkl"
        scaler_path = MODEL_DIR / "task3_scaler.pkl"

        try:
            if model_path.exists():
                self.model = joblib.load(model_path)
                logger.info(f"模型加载成功: {model_path}")
            else:
                logger.warning(f"模型文件不存在: {model_path}")
                return False

            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info(f"标准化器加载成功: {scaler_path}")
            else:
                logger.warning(f"标准化器文件不存在: {scaler_path}")
                return False

            return True

        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return False

    def _get_team_stats(self, team_name: str) -> Dict[str, Any]:
        """
        获取队伍的历史统计数据

        Args:
            team_name: 队伍名称

        Returns:
            队伍统计数据字典
        """
        if self.team_history is None:
            logger.warning("队伍历史数据未加载")
            return self._get_default_stats()

        if team_name not in self.team_history.index:
            logger.warning(f"队伍 {team_name} 不在历史数据中，使用默认值")
            return self._get_default_stats()

        stats = self.team_history.loc[team_name].to_dict()

        default_stats = self._get_default_stats()
        return {**default_stats, **stats}

    def _get_default_stats(self) -> Dict[str, Any]:
        """
        获取默认的队伍统计数据（用于新队伍或未知队伍）

        Returns:
            默认统计数据字典
        """
        return {
            "历史参赛次数": 0,
            "历史比赛场次": 0,
            "历史场均进球": 0,
            "历史场均失球": 0,
            "历史净胜球": 0,
            "历史场均净胜球": 0,
            "历史成绩排名": 0,
            "近3届场均进球": 0,
            "近3届胜率": 0,
            "历史淘汰赛胜率": 0,
            "历史小组赛胜率": 0,
            "历史场均半场进球": 0,
            "历史半场胜率": 0
        }

    def _build_feature_vector(
        self,
        home_team: str,
        away_team: str,
        stage: str = "小组赛",
        h2h_record: Optional[Dict[str, int]] = None
    ) -> pd.DataFrame:
        """
        构建比赛特征向量

        Args:
            home_team: 主队名称
            away_team: 客队名称
            stage: 比赛阶段（小组赛/淘汰赛）
            h2h_record: 交锋记录

        Returns:
            特征向量数据框
        """
        logger.debug(f"构建特征向量: {home_team} vs {away_team}")

        home_stats = self._get_team_stats(home_team)
        away_stats = self._get_team_stats(away_team)

        if h2h_record is None:
            h2h_record = {"胜场": 0, "平局": 0, "负场": 0, "净胜球": 0}

        stage_type = 0 if stage == "小组赛" else 1

        h2h_total = h2h_record["胜场"] + h2h_record["平局"] + h2h_record["负场"]
        h2h_win_rate = h2h_record["胜场"] / h2h_total if h2h_total > 0 else 0

        feature_dict = {
            "阶段类型": stage_type,
            "参赛次数差": home_stats["历史参赛次数"] - away_stats["历史参赛次数"],
            "比赛场次差": home_stats["历史比赛场次"] - away_stats["历史比赛场次"],
            "场均进球差": home_stats["历史场均进球"] - away_stats["历史场均进球"],
            "场均失球差": home_stats["历史场均失球"] - away_stats["历史场均失球"],
            "净胜球差": home_stats["历史净胜球"] - away_stats["历史净胜球"],
            "成绩排名差": home_stats["历史成绩排名"] - away_stats["历史成绩排名"],
            "近3届场均进球差": home_stats["近3届场均进球"] - away_stats["近3届场均进球"],
            "近3届胜率差": home_stats["近3届胜率"] - away_stats["近3届胜率"],
            "场均净胜球差": home_stats["历史场均净胜球"] - away_stats["历史场均净胜球"],
            "淘汰赛胜率差": home_stats["历史淘汰赛胜率"] - away_stats["历史淘汰赛胜率"],
            "小组赛胜率差": home_stats["历史小组赛胜率"] - away_stats["历史小组赛胜率"],
            "场均半场进球差": home_stats["历史场均半场进球"] - away_stats["历史场均半场进球"],
            "半场胜率差": home_stats["历史半场胜率"] - away_stats["历史半场胜率"],
            "交锋胜场": h2h_record["胜场"],
            "交锋平局": h2h_record["平局"],
            "交锋负场": h2h_record["负场"],
            "交锋净胜球": h2h_record["净胜球"],
            "交锋总场次": h2h_total,
            "交锋胜率": h2h_win_rate
        }

        feature_df = pd.DataFrame([feature_dict])

        for col in self.feature_columns:
            if col not in feature_df.columns:
                feature_df[col] = 0

        return feature_df[self.feature_columns]

    def predict(
        self,
        home_team: str,
        away_team: str,
        stage: str = "小组赛",
        h2h_record: Optional[Dict[str, int]] = None
    ) -> Dict[str, Any]:
        """
        预测比赛结果

        Args:
            home_team: 主队名称
            away_team: 客队名称
            stage: 比赛阶段（小组赛/淘汰赛）
            h2h_record: 交锋记录

        Returns:
            预测结果字典，包含预测结果、概率分布等
        """
        logger.info(f"开始预测: {home_team} vs {away_team}")

        if self.model is None:
            logger.error("模型未加载")
            return {"error": "模型未加载"}

        if self.scaler is None:
            logger.error("标准化器未加载")
            return {"error": "标准化器未加载"}

        feature_df = self._build_feature_vector(home_team, away_team, stage, h2h_record)

        try:
            scaled_features = self.scaler.transform(feature_df)

            raw_proba = self.model.predict_proba(scaled_features)[0]
            raw_proba = normalize_probabilities(raw_proba.tolist())

            adjusted_proba = [
                raw_proba[0] * self.home_advantage_factor,
                raw_proba[1],
                raw_proba[2]
            ]
            adjusted_proba = normalize_probabilities(adjusted_proba)

            pred_label = self.model.predict(scaled_features)[0]

            if isinstance(pred_label, (np.ndarray, list)):
                pred_label = pred_label[0]

            if isinstance(pred_label, int):
                result_label = self.num_to_label.get(pred_label, "平局")
            else:
                result_label = str(pred_label)

            adjusted_pred_label = self.num_to_label.get(
                int(np.argmax(adjusted_proba)),
                result_label
            )

            result = {
                "主队": home_team,
                "客队": away_team,
                "比赛阶段": stage,
                "预测结果": adjusted_pred_label,
                "主队胜概率": adjusted_proba[0],
                "平局概率": adjusted_proba[1],
                "客队胜概率": adjusted_proba[2],
                "概率分布": {
                    "主队胜": adjusted_proba[0],
                    "平局": adjusted_proba[1],
                    "客队胜": adjusted_proba[2]
                },
                "原始概率分布": {
                    "主队胜": raw_proba[0],
                    "平局": raw_proba[1],
                    "客队胜": raw_proba[2]
                },
                "主队优势因子": self.home_advantage_factor
            }

            logger.info(f"预测完成: {home_team} vs {away_team} -> {adjusted_pred_label} (原始: {result_label})")

            return result

        except Exception as e:
            logger.error(f"预测失败: {e}")
            return {"error": str(e)}

    def predict_batch(
        self,
        matches: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        批量预测比赛结果

        Args:
            matches: 比赛列表，每个元素包含 home_team, away_team, stage（可选）

        Returns:
            预测结果列表
        """
        logger.info(f"开始批量预测，共 {len(matches)} 场比赛")

        results = []
        for match in matches:
            home_team = match.get("home_team", "")
            away_team = match.get("away_team", "")
            stage = match.get("stage", "小组赛")
            h2h_record = match.get("h2h_record", None)

            if not home_team or not away_team:
                logger.warning("缺少队伍名称")
                results.append({"error": "缺少队伍名称"})
                continue

            result = self.predict(home_team, away_team, stage, h2h_record)
            results.append(result)

        logger.info(f"批量预测完成")

        return results

    def print_prediction_result(self, result: Dict[str, Any]) -> None:
        """
        打印预测结果

        Args:
            result: 预测结果字典
        """
        if "error" in result:
            print(f"❌ 预测失败: {result['error']}")
            return

        print(f"\n{'='*60}")
        print(f"预测结果: {result['主队']} vs {result['客队']}")
        print(f"比赛阶段: {result['比赛阶段']}")
        print(f"{'='*60}")
        print(f"预测结果: {result['预测结果']}")
        print(f"主队胜概率: {format_probability(result['主队胜概率'])}")
        print(f"平局概率: {format_probability(result['平局概率'])}")
        print(f"客队胜概率: {format_probability(result['客队胜概率'])}")
        print(f"{'='*60}")

    def get_probability_distribution(self, result: Dict[str, Any]) -> Dict[str, str]:
        """
        获取格式化的概率分布

        Args:
            result: 预测结果字典

        Returns:
            格式化的概率分布字典
        """
        if "error" in result:
            return {}

        return {
            "主队胜": format_probability(result["主队胜概率"]),
            "平局": format_probability(result["平局概率"]),
            "客队胜": format_probability(result["客队胜概率"])
        }
