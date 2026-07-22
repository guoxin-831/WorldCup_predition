"""
feature_engineering.py
==========================
Task3 第一小问：队伍历史特征工程
==========================
功能：
1. 计算每支队伍历史指标：历史参赛次数、历史最佳成绩、历史场均进球、历史场均失球
2. 自动构建特征矩阵（无数据泄漏）
3. 计算各特征与比赛胜负的相关性
4. 输出相关性热力图并保存
5. 打印关键高相关特征
==========================
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from src.config import TABLE_DIR, REPORT_DIR
from .utils import setup_logger, handle_exceptions, calculate_time_diff


logger = setup_logger(__name__)


class TeamFeatureEngineering:
    """
    队伍历史特征工程（无数据泄漏版本）

    Attributes:
        df: 原始数据框
        stage_rank: 阶段排名映射
        history_cache: 历史数据缓存 {year: history_dict}
        h2h_cache: 交锋记录缓存 {year: h2h_dict}
    """

    def __init__(self, df: pd.DataFrame):
        """
        初始化特征工程类

        Args:
            df: 包含比赛数据的数据框
        """
        self.df = df.copy()
        self.df = self.df.sort_values("年份").reset_index(drop=True)

        self.stage_rank = {
            "决赛": 100,
            "三四名决赛": 80,
            "半决赛": 60,
            "1/4决赛": 40,
            "1/8决赛": 20,
            "小组赛": 10
        }

        self.history_cache: Dict[int, Dict[str, Any]] = {}
        self.h2h_cache: Dict[int, Dict[tuple, Dict[str, int]]] = {}

        self._preprocess_team_data()

    def _preprocess_team_data(self):
        """
        预处理队伍数据，构建基础数据结构

        提前将主队和客队数据统一处理，减少重复计算
        """
        home_teams = self.df[["年份", "主队名称", "主队进球", "客队进球", "半场主队进球", "半场客队进球", "阶段", "胜负结果"]].copy()
        home_teams.rename(columns={
            "主队名称": "队伍名称",
            "主队进球": "进球数",
            "客队进球": "失球数",
            "半场主队进球": "半场进球数",
            "半场客队进球": "半场失球数"
        }, inplace=True)

        away_teams = self.df[["年份", "客队名称", "客队进球", "主队进球", "半场客队进球", "半场主队进球", "阶段", "胜负结果"]].copy()
        away_teams.rename(columns={
            "客队名称": "队伍名称",
            "客队进球": "进球数",
            "主队进球": "失球数",
            "半场客队进球": "半场进球数",
            "半场主队进球": "半场失球数"
        }, inplace=True)

        self.all_teams = pd.concat([home_teams, away_teams], ignore_index=True)

        logger.debug("队伍数据预处理完成")

    def _get_best_result(self, stage):

        result_map = {
            "决赛": "冠军/亚军",
            "半决赛": "四强",
            "三四名决赛": "三四名",
            "1/4决赛": "八强",
            "1/8决赛": "十六强",
            "小组赛": "小组赛"
        }
        return result_map.get(stage, stage)

    def _calculate_history_up_to_year(self, year: int) -> Dict[str, Any]:
        """
        计算指定年份之前所有队伍的历史数据（带缓存）

        Args:
            year: 截止年份（不包含）

        Returns:
            队伍历史数据字典 {team_name: {stat: value}}
        """
        if year in self.history_cache:
            logger.debug(f"使用缓存: 年份 {year}")
            return self.history_cache[year]

        prev_data = self.df[self.df["年份"] < year]

        if len(prev_data) == 0:
            self.history_cache[year] = {}
            return {}

        prev_team_data = self.all_teams[self.all_teams["年份"] < year]

        team_history = prev_team_data.groupby("队伍名称").agg(
            历史参赛次数=("年份", "nunique"),
            历史比赛场次=("年份", "count"),
            历史总进球=("进球数", "sum"),
            历史总失球=("失球数", "sum"),
            历史总半场进球=("半场进球数", "sum"),
            历史最高阶段=("阶段", lambda x: max(x, key=lambda s: self.stage_rank.get(s, 0)))
        ).reset_index()

        team_history["历史场均进球"] = team_history["历史总进球"] / team_history["历史比赛场次"]
        team_history["历史场均失球"] = team_history["历史总失球"] / team_history["历史比赛场次"]
        team_history["历史净胜球"] = team_history["历史总进球"] - team_history["历史总失球"]
        team_history["历史场均净胜球"] = team_history["历史场均进球"] - team_history["历史场均失球"]
        team_history["历史场均半场进球"] = team_history["历史总半场进球"] / team_history["历史比赛场次"]

        knockout_data = prev_team_data[prev_team_data["阶段"] != "小组赛"]
        group_stage_data = prev_team_data[prev_team_data["阶段"] == "小组赛"]

        knockout_win_rate = knockout_data[knockout_data["胜负结果"] == "主队胜"].groupby("队伍名称").size().reset_index(name="淘汰赛胜场").set_index("队伍名称")
        knockout_total = knockout_data.groupby("队伍名称").size().reset_index(name="淘汰赛总场次").set_index("队伍名称")

        group_win_rate = group_stage_data[group_stage_data["胜负结果"] == "主队胜"].groupby("队伍名称").size().reset_index(name="小组赛胜场").set_index("队伍名称")
        group_total = group_stage_data.groupby("队伍名称").size().reset_index(name="小组赛总场次").set_index("队伍名称")

        team_history = team_history.set_index("队伍名称")
        team_history["淘汰赛胜场"] = knockout_win_rate["淘汰赛胜场"].reindex(team_history.index).fillna(0)
        team_history["淘汰赛总场次"] = knockout_total["淘汰赛总场次"].reindex(team_history.index).fillna(0)
        team_history["小组赛胜场"] = group_win_rate["小组赛胜场"].reindex(team_history.index).fillna(0)
        team_history["小组赛总场次"] = group_total["小组赛总场次"].reindex(team_history.index).fillna(0)

        team_history["历史淘汰赛胜率"] = team_history["淘汰赛胜场"] / team_history["淘汰赛总场次"].replace(0, np.nan)
        team_history["历史小组赛胜率"] = team_history["小组赛胜场"] / team_history["小组赛总场次"].replace(0, np.nan)

        team_history["历史淘汰赛胜率"] = team_history["历史淘汰赛胜率"].fillna(0)
        team_history["历史小组赛胜率"] = team_history["历史小组赛胜率"].fillna(0)

        half_win_count = prev_team_data[prev_team_data["半场进球数"] > prev_team_data["半场失球数"]].groupby("队伍名称").size().reset_index(name="半场胜场").set_index("队伍名称")
        team_history["半场胜场"] = half_win_count["半场胜场"].reindex(team_history.index).fillna(0)
        team_history["历史半场胜率"] = team_history["半场胜场"] / team_history["历史比赛场次"]

        team_history = team_history.reset_index()

        team_history["历史最佳成绩"] = team_history["历史最高阶段"].map(self._get_best_result)
        team_history["历史成绩排名"] = team_history["历史最高阶段"].map(self.stage_rank)

        recent_years = sorted(prev_data["年份"].unique())[-3:]
        recent_data = prev_team_data[prev_team_data["年份"].isin(recent_years)]

        if len(recent_data) > 0:
            recent_stats = recent_data.groupby("队伍名称").agg(
                近3届比赛场次=("年份", "count"),
                近3届总进球=("进球数", "sum"),
                近3届总失球=("失球数", "sum")
            ).reset_index()

            recent_stats["近3届场均进球"] = recent_stats["近3届总进球"] / recent_stats["近3届比赛场次"]
            recent_stats["近3届胜率"] = recent_data[recent_data["胜负结果"] == "主队胜"].groupby("队伍名称").size().reset_index(name="近3届胜场").set_index("队伍名称").reindex(recent_stats["队伍名称"]).fillna(0).values.flatten() / recent_stats["近3届比赛场次"]

            team_history = team_history.merge(
                recent_stats[["队伍名称", "近3届场均进球", "近3届胜率"]],
                on="队伍名称",
                how="left"
            ).fillna(0)
        else:
            team_history["近3届场均进球"] = 0
            team_history["近3届胜率"] = 0

        result = team_history.set_index("队伍名称").to_dict("index")
        self.history_cache[year] = result

        logger.debug(f"计算完成并缓存: 年份 {year}, 队伍数量 {len(result)}")

        return result

    def _calculate_head_to_head(self, year: int) -> Dict[tuple, Dict[str, int]]:
        """
        计算指定年份之前所有队伍之间的交锋记录（带缓存）

        Args:
            year: 截止年份（不包含）

        Returns:
            交锋记录字典 {(home_team, away_team): {胜场, 平局, 负场, 净胜球}}
        """
        if year in self.h2h_cache:
            logger.debug(f"使用缓存: 交锋记录 年份 {year}")
            return self.h2h_cache[year]

        prev_data = self.df[self.df["年份"] < year]

        if len(prev_data) == 0:
            self.h2h_cache[year] = {}
            return {}

        h2h_records: Dict[tuple, Dict[str, int]] = {}

        for _, row in prev_data.iterrows():
            home_team = row["主队名称"]
            away_team = row["客队名称"]
            home_goals = row["主队进球"]
            away_goals = row["客队进球"]

            key1 = (home_team, away_team)
            key2 = (away_team, home_team)

            if key1 not in h2h_records:
                h2h_records[key1] = {"胜场": 0, "平局": 0, "负场": 0, "净胜球": 0}
            if key2 not in h2h_records:
                h2h_records[key2] = {"胜场": 0, "平局": 0, "负场": 0, "净胜球": 0}

            if home_goals > away_goals:
                h2h_records[key1]["胜场"] += 1
                h2h_records[key1]["净胜球"] += (home_goals - away_goals)
                h2h_records[key2]["负场"] += 1
                h2h_records[key2]["净胜球"] += (away_goals - home_goals)
            elif home_goals == away_goals:
                h2h_records[key1]["平局"] += 1
                h2h_records[key2]["平局"] += 1
            else:
                h2h_records[key1]["负场"] += 1
                h2h_records[key1]["净胜球"] += (home_goals - away_goals)
                h2h_records[key2]["胜场"] += 1
                h2h_records[key2]["净胜球"] += (away_goals - home_goals)

        self.h2h_cache[year] = h2h_records

        logger.debug(f"计算完成并缓存: 交锋记录 年份 {year}, 交锋对数 {len(h2h_records)}")

        return h2h_records

    def build_feature_matrix(self):

        print("\n" + "=" * 60)
        print("Step 1: 构建特征矩阵（无数据泄漏）")
        print("=" * 60)
        print("说明: 对于每场比赛，仅使用该比赛年份之前的数据计算队伍历史特征")
        print("      这确保了模型训练时不会看到未来数据，避免数据泄漏")

        features = []

        years = sorted(self.df["年份"].unique())

        for year in years:
            history_dict = self._calculate_history_up_to_year(year)
            h2h_dict = self._calculate_head_to_head(year)

            year_matches = self.df[self.df["年份"] == year].copy()

            for _, row in year_matches.iterrows():
                home_team = row["主队名称"]
                away_team = row["客队名称"]

                home_hist = history_dict.get(home_team, {})
                away_hist = history_dict.get(away_team, {})

                h2h_key = (home_team, away_team)
                h2h_record = h2h_dict.get(h2h_key, {"胜场": 0, "平局": 0, "负场": 0, "净胜球": 0})

                default_hist = {
                    "历史参赛次数": 0,
                    "历史比赛场次": 0,
                    "历史总进球": 0,
                    "历史总失球": 0,
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

                home_hist = {**default_hist, **home_hist}
                away_hist = {**default_hist, **away_hist}

                feature_row = {
                    "年份": year,
                    "主队名称": home_team,
                    "客队名称": away_team,
                    "阶段": row["阶段"],
                    "阶段类型": 0 if row["阶段"] == "小组赛" else 1,
                    "参赛次数差": home_hist["历史参赛次数"] - away_hist["历史参赛次数"],
                    "比赛场次差": home_hist["历史比赛场次"] - away_hist["历史比赛场次"],
                    "场均进球差": home_hist["历史场均进球"] - away_hist["历史场均进球"],
                    "场均失球差": home_hist["历史场均失球"] - away_hist["历史场均失球"],
                    "净胜球差": home_hist["历史净胜球"] - away_hist["历史净胜球"],
                    "成绩排名差": home_hist["历史成绩排名"] - away_hist["历史成绩排名"],
                    "近3届场均进球差": home_hist["近3届场均进球"] - away_hist["近3届场均进球"],
                    "近3届胜率差": home_hist["近3届胜率"] - away_hist["近3届胜率"],
                    "场均净胜球差": home_hist["历史场均净胜球"] - away_hist["历史场均净胜球"],
                    "淘汰赛胜率差": home_hist["历史淘汰赛胜率"] - away_hist["历史淘汰赛胜率"],
                    "小组赛胜率差": home_hist["历史小组赛胜率"] - away_hist["历史小组赛胜率"],
                    "场均半场进球差": home_hist["历史场均半场进球"] - away_hist["历史场均半场进球"],
                    "半场胜率差": home_hist["历史半场胜率"] - away_hist["历史半场胜率"],
                    "交锋胜场": h2h_record["胜场"],
                    "交锋平局": h2h_record["平局"],
                    "交锋负场": h2h_record["负场"],
                    "交锋净胜球": h2h_record["净胜球"],
                    "交锋总场次": h2h_record["胜场"] + h2h_record["平局"] + h2h_record["负场"],
                    "交锋胜率": h2h_record["胜场"] / (h2h_record["胜场"] + h2h_record["平局"] + h2h_record["负场"]) if (h2h_record["胜场"] + h2h_record["平局"] + h2h_record["负场"]) > 0 else 0,
                    "胜负结果": row["胜负结果"]
                }

                features.append(feature_row)

        feature_matrix = pd.DataFrame(features)

        feature_matrix["胜负编码"] = feature_matrix["胜负结果"].map({
            "主队胜": 1,
            "平局": 0,
            "客队胜": -1
        })

        self.feature_matrix = feature_matrix

        feature_cols = [col for col in feature_matrix.columns
                       if col not in ["年份", "主队名称", "客队名称", "胜负结果"]]

        print(f"\n特征矩阵形状：{feature_matrix.shape}")
        print(f"特征数量：{len(feature_cols)}")
        print(f"\n特征列表：")
        for i, col in enumerate(feature_cols, 1):
            print(f"  {i}. {col}")

        print(f"\n特征矩阵预览（前5行）：")
        print("-" * 80)
        print(feature_matrix.head().to_string(index=False))

        return feature_matrix

    def calculate_team_history(self):

        print("\n" + "=" * 60)
        print("Step 2: 计算每支队伍完整历史指标")
        print("=" * 60)

        home_teams = self.df[["年份", "主队名称", "主队进球", "客队进球", "阶段", "胜负结果"]].copy()
        home_teams.rename(columns={
            "主队名称": "队伍名称",
            "主队进球": "进球数",
            "客队进球": "失球数"
        }, inplace=True)

        away_teams = self.df[["年份", "客队名称", "客队进球", "主队进球", "阶段", "胜负结果"]].copy()
        away_teams.rename(columns={
            "客队名称": "队伍名称",
            "客队进球": "进球数",
            "主队进球": "失球数"
        }, inplace=True)

        all_teams = pd.concat([home_teams, away_teams], ignore_index=True)

        team_history = all_teams.groupby("队伍名称").agg(
            历史参赛次数=("年份", "nunique"),
            历史比赛场次=("年份", "count"),
            历史总进球=("进球数", "sum"),
            历史总失球=("失球数", "sum"),
            历史最高阶段=("阶段", lambda x: max(x, key=lambda s: self.stage_rank.get(s, 0)))
        ).reset_index()

        team_history["历史场均进球"] = team_history["历史总进球"] / team_history["历史比赛场次"]
        team_history["历史场均失球"] = team_history["历史总失球"] / team_history["历史比赛场次"]
        team_history["历史净胜球"] = team_history["历史总进球"] - team_history["历史总失球"]
        team_history["历史场均净胜球"] = team_history["历史场均进球"] - team_history["历史场均失球"]

        team_history["历史最佳成绩"] = team_history["历史最高阶段"].map(self._get_best_result)
        team_history["历史成绩排名"] = team_history["历史最高阶段"].map(self.stage_rank)

        recent_years = sorted(self.df["年份"].unique())[-3:]
        recent_data = all_teams[all_teams["年份"].isin(recent_years)]

        recent_stats = recent_data.groupby("队伍名称").agg(
            近3届比赛场次=("年份", "count"),
            近3届总进球=("进球数", "sum"),
            近3届总失球=("失球数", "sum")
        ).reset_index()

        recent_stats["近3届场均进球"] = recent_stats["近3届总进球"] / recent_stats["近3届比赛场次"]
        recent_stats["近3届胜率"] = recent_data[recent_data["胜负结果"] == "主队胜"].groupby("队伍名称").size().reset_index(name="近3届胜场").set_index("队伍名称").reindex(recent_stats["队伍名称"]).fillna(0).values.flatten() / recent_stats["近3届比赛场次"]

        team_history = team_history.merge(
            recent_stats[["队伍名称", "近3届场均进球", "近3届胜率"]],
            on="队伍名称",
            how="left"
        ).fillna(0)

        team_history = team_history.sort_values("历史参赛次数", ascending=False).reset_index(drop=True)

        self.team_history = team_history

        print(f"\n共统计 {len(team_history)} 支队伍的历史数据")
        print(f"\n历史指标字段：")
        print(f"  - 历史参赛次数: 队伍参加世界杯的届数")
        print(f"  - 历史比赛场次: 队伍参加的总比赛场次")
        print(f"  - 历史总进球: 队伍历史总进球数")
        print(f"  - 历史总失球: 队伍历史总失球数")
        print(f"  - 历史最高阶段: 队伍历史最佳成绩阶段")
        print(f"  - 历史场均进球: 场均进球数")
        print(f"  - 历史场均失球: 场均失球数")
        print(f"  - 历史净胜球: 总进球-总失球")
        print(f"  - 历史场均净胜球: 场均进球-场均失球")
        print(f"  - 历史最佳成绩: 最佳成绩描述")
        print(f"  - 历史成绩排名: 成绩排名数值(越大越好)")
        print(f"  - 近3届场均进球: 最近3届世界杯场均进球数")
        print(f"  - 近3届胜率: 最近3届世界杯胜率")

        print(f"\n历史指标前10名队伍：")
        print("-" * 80)
        print(team_history.head(10).to_string(index=False))

        return team_history

    def analyze_correlation(self):

        print("\n" + "=" * 60)
        print("Step 3: 计算各特征与比赛胜负的相关性")
        print("=" * 60)

        numeric_features = [
            "参赛次数差", "比赛场次差", "场均进球差", "场均失球差",
            "净胜球差", "成绩排名差", "近3届场均进球差", "近3届胜率差",
            "场均净胜球差", "淘汰赛胜率差", "小组赛胜率差",
            "场均半场进球差", "半场胜率差", "阶段类型",
            "交锋胜场", "交锋平局", "交锋负场", "交锋净胜球",
            "交锋总场次", "交锋胜率"
        ]

        correlations = {}
        for feature in numeric_features:
            if feature in self.feature_matrix.columns:
                corr_pearson = self.feature_matrix[feature].corr(
                    self.feature_matrix["胜负编码"], method="pearson"
                )
                corr_spearman = self.feature_matrix[feature].corr(
                    self.feature_matrix["胜负编码"], method="spearman"
                )
                correlations[feature] = {
                    "Pearson相关系数": corr_pearson,
                    "Spearman相关系数": corr_spearman
                }

        corr_df = pd.DataFrame(correlations).T
        corr_df["平均相关系数"] = (corr_df["Pearson相关系数"] + corr_df["Spearman相关系数"]) / 2
        corr_df = corr_df.sort_values("平均相关系数", ascending=False)

        self.correlation_df = corr_df

        print(f"\n各特征与比赛胜负的相关性：")
        print("-" * 80)
        print(corr_df.to_string())

        print(f"\n相关性解释：")
        print("-" * 40)
        print("相关系数范围: [-1, 1]")
        print("  |r| >= 0.8: 极强相关")
        print("  0.6 <= |r| < 0.8: 强相关")
        print("  0.4 <= |r| < 0.6: 中等相关")
        print("  0.2 <= |r| < 0.4: 弱相关")
        print("  |r| < 0.2: 极弱相关或无相关")

        return corr_df

    def print_key_features(self, threshold=0.2):

        print("\n" + "=" * 60)
        print("Step 4: 关键高相关特征")
        print("=" * 60)

        high_corr = self.correlation_df[
            self.correlation_df["平均相关系数"].abs() >= threshold
        ]

        print(f"\n平均相关系数绝对值 >= {threshold} 的关键特征：")
        print("-" * 80)
        print(high_corr.to_string())

        print(f"\n关键特征解读：")
        print("-" * 40)
        for feature, row in high_corr.iterrows():
            corr = row["平均相关系数"]
            strength = self._interpret_correlation(corr)
            direction = "正相关(值越大主队越可能胜)" if corr > 0 else "负相关(值越大客队越可能胜)"
            print(f"  {feature}:")
            print(f"    相关系数: {corr:.4f} ({strength})")
            print(f"    方向: {direction}")

        self.key_features = high_corr

        return high_corr

    def _interpret_correlation(self, corr):

        abs_corr = abs(corr)
        if abs_corr >= 0.8:
            return "极强相关"
        elif abs_corr >= 0.6:
            return "强相关"
        elif abs_corr >= 0.4:
            return "中等相关"
        elif abs_corr >= 0.2:
            return "弱相关"
        else:
            return "极弱相关或无相关"

    def save_results(self):

        print("\n" + "=" * 60)
        print("Step 5: 保存结果")
        print("=" * 60)

        history_path = TABLE_DIR / "task3_team_history.csv"
        self.team_history.to_csv(history_path, index=False, encoding="utf-8-sig")
        print(f"✓ 队伍历史指标已保存到: {history_path}")

        matrix_path = TABLE_DIR / "task3_feature_matrix.csv"
        self.feature_matrix.to_csv(matrix_path, index=False, encoding="utf-8-sig")
        print(f"✓ 特征矩阵已保存到: {matrix_path}")

        corr_path = TABLE_DIR / "task3_correlation.csv"
        self.correlation_df.to_csv(corr_path, encoding="utf-8-sig")
        print(f"✓ 相关性分析结果已保存到: {corr_path}")

        if hasattr(self, "key_features"):
            key_path = TABLE_DIR / "task3_key_features.csv"
            self.key_features.to_csv(key_path, encoding="utf-8-sig")
            print(f"✓ 关键特征已保存到: {key_path}")

        self.generate_report()

        return True

    def generate_report(self):

        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("Task3 第一小问：队伍历史特征工程报告")
        report_lines.append("=" * 70)
        report_lines.append("")

        report_lines.append("一、数据概况")
        report_lines.append("-" * 40)
        report_lines.append(f"原始数据样本数: {len(self.df)}")
        report_lines.append(f"统计队伍数量: {len(self.team_history)}")
        report_lines.append(f"特征矩阵样本数: {len(self.feature_matrix)}")
        report_lines.append("")

        report_lines.append("二、特征工程方法说明")
        report_lines.append("-" * 40)
        report_lines.append("数据泄漏处理:")
        report_lines.append("  为避免数据泄漏，对于每场比赛，仅使用该比赛年份之前的历史数据")
        report_lines.append("  计算队伍特征。例如：2014年的比赛只使用1930-2010年的数据。")
        report_lines.append("  这确保了模型训练时不会看到未来数据，保证了模型的泛化能力。")
        report_lines.append("")

        report_lines.append("三、队伍历史指标说明")
        report_lines.append("-" * 40)
        report_lines.append("1. 历史参赛次数: 队伍参加世界杯的届数")
        report_lines.append("2. 历史比赛场次: 队伍参加的总比赛场次")
        report_lines.append("3. 历史总进球: 队伍历史总进球数")
        report_lines.append("4. 历史总失球: 队伍历史总失球数")
        report_lines.append("5. 历史最高阶段: 队伍历史最佳成绩阶段")
        report_lines.append("6. 历史场均进球: 场均进球数 = 历史总进球 / 历史比赛场次")
        report_lines.append("7. 历史场均失球: 场均失球数 = 历史总失球 / 历史比赛场次")
        report_lines.append("8. 历史净胜球: 历史总进球 - 历史总失球")
        report_lines.append("9. 历史场均净胜球: 场均进球 - 场均失球")
        report_lines.append("10. 历史最佳成绩: 最佳成绩描述(冠军/亚军、四强、八强等)")
        report_lines.append("11. 历史成绩排名: 成绩排名数值(1-6，越大越好)")
        report_lines.append("")

        report_lines.append("四、历史指标前10名队伍")
        report_lines.append("-" * 40)
        report_lines.append(self.team_history.head(10).to_string(index=False))
        report_lines.append("")

        report_lines.append("五、特征矩阵说明")
        report_lines.append("-" * 40)
        report_lines.append(f"特征矩阵形状: {self.feature_matrix.shape}")
        report_lines.append("主队特征: 主队历史参赛次数、场均进球、场均失球等")
        report_lines.append("客队特征: 客队历史参赛次数、场均进球、场均失球等")
        report_lines.append("差值特征: 主客队各项指标的差值")
        report_lines.append("目标变量: 胜负编码(1=主队胜, 0=平局, -1=客队胜)")
        report_lines.append("")

        report_lines.append("六、相关性分析结果")
        report_lines.append("-" * 40)
        report_lines.append(self.correlation_df.to_string())
        report_lines.append("")

        report_lines.append("相关性解释:")
        report_lines.append("  相关系数范围: [-1, 1]")
        report_lines.append("  |r| >= 0.8: 极强相关")
        report_lines.append("  0.6 <= |r| < 0.8: 强相关")
        report_lines.append("  0.4 <= |r| < 0.6: 中等相关")
        report_lines.append("  0.2 <= |r| < 0.4: 弱相关")
        report_lines.append("  |r| < 0.2: 极弱相关或无相关")
        report_lines.append("")

        if hasattr(self, "key_features"):
            report_lines.append("七、关键高相关特征")
            report_lines.append("-" * 40)
            report_lines.append(self.key_features.to_string())
            report_lines.append("")
            report_lines.append("关键特征解读:")
            for feature, row in self.key_features.iterrows():
                corr = row["平均相关系数"]
                strength = self._interpret_correlation(corr)
                direction = "正相关(值越大主队越可能胜)" if corr > 0 else "负相关(值越大客队越可能胜)"
                report_lines.append(f"  {feature}: {corr:.4f} ({strength}, {direction})")
            report_lines.append("")

        report_lines.append("八、方法局限性")
        report_lines.append("-" * 40)
        report_lines.append("1. 历史数据假设: 假设队伍历史表现能够预测未来表现，")
        report_lines.append("   但队伍实力可能随时间变化(如教练更替、球员换代)。")
        report_lines.append("2. 样本量限制: 早期世界杯参赛队伍较少，历史数据有限。")
        report_lines.append("3. 新队伍问题: 首次参赛的队伍没有历史数据，使用默认值。")
        report_lines.append("4. 赛事规则变化: 不同年份世界杯的赛制、规则可能不同。")
        report_lines.append("")

        report_lines.append("=" * 70)
        report_lines.append("报告生成完毕")
        report_lines.append("=" * 70)

        report = "\n".join(report_lines)

        report_path = REPORT_DIR / "task3_feature_engineering_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"✓ 特征工程报告已保存到: {report_path}")

        return report_path

    def run(self):

        self.build_feature_matrix()
        self.calculate_team_history()
        self.analyze_correlation()
        self.print_key_features()
        self.save_results()

        return self.feature_matrix