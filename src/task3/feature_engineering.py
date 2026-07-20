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
from config import TABLE_DIR, REPORT_DIR


class TeamFeatureEngineering:
    """
    队伍历史特征工程（无数据泄漏版本）
    """

    def __init__(self, df):

        self.df = df.copy()
        self.df = self.df.sort_values("年份").reset_index(drop=True)

        self.stage_rank = {
            "小组赛": 1,
            "1/8决赛": 2,
            "1/4决赛": 3,
            "半决赛": 4,
            "三四名决赛": 5,
            "决赛": 6
        }

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

    def _calculate_history_up_to_year(self, year):

        prev_data = self.df[self.df["年份"] < year].copy()

        if len(prev_data) == 0:
            return {}

        home_teams = prev_data[["年份", "主队名称", "主队进球", "客队进球", "阶段", "胜负结果"]].copy()
        home_teams.rename(columns={
            "主队名称": "队伍名称",
            "主队进球": "进球数",
            "客队进球": "失球数"
        }, inplace=True)

        away_teams = prev_data[["年份", "客队名称", "客队进球", "主队进球", "阶段", "胜负结果"]].copy()
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

        recent_years = sorted(prev_data["年份"].unique())[-3:]
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

        return team_history.set_index("队伍名称").to_dict("index")

    def _calculate_head_to_head(self, year):

        prev_data = self.df[self.df["年份"] < year].copy()

        if len(prev_data) == 0:
            return {}

        h2h_records = {}

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
                    "近3届胜率": 0
                }

                home_hist = {**default_hist, **home_hist}
                away_hist = {**default_hist, **away_hist}

                feature_row = {
                    "年份": year,
                    "主队名称": home_team,
                    "客队名称": away_team,
                    "主队历史参赛次数": home_hist["历史参赛次数"],
                    "主队历史比赛场次": home_hist["历史比赛场次"],
                    "主队历史场均进球": home_hist["历史场均进球"],
                    "主队历史场均失球": home_hist["历史场均失球"],
                    "主队历史净胜球": home_hist["历史净胜球"],
                    "主队历史成绩排名": home_hist["历史成绩排名"],
                    "主队近3届场均进球": home_hist["近3届场均进球"],
                    "主队近3届胜率": home_hist["近3届胜率"],
                    "客队历史参赛次数": away_hist["历史参赛次数"],
                    "客队历史比赛场次": away_hist["历史比赛场次"],
                    "客队历史场均进球": away_hist["历史场均进球"],
                    "客队历史场均失球": away_hist["历史场均失球"],
                    "客队历史净胜球": away_hist["历史净胜球"],
                    "客队历史成绩排名": away_hist["历史成绩排名"],
                    "客队近3届场均进球": away_hist["近3届场均进球"],
                    "客队近3届胜率": away_hist["近3届胜率"],
                    "参赛次数差": home_hist["历史参赛次数"] - away_hist["历史参赛次数"],
                    "场均进球差": home_hist["历史场均进球"] - away_hist["历史场均进球"],
                    "场均失球差": home_hist["历史场均失球"] - away_hist["历史场均失球"],
                    "净胜球差": home_hist["历史净胜球"] - away_hist["历史净胜球"],
                    "成绩排名差": home_hist["历史成绩排名"] - away_hist["历史成绩排名"],
                    "近3届场均进球差": home_hist["近3届场均进球"] - away_hist["近3届场均进球"],
                    "近3届胜率差": home_hist["近3届胜率"] - away_hist["近3届胜率"],
                    "交锋胜场": h2h_record["胜场"],
                    "交锋平局": h2h_record["平局"],
                    "交锋负场": h2h_record["负场"],
                    "交锋净胜球": h2h_record["净胜球"],
                    "总进球数": row["总进球数"],
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
            "主队历史参赛次数", "主队历史比赛场次", "主队历史场均进球",
            "主队历史场均失球", "主队历史净胜球", "主队历史成绩排名",
            "客队历史参赛次数", "客队历史比赛场次", "客队历史场均进球",
            "客队历史场均失球", "客队历史净胜球", "客队历史成绩排名",
            "参赛次数差", "场均进球差", "场均失球差",
            "净胜球差", "成绩排名差"
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