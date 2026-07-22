"""
statistics.py
==========================
Task2 统计分析模块
==========================
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from src.config import TABLE_DIR, REPORT_DIR


class MatchStatistics:
    """
    比赛数据统计分析
    """

    def __init__(self, df):

        self.df = df

        self.stage_order = ["小组赛", "1/8决赛", "1/4决赛", "半决赛", "三四名决赛", "决赛"]

    def calculate_match_stats(self):

        print("\n" + "=" * 60)
        print("计算比赛统计量")
        print("=" * 60)

        stats = {}

        stats["场均进球"] = self.df["总进球数"].mean()
        stats["进球中位数"] = self.df["总进球数"].median()
        stats["进球最大值"] = self.df["总进球数"].max()
        stats["进球最小值"] = self.df["总进球数"].min()
        stats["进球标准差"] = self.df["总进球数"].std()

        stats["场均主队进球"] = self.df["主队进球"].mean()
        stats["场均客队进球"] = self.df["客队进球"].mean()
        stats["主队胜率"] = (self.df["胜负结果"] == "主队胜").mean()
        stats["客队胜率"] = (self.df["胜负结果"] == "客队胜").mean()
        stats["平局率"] = (self.df["胜负结果"] == "平局").mean()

        stats["场均观众人数"] = self.df["观众人数"].mean()
        stats["观众人数中位数"] = self.df["观众人数"].median()

        stats["半场进球占比"] = self.df["半场总进球"].sum() / self.df["总进球数"].sum()

        self.stats = pd.DataFrame([stats])

        print("\n比赛统计结果：")
        print("-" * 40)
        print(self.stats.to_string(index=False))

        return self.stats

    def yearly_stats(self):

        yearly = self.df.groupby("年份").agg(
            比赛场次=("比赛ID", "count"),
            总进球数=("总进球数", "sum"),
            场均进球=("总进球数", "mean"),
            主队进球=("主队进球", "sum"),
            客队进球=("客队进球", "sum"),
            主队胜场=("胜负结果", lambda x: (x == "主队胜").sum()),
            客队胜场=("胜负结果", lambda x: (x == "客队胜").sum()),
            平局场数=("胜负结果", lambda x: (x == "平局").sum()),
            场均观众=("观众人数", "mean"),
            半场进球=("半场总进球", "sum")
        ).reset_index()

        yearly["主队胜率"] = yearly["主队胜场"] / yearly["比赛场次"]
        yearly["客队胜率"] = yearly["客队胜场"] / yearly["比赛场次"]
        yearly["平局率"] = yearly["平局场数"] / yearly["比赛场次"]
        yearly["半场进球占比"] = yearly["半场进球"] / yearly["总进球数"]

        self.yearly_stats_df = yearly

        print("\n年度统计结果：")
        print("-" * 40)
        print(yearly.to_string(index=False))

        return yearly

    def stage_stats(self):

        print("\n" + "=" * 60)
        print("按赛事阶段分组统计")
        print("=" * 60)

        stage = self.df.groupby("阶段").agg(
            比赛场次=("比赛ID", "count"),
            总进球数=("总进球数", "sum"),
            场均进球=("总进球数", "mean"),
            进球中位数=("总进球数", "median"),
            进球标准差=("总进球数", "std"),
            场均主队进球=("主队进球", "mean"),
            场均客队进球=("客队进球", "mean"),
            主队胜场=("胜负结果", lambda x: (x == "主队胜").sum()),
            客队胜场=("胜负结果", lambda x: (x == "客队胜").sum()),
            平局场数=("胜负结果", lambda x: (x == "平局").sum()),
            场均观众=("观众人数", "mean"),
            半场进球=("半场总进球", "sum"),
            半场场均进球=("半场总进球", "mean")
        ).reset_index()

        stage["主队胜率"] = stage["主队胜场"] / stage["比赛场次"]
        stage["客队胜率"] = stage["客队胜场"] / stage["比赛场次"]
        stage["平局率"] = stage["平局场数"] / stage["比赛场次"]
        stage["半场进球占比"] = stage["半场进球"] / stage["总进球数"]

        stage["阶段排序"] = stage["阶段"].apply(
            lambda x: self.stage_order.index(x) if x in self.stage_order else len(self.stage_order)
        )
        stage.sort_values("阶段排序", inplace=True)
        stage.drop("阶段排序", axis=1, inplace=True)

        self.stage_stats_df = stage

        print("\n阶段统计结果（场均进球排序）：")
        print("-" * 40)
        print(stage.to_string(index=False))

        avg_goals_order = stage.sort_values("场均进球", ascending=False)
        print("\n按场均进球从高到低排序：")
        print("-" * 40)
        for _, row in avg_goals_order.iterrows():
            print(f"{row['阶段']}: {row['场均进球']:.2f} 球/场")

        return stage

    def goal_distribution(self):

        dist = self.df["总进球数"].value_counts().sort_index().reset_index()
        dist.columns = ["进球数", "场次"]
        self.goal_dist_df = dist

        print("\n进球数分布：")
        print("-" * 40)
        print(dist.to_string(index=False))

        return dist

    def half_time_correlation(self):

        print("\n" + "=" * 60)
        print("上半场进球与全场进球相关性分析")
        print("=" * 60)

        valid_df = self.df.dropna(subset=["半场总进球", "总进球数"])

        x = valid_df["半场总进球"]
        y = valid_df["总进球数"]

        pearson_corr, pearson_p = pearsonr(x, y)
        spearman_corr, spearman_p = spearmanr(x, y)

        self.correlation_results = {
            "Pearson相关系数": pearson_corr,
            "Pearson p值": pearson_p,
            "Spearman相关系数": spearman_corr,
            "Spearman p值": spearman_p,
            "样本数量": len(valid_df)
        }

        print(f"\n分析样本数量：{len(valid_df)}")
        print(f"Pearson相关系数: {pearson_corr:.4f} (p值: {pearson_p:.6f})")
        print(f"Spearman相关系数: {spearman_corr:.4f} (p值: {spearman_p:.6f})")

        correlation_strength = self._interpret_correlation(pearson_corr)

        print(f"\n相关性强弱结论：{correlation_strength}")

        return self.correlation_results, correlation_strength

    def _interpret_correlation(self, corr):

        abs_corr = abs(corr)
        if abs_corr >= 0.8:
            return "极强正相关" if corr > 0 else "极强负相关"
        elif abs_corr >= 0.6:
            return "强正相关" if corr > 0 else "强负相关"
        elif abs_corr >= 0.4:
            return "中等正相关" if corr > 0 else "中等负相关"
        elif abs_corr >= 0.2:
            return "弱正相关" if corr > 0 else "弱负相关"
        else:
            return "极弱相关或无相关"

    def save_statistics(self):

        float_cols_stats = ["场均进球", "进球中位数", "进球标准差",
                            "场均主队进球", "场均客队进球",
                            "主队胜率", "客队胜率", "平局率",
                            "场均观众人数", "观众人数中位数", "半场进球占比"]
        self.stats[float_cols_stats] = self.stats[float_cols_stats].round(4)

        stats_path = TABLE_DIR / "task2_match_statistics.csv"
        self.stats.to_csv(stats_path, index=False, encoding="utf-8-sig")
        print(f"\n✓ 比赛统计表格已保存到: {stats_path}")

        float_cols_yearly = ["场均进球", "场均观众",
                             "主队胜率", "客队胜率", "平局率", "半场进球占比"]
        self.yearly_stats_df[float_cols_yearly] = self.yearly_stats_df[float_cols_yearly].round(4)

        yearly_path = TABLE_DIR / "task2_yearly_statistics.csv"
        self.yearly_stats_df.to_csv(yearly_path, index=False, encoding="utf-8-sig")
        print(f"✓ 年度统计表格已保存到: {yearly_path}")

        float_cols_stage = ["场均进球", "进球中位数", "进球标准差",
                            "场均主队进球", "场均客队进球",
                            "主队胜率", "客队胜率", "平局率",
                            "场均观众", "半场进球占比"]
        self.stage_stats_df[float_cols_stage] = self.stage_stats_df[float_cols_stage].round(4)

        stage_path = TABLE_DIR / "task2_stage_statistics.csv"
        self.stage_stats_df.to_csv(stage_path, index=False, encoding="utf-8-sig")
        print(f"✓ 阶段统计表格已保存到: {stage_path}")

        dist_path = TABLE_DIR / "task2_goal_distribution.csv"
        self.goal_dist_df.to_csv(dist_path, index=False, encoding="utf-8-sig")
        print(f"✓ 进球分布表格已保存到: {dist_path}")

        if hasattr(self, "correlation_results"):
            corr_df = pd.DataFrame([self.correlation_results])
            float_cols_corr = ["Pearson相关系数", "Pearson p值",
                               "Spearman相关系数", "Spearman p值"]
            corr_df[float_cols_corr] = corr_df[float_cols_corr].round(4)
            corr_path = TABLE_DIR / "task2_half_time_correlation.csv"
            corr_df.to_csv(corr_path, index=False, encoding="utf-8-sig")
            print(f"✓ 相关性分析结果已保存到: {corr_path}")

        return [stats_path, yearly_path, stage_path, dist_path]

    def generate_report(self):

        print("\n" + "=" * 60)
        print("生成统计分析报告")
        print("=" * 60)

        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("Task2 世界杯比赛数据分析报告")
        report_lines.append("=" * 70)
        report_lines.append("")

        report_lines.append("一、数据概况")
        report_lines.append("-" * 30)
        report_lines.append(f"样本数量: {self.df.shape[0]}")
        report_lines.append(f"字段数量: {self.df.shape[1]}")
        report_lines.append(f"年份范围: {self.df['年份'].min()} ~ {self.df['年份'].max()}")
        report_lines.append("")

        report_lines.append("二、整体比赛统计")
        report_lines.append("-" * 30)
        report_lines.append(self.stats.to_string(index=False))
        report_lines.append("")

        report_lines.append("三、按赛事阶段分组统计")
        report_lines.append("-" * 30)
        report_lines.append(self.stage_stats_df.to_string(index=False))
        report_lines.append("")

        avg_goals_order = self.stage_stats_df.sort_values("场均进球", ascending=False)
        report_lines.append("场均进球排名：")
        for _, row in avg_goals_order.iterrows():
            report_lines.append(f"  {row['阶段']}: {row['场均进球']:.2f} 球/场")
        report_lines.append("")

        report_lines.append("四、上半场进球与全场进球相关性分析")
        report_lines.append("-" * 30)
        if hasattr(self, "correlation_results"):
            report_lines.append(f"分析样本数量: {self.correlation_results['样本数量']}")
            report_lines.append(f"Pearson相关系数: {self.correlation_results['Pearson相关系数']:.4f}")
            report_lines.append(f"Pearson p值: {self.correlation_results['Pearson p值']:.6f}")
            report_lines.append(f"Spearman相关系数: {self.correlation_results['Spearman相关系数']:.4f}")
            report_lines.append(f"Spearman p值: {self.correlation_results['Spearman p值']:.6f}")

            strength = self._interpret_correlation(self.correlation_results["Pearson相关系数"])
            report_lines.append(f"")
            report_lines.append(f"结论：上半场进球与全场进球呈{strength}")
            report_lines.append(f"说明：相关系数绝对值越大，说明上半场进球对全场进球的预测能力越强")

        report_lines.append("")
        report_lines.append("=" * 70)
        report_lines.append("报告生成完毕")
        report_lines.append("=" * 70)

        self.report = "\n".join(report_lines)

        report_path = REPORT_DIR / "task2_statistics_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(self.report)

        print(f"\n✓ 统计分析报告已保存到: {report_path}")

        return report_path

    def run(self):

        self.calculate_match_stats()
        self.yearly_stats()
        self.stage_stats()
        self.goal_distribution()
        self.half_time_correlation()
        self.save_statistics()
        self.generate_report()

        return self.stats


class PlayerStatistics:
    """
    球员数据统计分析
    """

    def __init__(self, df):

        self.df = df

    def top_scorers(self, top_n=10):

        scorers = self.df.groupby("球员姓名").agg(
            总进球=("进球数", "sum"),
            总助攻=("助攻数", "sum"),
            出场次数=("比赛ID", "nunique"),
            黄牌数=("黄牌数", "sum"),
            红牌数=("红牌数", "sum")
        ).sort_values("总进球", ascending=False).head(top_n)

        self.top_scorers_df = scorers

        print("\n射手榜前10：")
        print("-" * 40)
        print(scorers.to_string())

        return scorers

    def top_assistants(self, top_n=10):

        assistants = self.df.groupby("球员姓名").agg(
            总助攻=("助攻数", "sum"),
            总进球=("进球数", "sum"),
            出场次数=("比赛ID", "nunique")
        ).sort_values("总助攻", ascending=False).head(top_n)

        self.top_assistants_df = assistants

        print("\n助攻榜前10：")
        print("-" * 40)
        print(assistants.to_string())

        return assistants

    def team_stats(self):

        team_stats = self.df.groupby("球队缩写").agg(
            进球数=("进球数", "sum"),
            助攻数=("助攻数", "sum"),
            出场球员=("球员姓名", "nunique"),
            黄牌数=("黄牌数", "sum"),
            红牌数=("红牌数", "sum")
        ).sort_values("进球数", ascending=False)

        self.team_stats_df = team_stats

        print("\n球队统计：")
        print("-" * 40)
        print(team_stats.head(10).to_string())

        return team_stats

    def position_stats(self):

        position_stats = self.df.groupby("位置").agg(
            进球数=("进球数", "sum"),
            助攻数=("助攻数", "sum"),
            出场次数=("球员姓名", "nunique"),
            黄牌数=("黄牌数", "sum"),
            红牌数=("红牌数", "sum")
        ).sort_values("进球数", ascending=False)

        self.position_stats_df = position_stats

        print("\n位置统计：")
        print("-" * 40)
        print(position_stats.to_string())

        return position_stats

    def save_player_stats(self):

        scorers_path = TABLE_DIR / "task2_top_scorers.csv"
        self.top_scorers_df.to_csv(scorers_path, encoding="utf-8-sig")
        print(f"\n✓ 射手榜已保存到: {scorers_path}")

        assistants_path = TABLE_DIR / "task2_top_assistants.csv"
        self.top_assistants_df.to_csv(assistants_path, encoding="utf-8-sig")
        print(f"✓ 助攻榜已保存到: {assistants_path}")

        team_path = TABLE_DIR / "task2_team_stats.csv"
        self.team_stats_df.to_csv(team_path, encoding="utf-8-sig")
        print(f"✓ 球队统计已保存到: {team_path}")

        position_path = TABLE_DIR / "task2_position_stats.csv"
        self.position_stats_df.to_csv(position_path, encoding="utf-8-sig")
        print(f"✓ 位置统计已保存到: {position_path}")

        return [scorers_path, assistants_path, team_path, position_path]

    def run(self):

        self.top_scorers()
        self.top_assistants()
        self.team_stats()
        self.position_stats()
        self.save_player_stats()

        return self.top_scorers_df