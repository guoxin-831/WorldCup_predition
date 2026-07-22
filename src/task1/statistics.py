"""
statistics.py
Task1 描述统计分析模块
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import TABLE_DIR


class StatisticsAnalyzer:

    def __init__(self, df):
        self.df = df.copy()

    def descriptive_statistics(self):

        cols = [
            "总进球数",
            "场均进球",
            "参赛队伍数量",
            "总比赛场次",
            "总观众人数"
        ]

        stats = pd.DataFrame(index=cols)

        stats["Mean"] = self.df[cols].mean()
        stats["Median"] = self.df[cols].median()
        stats["Std"] = self.df[cols].std()
        stats["Variance"] = self.df[cols].var()
        stats["Min"] = self.df[cols].min()
        stats["Max"] = self.df[cols].max()

        stats["Range"] = (
            stats["Max"] -
            stats["Min"]
        )

        stats["CV"] = (
            stats["Std"] /
            stats["Mean"]
        )

        stats["Skewness"] = self.df[cols].skew()

        stats["Kurtosis"] = self.df[cols].kurt()

        stats = stats.round(4)

        stats.to_csv(
            TABLE_DIR / "descriptive_statistics.csv",
            encoding="utf-8-sig"
        )

        return stats

    def trend_analysis(self):

        goal = self.df["总进球数"]

        year = self.df["年份"]

        diff = goal.diff()

        increase_year = year.iloc[
            diff.idxmax()
        ]

        decrease_year = year.iloc[
            diff.idxmin()
        ]

        slope = np.polyfit(
            year,
            goal,
            1
        )[0]

        trend = {

            "平均总进球": goal.mean(),

            "最高进球":

                goal.max(),

            "最低进球":

                goal.min(),

            "增长最快年份":

                int(increase_year),

            "下降最快年份":

                int(decrease_year),

            "趋势斜率":

                slope

        }

        trend_df = pd.DataFrame(
            trend,
            index=[0]
        )

        trend_df.to_csv(
            TABLE_DIR / "trend_analysis.csv",
            index=False,
            encoding="utf-8-sig"
        )

        return trend

    def correlation_analysis(self):

        cols = [

            "总进球数",

            "场均进球",

            "参赛队伍数量",

            "总比赛场次",

            "总观众人数"

        ]

        corr = self.df[
            cols
        ].corr(
            method="pearson"
        )

        corr.to_csv(

            TABLE_DIR / "correlation_matrix.csv",

            encoding="utf-8-sig"

        )

        return corr

    def quality_report(self):

        quality = pd.DataFrame({

            "缺失值":

                self.df.isnull().sum(),

            "数据类型":

                self.df.dtypes.astype(str)

        })

        quality["重复值"] = self.df.duplicated().sum()

        quality.to_csv(

            TABLE_DIR / "quality_report.csv",

            encoding="utf-8-sig"

        )

        return quality

    def run(self):

        result = {

            "statistics":

                self.descriptive_statistics(),

            "trend":

                self.trend_analysis(),

            "correlation":

                self.correlation_analysis(),

            "quality":

                self.quality_report()

        }

        print("✓ 描述统计完成")

        return result