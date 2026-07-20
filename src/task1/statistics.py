"""
statistics.py
==========================
Task1 统计分析模块
==========================
"""

import pandas as pd
import numpy as np

from config import TABLE_DIR


class StatisticsAnalyzer:

    def __init__(self, df):

        self.df = df

        self.target_columns = [
            "总进球数",
            "场均进球",
            "参赛队伍数量",
            "总比赛场次"
        ]

    def calculate_statistics(self):

        print("=" * 60)
        print("开始计算统计量")
        print("=" * 60)

        stats_data = []

        for col in self.target_columns:

            data = self.df[col]

            stats = {
                "指标": col,
                "均值": round(data.mean(), 2),
                "中位数": round(data.median(), 2),
                "最大值": round(data.max(), 2),
                "最小值": round(data.min(), 2),
                "标准差": round(data.std(), 2),
                "样本数": int(data.count())
            }

            stats_data.append(stats)

        self.stats_df = pd.DataFrame(stats_data)

        print("\n统计结果")
        print("-" * 60)
        print(self.stats_df.to_string(index=False))
        print("-" * 60)

        return self.stats_df

    def save_statistics(self):

        output_path = TABLE_DIR / "statistics_summary.csv"

        self.stats_df.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig"
        )

        print(f"\n✓ 统计表格已保存到: {output_path}")

        return output_path

    def run(self):

        self.calculate_statistics()

        self.save_statistics()

        return self.stats_df