"""
report.py
==========================
Task1 报告生成模块
自动分析数据并生成文字总结
==========================
"""

import pandas as pd
import numpy as np

from src.config import REPORT_DIR


class ReportGenerator:

    def __init__(self, df):

        self.df = df

    def detect_inflection_points(self, column_name):

        data = self.df[column_name]

        year_data = self.df["年份"]

        diff = data.diff().dropna()

        diff_sign = np.sign(diff)

        inflection_points = []

        for i in range(1, len(diff_sign)):

            prev_sign = diff_sign.iloc[i-1]

            curr_sign = diff_sign.iloc[i]

            if prev_sign != curr_sign and prev_sign != 0 and curr_sign != 0:

                year = year_data.iloc[i+1]

                prev_val = data.iloc[i]

                curr_val = data.iloc[i+1]

                change = curr_val - prev_val

                direction = "上升" if change > 0 else "下降"

                inflection_points.append({
                    "年份": int(year),
                    "前值": round(prev_val, 2),
                    "后值": round(curr_val, 2),
                    "变化量": round(change, 2),
                    "变化方向": direction
                })

        return inflection_points

    def detect_local_extremes(self, column_name):

        data = self.df[column_name]

        year_data = self.df["年份"]

        extremes = []

        for i in range(1, len(data)-1):

            prev_val = data.iloc[i-1]

            curr_val = data.iloc[i]

            next_val = data.iloc[i+1]

            if curr_val > prev_val and curr_val > next_val:

                extremes.append({
                    "年份": int(year_data.iloc[i]),
                    "类型": "局部最大值",
                    "数值": round(curr_val, 2)
                })

            elif curr_val < prev_val and curr_val < next_val:

                extremes.append({
                    "年份": int(year_data.iloc[i]),
                    "类型": "局部最小值",
                    "数值": round(curr_val, 2)
                })

        return extremes

    def analyze_trend(self, column_name):

        data = self.df[column_name]

        years = self.df["年份"]

        first_year = int(years.iloc[0])

        last_year = int(years.iloc[-1])

        first_val = data.iloc[0]

        last_val = data.iloc[-1]

        overall_change = last_val - first_val

        overall_pct_change = (overall_change / first_val) * 100

        avg_annual_change = data.diff().mean()

        trend_direction = "上升" if overall_change > 0 else "下降" if overall_change < 0 else "持平"

        return {
            "指标": column_name,
            "起始年份": first_year,
            "结束年份": last_year,
            "起始值": round(first_val, 2),
            "结束值": round(last_val, 2),
            "总变化量": round(overall_change, 2),
            "总变化率": round(overall_pct_change, 2),
            "年均变化": round(avg_annual_change, 2),
            "整体趋势": trend_direction
        }

    def generate_report(self):

        print("=" * 60)
        print("开始生成分析报告")
        print("=" * 60)

        report_parts = []

        report_parts.append("=" * 60)
        report_parts.append("世界杯数据统计分析报告")
        report_parts.append("=" * 60)
        report_parts.append("")

        report_parts.append("一、数据概况")
        report_parts.append("-" * 40)
        report_parts.append(f"数据覆盖年份：{self.df['年份'].min()} - {self.df['年份'].max()}")
        report_parts.append(f"数据记录数：{len(self.df)}")
        report_parts.append("")

        for column in ["总进球数", "场均进球"]:

            trend = self.analyze_trend(column)

            report_parts.append(f"二、{column}分析")
            report_parts.append("-" * 40)
            report_parts.append(f"整体趋势：从{trend['起始年份']}年到{trend['结束年份']}年，{column}呈{trend['整体趋势']}趋势")
            report_parts.append(f"数值变化：从{trend['起始值']}变化到{trend['结束值']}，变化量为{trend['总变化量']}")
            report_parts.append(f"变化率：{trend['总变化率']}%")
            report_parts.append(f"年均变化：{trend['年均变化']}")
            report_parts.append("")

            inflection_points = self.detect_inflection_points(column)

            if inflection_points:

                report_parts.append(f"拐点年份（趋势反转点）：")

                for point in inflection_points:

                    report_parts.append(f"  - {point['年份']}年：从{point['前值']}{point['变化方向']}到{point['后值']}，变化量{point['变化量']}")

            else:

                report_parts.append("未检测到明显拐点")

            report_parts.append("")

            extremes = self.detect_local_extremes(column)

            if extremes:

                report_parts.append(f"局部极值点：")

                for ext in extremes:

                    report_parts.append(f"  - {ext['年份']}年：{ext['类型']}，数值为{ext['数值']}")

            report_parts.append("")

        report_parts.append("三、整体变化规律总结")
        report_parts.append("-" * 40)

        total_goals_trend = self.analyze_trend("总进球数")
        avg_goals_trend = self.analyze_trend("场均进球")

        total_inflection = self.detect_inflection_points("总进球数")
        avg_inflection = self.detect_inflection_points("场均进球")

        if total_goals_trend["整体趋势"] == "上升":

            report_parts.append("总进球数整体呈上升趋势，说明世界杯赛事的进攻火力整体增强。")

        elif total_goals_trend["整体趋势"] == "下降":

            report_parts.append("总进球数整体呈下降趋势，说明世界杯赛事的防守水平可能有所提升。")

        else:

            report_parts.append("总进球数整体保持稳定。")

        if avg_goals_trend["整体趋势"] == "上升":

            report_parts.append("场均进球数整体呈上升趋势，说明单场比赛的观赏性有所提高。")

        elif avg_goals_trend["整体趋势"] == "下降":

            report_parts.append("场均进球数整体呈下降趋势，说明比赛节奏可能变慢或防守更加严密。")

        else:

            report_parts.append("场均进球数整体保持稳定。")

        all_inflection_years = sorted(set([p["年份"] for p in total_inflection] + [p["年份"] for p in avg_inflection]))

        if all_inflection_years:

            report_parts.append(f"")

            report_parts.append(f"关键拐点年份：{', '.join(map(str, all_inflection_years))}")

            report_parts.append(f"这些年份是世界杯进球特征发生显著变化的重要节点。")

        report_parts.append("")

        report_parts.append("=" * 60)
        report_parts.append("报告生成时间：自动生成")
        report_parts.append("=" * 60)

        self.report_text = "\n".join(report_parts)

        print(self.report_text)

        return self.report_text

    def save_report(self):

        output_path = REPORT_DIR / "analysis_report.txt"

        with open(output_path, "w", encoding="utf-8") as f:

            f.write(self.report_text)

        print(f"\n✓ 分析报告已保存到: {output_path}")

        return output_path

    def run(self):

        self.generate_report()

        self.save_report()

        return self.report_text