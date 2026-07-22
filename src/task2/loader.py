"""
loader.py
==========================
Task2 数据读取与预处理
==========================
"""

import pandas as pd
import numpy as np
from src.config import DATA_DIR, OUTPUT_DIR


class MatchLoader:
    """
    世界杯比赛数据读取器
    包含完整的数据清洗功能：缺失值处理、异常值检测与处理
    """

    def __init__(self):

        self.file = DATA_DIR / "WorldCupMatches.csv"

        self.column_mapping = {
            "Year": "年份",
            "Datetime": "日期时间",
            "Stage": "阶段",
            "Stadium": "球场",
            "City": "城市",
            "Home Team Name": "主队名称",
            "Home Team Goals": "主队进球",
            "Away Team Goals": "客队进球",
            "Away Team Name": "客队名称",
            "Win conditions": "胜负条件",
            "Attendance": "观众人数",
            "Half-time Home Goals": "半场主队进球",
            "Half-time Away Goals": "半场客队进球",
            "Referee": "裁判",
            "Assistant 1": "助理裁判1",
            "Assistant 2": "助理裁判2",
            "RoundID": "轮次ID",
            "MatchID": "比赛ID",
            "Home Team Initials": "主队缩写",
            "Away Team Initials": "客队缩写"
        }

        self.stage_mapping = {
            "Group": "小组赛",
            "Group Stage": "小组赛",
            "First round": "小组赛",
            "Second Round": "1/8决赛",
            "Round of 16": "1/8决赛",
            "Quarter-finals": "1/4决赛",
            "Quarter Finals": "1/4决赛",
            "Semi-finals": "半决赛",
            "Semi Finals": "半决赛",
            "Final": "决赛",
            "Third place": "三四名决赛",
            "3rd Place": "三四名决赛",
            "Play-off for third place": "三四名决赛",
            "Match for third place": "三四名决赛",
            "Preliminary round": "预选赛",
            "Group 1": "小组赛",
            "Group 2": "小组赛",
            "Group 3": "小组赛",
            "Group 4": "小组赛",
            "Group 5": "小组赛",
            "Group 6": "小组赛",
            "Group A": "小组赛",
            "Group B": "小组赛",
            "Group C": "小组赛",
            "Group D": "小组赛",
            "Group E": "小组赛",
            "Group F": "小组赛",
            "Group G": "小组赛",
            "Group H": "小组赛"
        }

    def load(self):

        print("=" * 60)
        print("开始读取 WorldCupMatches.csv")
        print("=" * 60)

        encodings = ["utf-8", "utf-8-sig", "gbk", "latin1"]

        df = None
        for enc in encodings:
            try:
                df = pd.read_csv(self.file, encoding=enc)
                print(f"✓ 成功使用 {enc} 编码读取")
                break
            except UnicodeDecodeError:
                continue

        if df is None:
            raise FileNotFoundError("无法读取CSV文件")

        df.columns = df.columns.str.strip()

        df.rename(columns=self.column_mapping, inplace=True)

        print("\n数据清洗步骤：")
        print("-" * 40)

        self._convert_numeric(df)

        self._create_derived_columns(df)

        df = self._handle_missing_values(df)

        self._handle_outliers(df)

        self._standardize_stage(df)

        df["日期时间"] = pd.to_datetime(df["日期时间"], errors="coerce")

        df["比赛年份"] = df["日期时间"].dt.year
        df["比赛月份"] = df["日期时间"].dt.month
        df["比赛日"] = df["日期时间"].dt.day
        df["比赛小时"] = df["日期时间"].dt.hour

        df.sort_values(["年份", "日期时间"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        self._save_cleaned_data(df)

        print("\n数据概况")
        print("-" * 40)
        print(f"样本数量：{df.shape[0]}")
        print(f"字段数量：{df.shape[1]}")
        print(f"年份范围：{df['年份'].min()} ~ {df['年份'].max()}")
        print(f"重复记录：{df.duplicated().sum()} 条")

        print("\n缺失值统计")
        print(df.isnull().sum())

        print("\n比赛阶段分布：")
        print(df["阶段"].value_counts())

        print("\n进球统计：")
        print(f"场均进球：{df['总进球数'].mean():.2f}")
        print(f"最大进球：{df['总进球数'].max()}")
        print(f"最小进球：{df['总进球数'].min()}")

        print("=" * 60)

        return df

    def _convert_numeric(self, df):

        numeric_cols = [
            "年份", "主队进球", "客队进球", "观众人数",
            "半场主队进球", "半场客队进球", "轮次ID", "比赛ID"
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        print(f"✓ 数值类型转换完成：{', '.join(numeric_cols)}")

    def _create_derived_columns(self, df):

        df["总进球数"] = df["主队进球"] + df["客队进球"]

        df["半场总进球"] = df["半场主队进球"] + df["半场客队进球"]

        df["胜负结果"] = df.apply(self._determine_result, axis=1)

        df["进球差"] = df["主队进球"] - df["客队进球"]

        df["半场进球占比"] = df["半场总进球"] / df["总进球数"].replace(0, np.nan)

        print("✓ 派生列创建完成：总进球数、半场总进球、胜负结果、进球差、半场进球占比")

    def _handle_missing_values(self, df):

        print(f"\n缺失值处理前：")
        missing_before = df.isnull().sum().sum()
        print(f"  总缺失值：{missing_before}")

        critical_cols = ["年份", "主队进球", "客队进球"]
        df = df.dropna(subset=critical_cols)

        numeric_cols = ["半场主队进球", "半场客队进球"]
        for col in numeric_cols:
            if col in df.columns:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"  {col}：用中位数 {median_val} 填充")

        categorical_cols = ["阶段", "球场", "城市"]
        for col in categorical_cols:
            if col in df.columns:
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                print(f"  {col}：用众数 '{mode_val}' 填充")

        missing_after = df.isnull().sum().sum()
        print(f"✓ 缺失值处理完成，剩余缺失值：{missing_after}")

        return df

    def _handle_outliers(self, df):

        print(f"\n异常值处理：")

        for col in ["主队进球", "客队进球", "总进球数", "半场总进球", "观众人数"]:
            if col in df.columns:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                outliers_before = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()

                if col in ["主队进球", "客队进球", "总进球数", "半场总进球"]:
                    df.loc[df[col] < 0, col] = 0
                else:
                    df.loc[df[col] < lower_bound, col] = lower_bound
                    df.loc[df[col] > upper_bound, col] = upper_bound

                outliers_after = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()

                print(f"  {col}：处理前异常值 {outliers_before} 个，处理后 {outliers_after} 个")

        print("✓ 异常值处理完成")

    def _standardize_stage(self, df):

        df["阶段"] = df["阶段"].replace(self.stage_mapping)

        df["阶段"] = df["阶段"].apply(self._simplify_stage)

        print("✓ 阶段标准化完成")

    def _simplify_stage(self, stage):

        stage = str(stage).strip()
        if "小组" in stage:
            return "小组赛"
        elif "1/8" in stage or "十六" in stage:
            return "1/8决赛"
        elif "1/4" in stage or "八" in stage:
            return "1/4决赛"
        elif "半" in stage:
            return "半决赛"
        elif "三四" in stage:
            return "三四名决赛"
        elif "决" in stage and "三四" not in stage:
            return "决赛"
        else:
            return stage

    def _determine_result(self, row):
        if row["主队进球"] > row["客队进球"]:
            return "主队胜"
        elif row["主队进球"] < row["客队进球"]:
            return "客队胜"
        else:
            return "平局"

    def _save_cleaned_data(self, df):

        cleaned_path = OUTPUT_DIR / "tables" / "task2_cleaned_matches.csv"
        df.to_csv(cleaned_path, index=False, encoding="utf-8-sig")
        print(f"\n✓ 清洗后数据已保存到: {cleaned_path}")


class PlayerLoader:
    """
    世界杯球员数据读取器
    """

    def __init__(self):

        self.file = DATA_DIR / "WorldCupPlayers.csv"

        self.column_mapping = {
            "RoundID": "轮次ID",
            "MatchID": "比赛ID",
            "Team Initials": "球队缩写",
            "Coach Name": "教练姓名",
            "Line-up": "阵容",
            "Shirt Number": "球衣号码",
            "Player Name": "球员姓名",
            "Position": "位置",
            "Event": "事件"
        }

    def load(self):

        print("=" * 60)
        print("开始读取 WorldCupPlayers.csv")
        print("=" * 60)

        encodings = ["utf-8", "utf-8-sig", "gbk", "latin1"]

        df = None
        for enc in encodings:
            try:
                df = pd.read_csv(self.file, encoding=enc)
                print(f"✓ 成功使用 {enc} 编码读取")
                break
            except UnicodeDecodeError:
                continue

        if df is None:
            raise FileNotFoundError("无法读取CSV文件")

        df.columns = df.columns.str.strip()

        df.rename(columns=self.column_mapping, inplace=True)

        numeric_cols = ["轮次ID", "比赛ID", "球衣号码"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["进球数"] = df["事件"].apply(self._count_goals)

        df["助攻数"] = df["事件"].apply(self._count_assists)

        df["黄牌数"] = df["事件"].apply(self._count_yellow_cards)

        df["红牌数"] = df["事件"].apply(self._count_red_cards)

        print("\n数据概况")
        print("-" * 40)
        print(f"样本数量：{df.shape[0]}")
        print(f"字段数量：{df.shape[1]}")
        print(f"重复记录：{df.duplicated().sum()} 条")

        print("\n位置分布：")
        print(df["位置"].value_counts().head(10))

        print("\n球员统计：")
        print(f"总进球数：{df['进球数'].sum()}")
        print(f"总助攻数：{df['助攻数'].sum()}")

        print("=" * 60)

        return df

    def _count_goals(self, event):
        if pd.isna(event):
            return 0
        return event.count("G")

    def _count_assists(self, event):
        if pd.isna(event):
            return 0
        return event.count("A")

    def _count_yellow_cards(self, event):
        if pd.isna(event):
            return 0
        return event.count("Y")

    def _count_red_cards(self, event):
        if pd.isna(event):
            return 0
        return event.count("R")