"""
loader.py
==========================
Task1 数据读取与预处理
==========================
"""

import pandas as pd
from config import DATA_DIR


class WorldCupLoader:
    """
    世界杯宏观数据读取器
    """

    def __init__(self):

        self.file = DATA_DIR / "WorldCups_preview.csv"

        self.column_mapping = {

            "Year": "年份",
            "Country": "举办国家",
            "Winner": "冠军",
            "Runners-Up": "亚军",
            "Third": "季军",
            "Fourth": "殿军",
            "GoalsScored": "总进球数",
            "QualifiedTeams": "参赛队伍数量",
            "MatchesPlayed": "总比赛场次",
            "Attendance": "总观众人数"

        }

    def load(self):

        print("=" * 60)
        print("开始读取 WorldCups_preview.csv")
        print("=" * 60)

        encodings = [
            "utf-8",
            "utf-8-sig",
            "gbk",
            "latin1"
        ]

        df = None

        for enc in encodings:

            try:

                df = pd.read_csv(
                    self.file,
                    encoding=enc
                )

                print(f"✓ 成功使用 {enc} 编码读取")

                break

            except UnicodeDecodeError:

                continue

        if df is None:

            raise FileNotFoundError("无法读取CSV文件")

        # ------------------------
        # 去除空格
        # ------------------------

        df.columns = df.columns.str.strip()

        # ------------------------
        # 中文列名
        # ------------------------

        df.rename(
            columns=self.column_mapping,
            inplace=True
        )

        # ------------------------
        # 数值转换
        # ------------------------

        numeric_cols = [

            "年份",
            "总进球数",
            "参赛队伍数量",
            "总比赛场次",
            "总观众人数"

        ]

        for col in numeric_cols:

            df[col] = pd.to_numeric(

                df[col],

                errors="coerce"

            )

        # ------------------------
        # 场均进球
        # ------------------------

        df["场均进球"] = (

            df["总进球数"]

            /

            df["总比赛场次"]

        )

        # ------------------------
        # 缺失值
        # ------------------------

        df.fillna(

            df.median(numeric_only=True),

            inplace=True

        )

        # ------------------------
        # 排序
        # ------------------------

        df.sort_values(

            "年份",

            inplace=True

        )

        df.reset_index(

            drop=True,

            inplace=True

        )

        # ------------------------
        # 输出数据概况
        # ------------------------

        print("\n数据概况")

        print("-" * 40)

        print(f"样本数量：{df.shape[0]}")

        print(f"字段数量：{df.shape[1]}")

        print(f"年份范围：{df['年份'].min()} ~ {df['年份'].max()}")

        print(f"重复记录：{df.duplicated().sum()} 条")

        print("\n缺失值统计")

        print(df.isnull().sum())

        print("=" * 60)

        return df