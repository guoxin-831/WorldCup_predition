"""
visualization.py
Task1 可视化模块
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

from config import FIGURE_DIR


class Visualizer:

    def __init__(self, df):

        self.df = df

        plt.rcParams["font.sans-serif"] = [
            "Microsoft YaHei",
            "SimHei",
            "Arial Unicode MS",
            "DejaVu Sans"
        ]

        plt.rcParams["axes.unicode_minus"] = False

        plt.rcParams["font.family"] = "sans-serif"

        sns.set_theme(
            style="whitegrid",
            font="Microsoft YaHei"
        )

    # -------------------------
    # 总进球趋势
    # -------------------------

    def total_goals_trend(self):

        plt.figure(figsize=(12,6))

        plt.plot(

            self.df["年份"],

            self.df["总进球数"],

            marker="o",

            linewidth=2

        )

        for x,y in zip(

            self.df["年份"],

            self.df["总进球数"]

        ):

            plt.text(

                x,

                y+2,

                int(y),

                ha="center",

                fontsize=8

            )

        plt.title("总进球数年份趋势", fontsize=14, fontweight="bold")

        plt.xlabel("年份", fontsize=12)

        plt.ylabel("总进球数", fontsize=12)

        plt.tight_layout()

        plt.savefig(

            FIGURE_DIR /

            "figure1_total_goals.png",

            dpi=300

        )

        plt.close()

    # -------------------------
    # 场均进球
    # -------------------------

    def average_goals(self):

        plt.figure(figsize=(12,6))

        plt.plot(

            self.df["年份"],

            self.df["场均进球"],

            marker="s",

            linewidth=2

        )

        for x,y in zip(

            self.df["年份"],

            self.df["场均进球"]

        ):

            plt.text(

                x,

                y+0.05,

                f"{y:.2f}",

                ha="center",

                fontsize=8

            )

        plt.title("场均进球年份趋势", fontsize=14, fontweight="bold")

        plt.xlabel("年份", fontsize=12)

        plt.ylabel("场均进球数", fontsize=12)

        plt.tight_layout()

        plt.savefig(

            FIGURE_DIR /

            "figure2_average_goals.png",

            dpi=300

        )

        plt.close()

    # -------------------------
    # Pearson热力图
    # -------------------------

    def correlation_heatmap(self):

        cols=[

            "总进球数",

            "场均进球",

            "参赛队伍数量",

            "总比赛场次",

            "总观众人数"

        ]

        corr=self.df[
            cols
        ].corr()

        plt.figure(figsize=(8,6))

        sns.heatmap(

            corr,

            annot=True,

            cmap="RdBu_r",

            square=True,

            fmt=".2f"

        )

        plt.tight_layout()

        plt.savefig(

            FIGURE_DIR/

            "figure3_heatmap.png",

            dpi=300

        )

        plt.close()

    # -------------------------
    # 箱线图
    # -------------------------

    def boxplot(self):

        plt.figure(figsize=(10,6))

        sns.boxplot(

            data=self.df[

                [

                    "总进球数",

                    "场均进球",

                    "参赛队伍数量",

                    "总比赛场次"

                ]

            ]

        )

        plt.tight_layout()

        plt.savefig(

            FIGURE_DIR/

            "figure4_boxplot.png",

            dpi=300

        )

        plt.close()

    # -------------------------
    # 运行全部
    # -------------------------

    def run(self):

        self.total_goals_trend()

        self.average_goals()

        self.correlation_heatmap()

        self.boxplot()

        print("✓ 图像全部保存完成")