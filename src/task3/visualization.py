"""
visualization.py
==========================
Task3 可视化模块
==========================
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import numpy as np
import pandas as pd
from config import FIGURE_DIR


def setup_chinese_font():
    """
    统一设置中文字体，解决图表中文显示问题
    """
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


class FeatureVisualization:
    """
    特征工程可视化
    """

    def __init__(self, team_history, feature_matrix, correlation_df):

        self.team_history = team_history
        self.feature_matrix = feature_matrix
        self.correlation_df = correlation_df

        setup_chinese_font()

    def plot_correlation_heatmap(self):

        print("\n绘制特征相关性热力图")

        numeric_features = [
            "主队历史参赛次数", "主队历史比赛场次", "主队历史场均进球",
            "主队历史场均失球", "主队历史净胜球", "主队历史成绩排名",
            "客队历史参赛次数", "客队历史比赛场次", "客队历史场均进球",
            "客队历史场均失球", "客队历史净胜球", "客队历史成绩排名",
            "参赛次数差", "场均进球差", "场均失球差",
            "净胜球差", "成绩排名差", "胜负编码"
        ]

        available_features = [f for f in numeric_features if f in self.feature_matrix.columns]

        corr_matrix = self.feature_matrix[available_features].corr()

        plt.figure(figsize=(16, 14))

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            mask=mask,
            annot_kws={"size": 8}
        )

        plt.title("队伍历史特征与比赛胜负相关性热力图", fontsize=16, fontweight="bold", pad=20)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(rotation=0, fontsize=10)

        plt.tight_layout()

        fig_path = FIGURE_DIR / "task3_correlation_heatmap.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"✓ 相关性热力图已保存到: {fig_path}")

        plt.close()

    def plot_correlation_bar(self):

        print("绘制特征与胜负相关性柱状图")

        corr_values = self.correlation_df["平均相关系数"].sort_values()

        plt.figure(figsize=(14, 10))

        colors = ["#E63946" if x < 0 else "#2A9D8F" for x in corr_values.values]

        bars = plt.barh(range(len(corr_values)), corr_values.values, color=colors,
                        edgecolor="black", linewidth=0.5)

        plt.yticks(range(len(corr_values)), corr_values.index, fontsize=10)
        plt.axvline(x=0, color="black", linewidth=0.8)
        plt.axvline(x=0.2, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
        plt.axvline(x=-0.2, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
        plt.axvline(x=0.4, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
        plt.axvline(x=-0.4, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)

        for bar, value in zip(bars, corr_values.values):
            x_pos = value + (0.01 if value >= 0 else -0.01)
            ha = "left" if value >= 0 else "right"
            plt.text(x_pos, bar.get_y() + bar.get_height() / 2,
                     f"{value:.3f}", ha=ha, va="center", fontsize=9)

        plt.title("各特征与比赛胜负相关性", fontsize=16, fontweight="bold", pad=20)
        plt.xlabel("平均相关系数", fontsize=12)
        plt.ylabel("特征", fontsize=12)

        plt.text(0.6, -1.5, "← 负相关(客队优势) | 正相关(主队优势) →",
                 fontsize=10, ha="center", style="italic")

        plt.grid(axis="x", alpha=0.3)

        plt.tight_layout()

        fig_path = FIGURE_DIR / "task3_correlation_bar.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"✓ 相关性柱状图已保存到: {fig_path}")

        plt.close()

    def plot_team_history_top(self, top_n=15):

        print("绘制队伍历史指标TOP15图")

        top_teams = self.team_history.head(top_n)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        axes[0, 0].barh(top_teams["队伍名称"][::-1], top_teams["历史参赛次数"][::-1],
                        color="#2A9D8F", edgecolor="black")
        axes[0, 0].set_title("历史参赛次数 TOP15", fontsize=14, fontweight="bold")
        axes[0, 0].set_xlabel("参赛次数", fontsize=11)
        axes[0, 0].grid(axis="x", alpha=0.3)

        axes[0, 1].barh(top_teams["队伍名称"][::-1], top_teams["历史总进球"][::-1],
                        color="#E63946", edgecolor="black")
        axes[0, 1].set_title("历史总进球 TOP15", fontsize=14, fontweight="bold")
        axes[0, 1].set_xlabel("总进球数", fontsize=11)
        axes[0, 1].grid(axis="x", alpha=0.3)

        axes[1, 0].barh(top_teams["队伍名称"][::-1], top_teams["历史场均进球"][::-1],
                        color="#457B9D", edgecolor="black")
        axes[1, 0].set_title("历史场均进球 TOP15", fontsize=14, fontweight="bold")
        axes[1, 0].set_xlabel("场均进球数", fontsize=11)
        axes[1, 0].grid(axis="x", alpha=0.3)

        axes[1, 1].barh(top_teams["队伍名称"][::-1], top_teams["历史净胜球"][::-1],
                        color="#E9C46A", edgecolor="black")
        axes[1, 1].set_title("历史净胜球 TOP15", fontsize=14, fontweight="bold")
        axes[1, 1].set_xlabel("净胜球数", fontsize=11)
        axes[1, 1].grid(axis="x", alpha=0.3)

        plt.suptitle("队伍历史指标排名", fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout()

        fig_path = FIGURE_DIR / "task3_team_history_top.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"✓ 队伍历史指标图已保存到: {fig_path}")

        plt.close()

    def plot_feature_distribution(self):

        print("绘制特征分布图")

        key_features = ["阶段类型", "参赛次数差", "比赛场次差", "场均进球差",
                       "净胜球差", "成绩排名差", "近3届胜率差", "交锋胜率"]

        available_features = [f for f in key_features if f in self.feature_matrix.columns]

        n_features = len(available_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

        colors = ["#2A9D8F", "#E63946", "#457B9D", "#E9C46A", "#F4A261", "#264653"]

        for i, feature in enumerate(available_features):
            axes[i].hist(self.feature_matrix[feature].dropna(), bins=20,
                        color=colors[i % len(colors)], edgecolor="black", alpha=0.7)
            axes[i].set_title(f"{feature} 分布", fontsize=12, fontweight="bold")
            axes[i].set_xlabel(feature, fontsize=10)
            axes[i].set_ylabel("频次", fontsize=10)
            axes[i].grid(axis="y", alpha=0.3)

        for i in range(n_features, len(axes)):
            axes[i].axis("off")

        plt.suptitle("队伍历史特征分布", fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout()

        fig_path = FIGURE_DIR / "task3_feature_distribution.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"✓ 特征分布图已保存到: {fig_path}")

        plt.close()

    def run(self):

        self.plot_correlation_heatmap()
        self.plot_correlation_bar()
        self.plot_team_history_top()
        self.plot_feature_distribution()
