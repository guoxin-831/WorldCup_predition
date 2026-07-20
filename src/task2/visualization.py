"""
visualization.py
==========================
Task2 可视化模块
==========================
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import numpy as np
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


class MatchVisualization:
    """
    比赛数据可视化
    """

    def __init__(self, match_stats, yearly_stats, stage_stats, goal_dist):

        self.match_stats = match_stats
        self.yearly_stats = yearly_stats
        self.stage_stats = stage_stats
        self.goal_dist = goal_dist

        setup_chinese_font()

    def plot_yearly_goals(self):

        plt.figure(figsize=(14, 7))

        plt.plot(
            self.yearly_stats["年份"],
            self.yearly_stats["总进球数"],
            marker="o",
            linestyle="-",
            color="#E63946",
            linewidth=2,
            markersize=8,
            label="总进球数"
        )

        plt.plot(
            self.yearly_stats["年份"],
            self.yearly_stats["场均进球"],
            marker="s",
            linestyle="--",
            color="#2A9D8F",
            linewidth=2,
            markersize=8,
            label="场均进球"
        )

        plt.title("世界杯年度进球趋势", fontsize=16, fontweight="bold", pad=20)
        plt.xlabel("年份", fontsize=12)
        plt.ylabel("进球数", fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)

        for x, y in zip(self.yearly_stats["年份"], self.yearly_stats["总进球数"]):
            plt.text(x, y + 5, str(int(y)), ha="center", va="bottom", fontsize=9)

        plt.tight_layout()

        fig_path = FIGURE_DIR / "task2_yearly_goals_trend.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"✓ 年度进球趋势图已保存到: {fig_path}")

        plt.close()

    def plot_goal_distribution(self):

        plt.figure(figsize=(14, 7))

        colors = ["#1D3557", "#457B9D", "#A8DADC", "#E63946", "#F1FAEE"]

        bars = plt.bar(
            self.goal_dist["进球数"],
            self.goal_dist["场次"],
            color=colors[:len(self.goal_dist)],
            edgecolor="black",
            linewidth=1
        )

        plt.title("比赛进球数分布", fontsize=16, fontweight="bold", pad=20)
        plt.xlabel("每场进球数", fontsize=12)
        plt.ylabel("比赛场次", fontsize=12)
        plt.grid(axis="y", alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + 5,
                str(int(height)),
                ha="center",
                va="bottom",
                fontsize=10
            )

        plt.tight_layout()

        fig_path = FIGURE_DIR / "task2_goal_distribution.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"✓ 进球数分布图已保存到: {fig_path}")

        plt.close()

    def plot_stage_comparison(self):

        plt.figure(figsize=(16, 8))

        stages = self.stage_stats["阶段"]
        x = range(len(stages))
        width = 0.25

        plt.bar(
            [i - width for i in x],
            self.stage_stats["场均进球"],
            width=width,
            label="场均进球",
            color="#E63946"
        )

        plt.bar(
            x,
            self.stage_stats["主队胜率"] * 10,
            width=width,
            label="主队胜率(×10)",
            color="#2A9D8F"
        )

        plt.bar(
            [i + width for i in x],
            self.stage_stats["平局率"] * 10,
            width=width,
            label="平局率(×10)",
            color="#E9C46A"
        )

        plt.title("各阶段比赛数据对比", fontsize=16, fontweight="bold", pad=20)
        plt.xlabel("比赛阶段", fontsize=12)
        plt.ylabel("数值", fontsize=12)
        plt.xticks(x, stages, rotation=45, ha="right")
        plt.legend(fontsize=12)
        plt.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        fig_path = FIGURE_DIR / "task2_stage_comparison.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"✓ 阶段对比图已保存到: {fig_path}")

        plt.close()

    def plot_result_pie(self):

        result_counts = self.stage_stats[["主队胜场", "客队胜场", "平局场数"]].sum()

        plt.figure(figsize=(8, 8))

        colors = ["#2A9D8F", "#E63946", "#E9C46A"]

        wedges, texts, autotexts = plt.pie(
            result_counts.values,
            labels=result_counts.index,
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
            textprops={"fontsize": 12}
        )

        plt.title("比赛结果分布", fontsize=16, fontweight="bold", pad=20)
        plt.legend(fontsize=12)

        plt.tight_layout()

        fig_path = FIGURE_DIR / "task2_result_distribution.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"✓ 结果分布图已保存到: {fig_path}")

        plt.close()

    def plot_correlation_heatmap(self, df):

        corr_df = df[["主队进球", "客队进球", "总进球数", "半场总进球", "观众人数", "进球差"]].corr()

        plt.figure(figsize=(10, 8))

        sns.heatmap(
            corr_df,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8}
        )

        plt.title("比赛特征相关性热力图", fontsize=16, fontweight="bold", pad=20)

        plt.tight_layout()

        fig_path = FIGURE_DIR / "task2_correlation_heatmap.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"✓ 相关性热力图已保存到: {fig_path}")

        plt.close()

    def plot_half_time_scatter(self, df, correlation_results=None):

        print("\n绘制上半场进球与全场进球散点相关图")

        valid_df = df.dropna(subset=["半场总进球", "总进球数"])

        x = valid_df["半场总进球"]
        y = valid_df["总进球数"]

        plt.figure(figsize=(12, 8))

        sns.scatterplot(
            x=x,
            y=y,
            alpha=0.7,
            color="#2A9D8F",
            s=100,
            edgecolor="black",
            linewidth=0.5
        )

        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        plt.plot(x, p(x), "r--", linewidth=2, label=f"拟合直线: y={z[0]:.2f}x+{z[1]:.2f}")

        if correlation_results is not None:
            pearson = correlation_results.get("Pearson相关系数", "")
            spearman = correlation_results.get("Spearman相关系数", "")
            plt.text(
                0.05, 0.95,
                f"Pearson相关系数: {pearson:.4f}\nSpearman相关系数: {spearman:.4f}",
                transform=plt.gca().transAxes,
                fontsize=12,
                bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray", boxstyle="round")
            )

        plt.title("上半场进球与全场进球散点相关图", fontsize=16, fontweight="bold", pad=20)
        plt.xlabel("上半场进球数", fontsize=12)
        plt.ylabel("全场进球数", fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        fig_path = FIGURE_DIR / "task2_half_time_full_time_scatter.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"✓ 上半场进球与全场进球散点图已保存到: {fig_path}")

        plt.close()

    def run(self, df, correlation_results=None):

        self.plot_yearly_goals()
        self.plot_goal_distribution()
        self.plot_stage_comparison()
        self.plot_result_pie()
        self.plot_correlation_heatmap(df)
        self.plot_half_time_scatter(df, correlation_results)


class PlayerVisualization:
    """
    球员数据可视化
    """

    def __init__(self, top_scorers, top_assistants, team_stats, position_stats):

        self.top_scorers = top_scorers
        self.top_assistants = top_assistants
        self.team_stats = team_stats
        self.position_stats = position_stats

        setup_chinese_font()

    def plot_top_scorers(self):

        plt.figure(figsize=(14, 8))

        colors = ["#E63946", "#F4A261", "#E9C46A", "#2A9D8F", "#457B9D"]

        bars = plt.barh(
            self.top_scorers.index[::-1],
            self.top_scorers["总进球"][::-1],
            color=colors[:len(self.top_scorers)],
            edgecolor="black",
            linewidth=1
        )

        plt.title("历史射手榜 TOP10", fontsize=16, fontweight="bold", pad=20)
        plt.xlabel("进球数", fontsize=12)
        plt.ylabel("球员姓名", fontsize=12)
        plt.grid(axis="x", alpha=0.3)

        for bar in bars:
            width = bar.get_width()
            plt.text(
                width + 0.5,
                bar.get_y() + bar.get_height() / 2,
                str(int(width)),
                ha="left",
                va="center",
                fontsize=10
            )

        plt.tight_layout()

        fig_path = FIGURE_DIR / "task2_top_scorers.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"✓ 射手榜图已保存到: {fig_path}")

        plt.close()

    def plot_top_assistants(self):

        plt.figure(figsize=(14, 8))

        colors = ["#457B9D", "#2A9D8F", "#A8DADC", "#E9C46A", "#F4A261"]

        bars = plt.barh(
            self.top_assistants.index[::-1],
            self.top_assistants["总助攻"][::-1],
            color=colors[:len(self.top_assistants)],
            edgecolor="black",
            linewidth=1
        )

        plt.title("历史助攻榜 TOP10", fontsize=16, fontweight="bold", pad=20)
        plt.xlabel("助攻数", fontsize=12)
        plt.ylabel("球员姓名", fontsize=12)
        plt.grid(axis="x", alpha=0.3)

        for bar in bars:
            width = bar.get_width()
            plt.text(
                width + 0.5,
                bar.get_y() + bar.get_height() / 2,
                str(int(width)),
                ha="left",
                va="center",
                fontsize=10
            )

        plt.tight_layout()

        fig_path = FIGURE_DIR / "task2_top_assistants.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"✓ 助攻榜图已保存到: {fig_path}")

        plt.close()

    def plot_team_goals(self):

        top_teams = self.team_stats.head(10)

        plt.figure(figsize=(14, 8))

        colors = sns.color_palette("viridis", len(top_teams))

        bars = plt.barh(
            top_teams.index[::-1],
            top_teams["进球数"][::-1],
            color=colors,
            edgecolor="black",
            linewidth=1
        )

        plt.title("球队历史进球数 TOP10", fontsize=16, fontweight="bold", pad=20)
        plt.xlabel("进球数", fontsize=12)
        plt.ylabel("球队缩写", fontsize=12)
        plt.grid(axis="x", alpha=0.3)

        for bar in bars:
            width = bar.get_width()
            plt.text(
                width + 2,
                bar.get_y() + bar.get_height() / 2,
                str(int(width)),
                ha="left",
                va="center",
                fontsize=10
            )

        plt.tight_layout()

        fig_path = FIGURE_DIR / "task2_team_goals.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"✓ 球队进球图已保存到: {fig_path}")

        plt.close()

    def plot_position_stats(self):

        plt.figure(figsize=(12, 7))

        x = range(len(self.position_stats))
        width = 0.35

        plt.bar(
            [i - width / 2 for i in x],
            self.position_stats["进球数"],
            width=width,
            label="进球数",
            color="#E63946"
        )

        plt.bar(
            [i + width / 2 for i in x],
            self.position_stats["助攻数"],
            width=width,
            label="助攻数",
            color="#2A9D8F"
        )

        plt.title("各位置进球与助攻统计", fontsize=16, fontweight="bold", pad=20)
        plt.xlabel("位置", fontsize=12)
        plt.ylabel("数量", fontsize=12)
        plt.xticks(x, self.position_stats.index, rotation=45, ha="right")
        plt.legend(fontsize=12)
        plt.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        fig_path = FIGURE_DIR / "task2_position_stats.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"✓ 位置统计图已保存到: {fig_path}")

        plt.close()

    def run(self):

        self.plot_top_scorers()
        self.plot_top_assistants()
        self.plot_team_goals()
        self.plot_position_stats()