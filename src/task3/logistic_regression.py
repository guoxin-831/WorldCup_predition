"""
logistic_regression.py
==========================
Task3 第二小问：基于逻辑回归的比赛胜负分类预测（主调度类）
==========================
功能：
1. 协调特征工程、模型训练、预测等模块
2. 提供统一的训练和预测接口
3. 生成评估报告和可视化图表
4. 支持交互式预测
==========================
"""

import pandas as pd
import numpy as np
import joblib
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import REPORT_DIR, MODEL_DIR, TABLE_DIR, FIGURE_DIR

from .utils import setup_logger, setup_random_seed, setup_chinese_font, format_probability
from .models import ModelFactory
from .trainer import Trainer
from .predictor import Predictor

logger = setup_logger(__name__)


class MatchResultClassifier:
    """
    比赛胜负分类预测器（主调度类）

    协调各个模块：特征工程、模型训练、预测、报告生成
    """

    def __init__(self, feature_matrix: pd.DataFrame, team_history: pd.DataFrame):
        """
        初始化预测器

        Args:
            feature_matrix: 特征矩阵
            team_history: 队伍历史数据
        """
        setup_random_seed()
        logger.info("初始化 MatchResultClassifier")

        self.feature_matrix = feature_matrix.copy()
        self.team_history = team_history.copy()

        self.feature_columns = [
            "阶段类型",
            "参赛次数差", "比赛场次差", "场均进球差", "场均失球差",
            "净胜球差", "成绩排名差", "近3届场均进球差", "近3届胜率差",
            "场均净胜球差", "淘汰赛胜率差", "小组赛胜率差",
            "场均半场进球差", "半场胜率差",
            "交锋胜场", "交锋平局", "交锋负场", "交锋净胜球",
            "交锋总场次", "交锋胜率"
        ]

        self.feature_columns = [c for c in self.feature_columns
                               if c in self.feature_matrix.columns]

        self.target_column = "胜负结果"

        self.trainer = None
        self.predictor = None

        self.best_model_name = None
        self.best_model = None
        self.all_metrics = {}

        logger.info("MatchResultClassifier 初始化完成")

    def _init_trainer(self) -> None:
        """初始化训练器"""
        if self.trainer is None:
            self.trainer = Trainer(
                feature_matrix=self.feature_matrix,
                feature_columns=self.feature_columns,
                target_column=self.target_column
            )
            logger.info("训练器初始化完成")

    def _init_predictor(self, home_advantage_factor: float = 0.85) -> None:
        """初始化预测器"""
        if self.predictor is None:
            self.predictor = Predictor(
                team_history=self.team_history,
                feature_columns=self.feature_columns,
                home_advantage_factor=home_advantage_factor
            )
            logger.info("预测器初始化完成")

    def split_data(self):
        """数据集划分（委托给训练器）"""
        self._init_trainer()
        return self.trainer.split_data()

    def preprocess(self):
        """数据预处理（委托给训练器）"""
        self._init_trainer()
        return self.trainer.preprocess()

    def train_all_models(self, parallel: bool = True):
        """训练所有模型（委托给训练器）"""
        self._init_trainer()
        self.all_metrics = self.trainer.train_all_models(parallel=parallel)
        self.best_model_name = self.trainer.best_model_name
        self.best_model = self.trainer.best_model
        return self.all_metrics

    def evaluate(self):
        """评估模型（委托给训练器）"""
        self._init_trainer()
        return self.trainer.evaluate()

    def retrain_with_full_data(self):
        """全量数据重新训练（委托给训练器）"""
        self._init_trainer()
        return self.trainer.retrain_with_full_data()

    def tune_hyperparameters(self):
        """参数调优（委托给训练器）"""
        self._init_trainer()
        return self.trainer.tune_hyperparameters()

    def predict(self, home_team: str, away_team: str, stage: str = "淘汰赛", home_advantage_factor: float = 0.85) -> Dict[str, Any]:
        """
        预测比赛结果

        Args:
            home_team: 主队名称
            away_team: 客队名称
            stage: 比赛阶段
            home_advantage_factor: 主队优势因子，用于调整主队获胜概率。
                                   值越小，主队获胜概率降低越多。默认0.85。

        Returns:
            预测结果字典
        """
        self._init_predictor(home_advantage_factor)

        self.predictor.home_advantage_factor = home_advantage_factor

        if not self.predictor.model:
            logger.info("预测器模型未加载，尝试从文件加载")
            if not self.predictor.load_model(self.best_model_name or "CatBoost"):
                logger.warning("模型加载失败，使用训练器中的模型")
                self.predictor.model = self.best_model
                self.predictor.scaler = self.trainer.scaler

        result = self.predictor.predict(home_team, away_team, stage)

        if "error" not in result:
            result["置信度"] = "高" if max(result["主队胜概率"], result["平局概率"], result["客队胜概率"]) > 0.7 else "中" if max(result["主队胜概率"], result["平局概率"], result["客队胜概率"]) > 0.5 else "低"
            result["使用模型"] = self.best_model_name

        return result

    def predict_2026(self, home_team: str, away_team: str, home_advantage_factor: float = 0.85) -> Dict[str, Any]:
        """预测2026年世界杯比赛结果"""
        return self.predict(home_team, away_team, stage="淘汰赛", home_advantage_factor=home_advantage_factor)

    def plot_confusion_matrix(self):
        """绘制混淆矩阵图"""
        if self.trainer is None:
            logger.warning("训练器未初始化")
            return

        setup_chinese_font()

        cm = self.trainer.confusion_matrix
        classes = self.trainer.best_model.classes_

        plt.figure(figsize=(10, 8))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
            cbar_kws={"shrink": 0.8},
            annot_kws={"size": 16}
        )

        plt.title(f"{self.best_model_name} 混淆矩阵", fontsize=16, fontweight="bold", pad=20)
        plt.xlabel("预测结果", fontsize=12)
        plt.ylabel("实际结果", fontsize=12)

        plt.tight_layout()

        fig_path = FIGURE_DIR / f"task3_confusion_matrix_{self.best_model_name}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"✓ 混淆矩阵图已保存到: {fig_path}")

        plt.close()

    def plot_model_comparison(self):
        """绘制模型对比图"""
        if not self.all_metrics:
            logger.warning("无模型评估数据")
            return

        setup_chinese_font()

        metrics_df = pd.DataFrame({
            name: {
                "训练集准确率": m["训练集准确率"],
                "测试集准确率": m["测试集准确率"],
                "测试集F1分数": m["测试集F1分数"]
            }
            for name, m in self.all_metrics.items()
        }).T

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        metrics_df[["训练集准确率", "测试集准确率"]].plot(kind="bar", ax=axes[0], color=["#2A9D8F", "#E63946"])
        axes[0].set_title("各模型准确率对比", fontsize=14, fontweight="bold")
        axes[0].set_xlabel("模型", fontsize=12)
        axes[0].set_ylabel("准确率", fontsize=12)
        axes[0].legend(["训练集", "测试集"])
        axes[0].grid(axis="y", alpha=0.3)
        axes[0].set_ylim(0, 1)

        metrics_df["测试集F1分数"].plot(kind="bar", ax=axes[1], color="#457B9D")
        axes[1].set_title("各模型F1分数对比", fontsize=14, fontweight="bold")
        axes[1].set_xlabel("模型", fontsize=12)
        axes[1].set_ylabel("F1分数", fontsize=12)
        axes[1].grid(axis="y", alpha=0.3)
        axes[1].set_ylim(0, 1)

        plt.suptitle("分类模型对比", fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout()

        fig_path = FIGURE_DIR / "task3_model_comparison.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"✓ 模型对比图已保存到: {fig_path}")

        plt.close()

    def save_results(self):
        """保存模型与结果"""
        print("\n" + "=" * 60)
        print("Step 5: 保存模型与结果")
        print("=" * 60)

        if self.best_model and self.trainer:
            model_path = MODEL_DIR / f"task3_{self.best_model_name}_model.pkl"
            joblib.dump(self.best_model, model_path)
            print(f"✓ {self.best_model_name}模型已保存到: {model_path}")

            scaler_path = MODEL_DIR / "task3_scaler.pkl"
            joblib.dump(self.trainer.scaler, scaler_path)
            print(f"✓ 标准化器已保存到: {scaler_path}")

        for name, m in self.all_metrics.items():
            metrics_df = pd.DataFrame([{
                "模型": name,
                "训练集准确率": m["训练集准确率"],
                "测试集准确率": m["测试集准确率"],
                "测试集精确率": m["测试集精确率"],
                "测试集召回率": m["测试集召回率"],
                "测试集F1分数": m["测试集F1分数"]
            }])
            metrics_path = TABLE_DIR / f"task3_{name}_metrics.csv"
            metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
            print(f"✓ {name}评估指标已保存到: {metrics_path}")

        comparison_df = pd.DataFrame({
            name: {
                "训练集准确率": m["训练集准确率"],
                "测试集准确率": m["测试集准确率"],
                "测试集精确率": m["测试集精确率"],
                "测试集召回率": m["测试集召回率"],
                "测试集F1分数": m["测试集F1分数"]
            }
            for name, m in self.all_metrics.items()
        }).T
        comparison_path = TABLE_DIR / "task3_model_comparison.csv"
        comparison_df.to_csv(comparison_path, encoding="utf-8-sig")
        print(f"✓ 模型对比表已保存到: {comparison_path}")

        if hasattr(self.trainer, 'confusion_matrix_df'):
            cm_path = TABLE_DIR / f"task3_confusion_matrix_{self.best_model_name}.csv"
            self.trainer.confusion_matrix_df.to_csv(cm_path, encoding="utf-8-sig")
            print(f"✓ 混淆矩阵已保存到: {cm_path}")

        self.generate_report()

        return True

    def generate_report(self):
        """生成报告"""
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("Task3 第二小问：比赛胜负分类预测模型报告")
        report_lines.append("=" * 70)
        report_lines.append("")

        report_lines.append("一、数据集划分")
        report_lines.append("-" * 40)
        report_lines.append("训练集: 80%随机抽取")
        report_lines.append(f"训练集样本数: {len(self.trainer.X_train) if hasattr(self.trainer, 'X_train') else '未知'}")
        report_lines.append("测试集: 20%随机抽取")
        report_lines.append(f"测试集样本数: {len(self.trainer.X_test) if hasattr(self.trainer, 'X_test') else '未知'}")
        report_lines.append(f"特征数量: {len(self.feature_columns)}")
        report_lines.append(f"特征列表: {', '.join(self.feature_columns)}")
        report_lines.append("")

        report_lines.append("二、模型对比")
        report_lines.append("-" * 40)
        comparison_df = pd.DataFrame({
            name: {
                "训练集准确率": m["训练集准确率"],
                "测试集准确率": m["测试集准确率"],
                "测试集F1分数": m["测试集F1分数"],
                "过拟合程度": m["训练集准确率"] - m["测试集准确率"]
            }
            for name, m in self.all_metrics.items()
        }).T
        comparison_df = comparison_df.sort_values("测试集准确率", ascending=False)
        report_lines.append(comparison_df.to_string())
        report_lines.append("")

        report_lines.append("模型选择理由:")
        for name, row in comparison_df.iterrows():
            is_best = name == self.best_model_name
            marker = "★" if is_best else " "
            report_lines.append(f"  {marker} {name}:")
            report_lines.append(f"    准确率: {row['测试集准确率']:.4f}")
            report_lines.append(f"    F1分数: {row['测试集F1分数']:.4f}")
            report_lines.append(f"    过拟合程度: {row['过拟合程度']:.4f}")
            if is_best:
                report_lines.append(f"    选择理由: 测试集准确率最高，泛化能力最好")
        report_lines.append("")

        report_lines.append("三、评估指标")
        report_lines.append("-" * 40)
        if self.all_metrics and self.best_model_name:
            m = self.all_metrics[self.best_model_name]
            report_lines.append(f"训练集准确率: {m['训练集准确率']:.4f} ({m['训练集准确率']*100:.2f}%)")
            report_lines.append(f"测试集准确率: {m['测试集准确率']:.4f} ({m['测试集准确率']*100:.2f}%)")
            report_lines.append(f"测试集精确率: {m['测试集精确率']:.4f}")
            report_lines.append(f"测试集召回率: {m['测试集召回率']:.4f}")
            report_lines.append(f"测试集F1分数: {m['测试集F1分数']:.4f}")
        report_lines.append("")

        report_lines.append("四、方法局限性")
        report_lines.append("-" * 40)
        report_lines.append("1. 历史数据假设: 假设队伍历史表现能够预测未来表现，")
        report_lines.append("   但队伍实力可能随时间变化(如教练更替、球员换代)。")
        report_lines.append("2. 样本量限制: 足球比赛结果受多种因素影响，历史数据有限。")
        report_lines.append("3. 特征局限性: 仅使用历史统计特征，未考虑临场因素。")
        report_lines.append("4. 赛事规则变化: 不同年份世界杯的赛制、规则可能不同。")
        report_lines.append("5. 新队伍问题: 首次参赛的队伍没有历史数据，使用默认值。")
        report_lines.append("")

        report_lines.append("五、预测函数使用说明")
        report_lines.append("-" * 40)
        report_lines.append("可通过 predict() 函数输入两支队伍名称进行预测:")
        report_lines.append("参数说明:")
        report_lines.append("  home_team: 主队名称(如: Brazil, Germany等)")
        report_lines.append("  away_team: 客队名称")
        report_lines.append("")
        report_lines.append("返回结果:")
        report_lines.append("  预测结果: 主队胜 / 平局 / 客队胜")
        report_lines.append("  各结果概率: 主队胜概率、平局概率、客队胜概率")
        report_lines.append("  置信度: 高(>70%) / 中(50%-70%) / 低(<50%)")
        report_lines.append("  使用模型: 当前最优模型名称")
        report_lines.append("")

        report_lines.append("=" * 70)
        report_lines.append("报告生成完毕")
        report_lines.append("=" * 70)

        report = "\n".join(report_lines)

        report_path = REPORT_DIR / f"task3_{self.best_model_name}_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"✓ 分类预测报告已保存到: {report_path}")

        return report_path

    def print_summary(self):
        """打印预测总结"""
        print("\n" + "=" * 70)
        print("Task3 第二小问：模型预测总结")
        print("=" * 70)

        if self.all_metrics and self.best_model_name:
            m = self.all_metrics[self.best_model_name]
            print("\n一、预测成功率统计")
            print("-" * 40)
            print(f"最优模型: {self.best_model_name}")
            print(f"测试集预测成功率(准确率): {m['测试集准确率']*100:.2f}%")
            print(f"测试集精确率: {m['测试集精确率']*100:.2f}%")
            print(f"测试集召回率: {m['测试集召回率']*100:.2f}%")
            print(f"测试集F1分数: {m['测试集F1分数']*100:.2f}%")

    def interactive_predict(self):
        """交互式预测"""
        print("\n" + "=" * 60)
        print("交互式预测")
        print("=" * 60)
        print("输入两支队伍名称进行预测，输入 'q' 退出")
        print("")

        while True:
            try:
                home_team = input("请输入主队名称: ").strip()
            except EOFError:
                print("\n检测到非交互式环境，跳过交互式预测")
                break

            if home_team.lower() == 'q':
                break

            try:
                away_team = input("请输入客队名称: ").strip()
            except EOFError:
                print("\n检测到非交互式环境，跳过交互式预测")
                break

            if away_team.lower() == 'q':
                break

            if not home_team or not away_team:
                print("队伍名称不能为空，请重新输入")
                continue

            result = self.predict(home_team, away_team)

            if "error" in result:
                print(f"预测失败: {result['error']}")
                continue

            print(f"\n预测结果: {home_team} vs {away_team}")
            print(f"预测: {result['预测结果']}")
            print(f"主队胜概率: {format_probability(result['主队胜概率'])}")
            print(f"平局概率: {format_probability(result['平局概率'])}")
            print(f"客队胜概率: {format_probability(result['客队胜概率'])}")
            print(f"置信度: {result['置信度']}")
            print(f"使用模型: {result['使用模型']}")
            print("")

    def run(self):
        """运行完整流程"""
        print("\n" + "=" * 60)
        print("阶段1: 80%/20%数据划分训练与评估")
        print("=" * 60)

        self.split_data()
        self.preprocess()
        self.train_all_models()
        self.evaluate()

        print("\n" + "=" * 60)
        print(f"阶段2: 用全量数据重新训练最佳模型({self.best_model_name})")
        print("=" * 60)

        self.retrain_with_full_data()

        print("\n" + "=" * 60)
        print(f"阶段3: {self.best_model_name} 参数调优")
        print("=" * 60)

        self.tune_hyperparameters()

        self.save_results()

        if "Brazil" in self.team_history["队伍名称"].values:
            self.predict_2026("Brazil", "Germany")
        else:
            top_team = self.team_history.iloc[0]["队伍名称"]
            second_team = self.team_history.iloc[1]["队伍名称"] if len(self.team_history) > 1 else top_team
            self.predict_2026(top_team, second_team)

        self.print_summary()

        return self.best_model
