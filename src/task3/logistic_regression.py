"""
logistic_regression.py
==========================
Task3 第二小问：基于逻辑回归的比赛胜负分类预测
==========================
功能：
1. 划分训练集（1930-2010）和测试集（2014）
2. 特征标准化与分类模型训练（含模型对比）
3. 自动计算准确率、混淆矩阵并保存图片
4. 封装预测接口，输入2026参赛队伍历史数据预测胜负
5. 模型对比：逻辑回归、随机森林、梯度提升
==========================
"""

import pandas as pd
import numpy as np
import joblib
import random
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from config import REPORT_DIR, MODEL_DIR, TABLE_DIR, FIGURE_DIR

RANDOM_SEED = 42


def setup_random_seed(seed=RANDOM_SEED):

    random.seed(seed)
    np.random.seed(seed)


def setup_chinese_font():

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.family"] = "sans-serif"
    sns.set_theme(style="whitegrid", font="Microsoft YaHei")


class MatchResultClassifier:
    """
    基于逻辑回归的比赛胜负分类预测器（含模型对比）
    """

    def __init__(self, feature_matrix, team_history):

        setup_random_seed()

        self.feature_matrix = feature_matrix.copy()
        self.team_history = team_history.copy()

        self.feature_columns = [
            "主队历史参赛次数", "主队历史比赛场次", "主队历史场均进球",
            "主队历史场均失球", "主队历史净胜球", "主队历史成绩排名",
            "主队近3届场均进球", "主队近3届胜率",
            "客队历史参赛次数", "客队历史比赛场次", "客队历史场均进球",
            "客队历史场均失球", "客队历史净胜球", "客队历史成绩排名",
            "客队近3届场均进球", "客队近3届胜率",
            "参赛次数差", "场均进球差", "场均失球差",
            "净胜球差", "成绩排名差",
            "近3届场均进球差", "近3届胜率差",
            "交锋胜场", "交锋平局", "交锋负场", "交锋净胜球"
        ]

        self.feature_columns = [c for c in self.feature_columns
                               if c in self.feature_matrix.columns]

        self.target_column = "胜负结果"

        self.scaler = StandardScaler()

        self.models = {
            "逻辑回归": LogisticRegression(
                solver="lbfgs",
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_SEED
            ),
            "随机森林": RandomForestClassifier(
                n_estimators=200,
                max_depth=5,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=RANDOM_SEED
            ),
            "梯度提升": GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=RANDOM_SEED
            )
        }

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.best_model_name = None
        self.best_model = None
        self.all_metrics = {}

    def split_data(self):

        print("\n" + "=" * 60)
        print("Step 1: 数据集划分")
        print("=" * 60)
        print(f"随机种子: {RANDOM_SEED}")
        print(f"训练集: 1930-2010年")
        print(f"测试集: 2014年")

        available_features = [c for c in self.feature_columns
                             if c in self.feature_matrix.columns]
        self.feature_columns = available_features

        train_df = self.feature_matrix[self.feature_matrix["年份"] <= 2010].copy()
        test_df = self.feature_matrix[self.feature_matrix["年份"] == 2014].copy()

        print(f"\n训练集年份范围: {train_df['年份'].min()} - {train_df['年份'].max()}")
        print(f"训练集样本数: {len(train_df)}")

        print(f"\n测试集年份: {test_df['年份'].min() if len(test_df) > 0 else '无'}")
        print(f"测试集样本数: {len(test_df)}")

        if len(test_df) == 0:
            print("警告: 2014年数据不足，使用2010年以后数据作为测试集")
            test_df = self.feature_matrix[self.feature_matrix["年份"] > 2010].copy()
            print(f"调整后测试集样本数: {len(test_df)}")

        print(f"\n训练集胜负分布:")
        print(train_df[self.target_column].value_counts())
        print(f"\n测试集胜负分布:")
        print(test_df[self.target_column].value_counts())

        self.X_train = train_df[self.feature_columns]
        self.y_train = train_df[self.target_column]
        self.X_test = test_df[self.feature_columns]
        self.y_test = test_df[self.target_column]

        self.test_df = test_df

        return self.X_train, self.X_test, self.y_train, self.y_test

    def preprocess(self):

        print("\n" + "=" * 60)
        print("Step 2: 特征标准化")
        print("=" * 60)

        print(f"\n缺失值统计:")
        print(f"  训练集缺失值: {self.X_train.isnull().sum().sum()}")
        print(f"  测试集缺失值: {self.X_test.isnull().sum().sum()}")

        for col in self.feature_columns:
            if self.X_train[col].isnull().any():
                median_val = self.X_train[col].median()
                if pd.isna(median_val):
                    median_val = 0
                self.X_train[col] = self.X_train[col].fillna(median_val)
                self.X_test[col] = self.X_test[col].fillna(median_val)
                print(f"  {col}: 用中位数 {median_val:.4f} 填充")

        self.X_train = self.X_train.reset_index(drop=True)
        self.y_train = self.y_train.reset_index(drop=True)
        self.X_test = self.X_test.reset_index(drop=True)
        self.y_test = self.y_test.reset_index(drop=True)

        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        print(f"\n✓ 训练集标准化完成，形状: {self.X_train_scaled.shape}")
        print(f"✓ 测试集标准化完成，形状: {self.X_test_scaled.shape}")

        return self.X_train_scaled, self.X_test_scaled

    def train_all_models(self):

        print("\n" + "=" * 60)
        print("Step 3: 模型训练与对比")
        print("=" * 60)

        for name, model in self.models.items():
            print(f"\n训练 {name}...")

            if name == "逻辑回归":
                model.fit(self.X_train_scaled, self.y_train)
            else:
                model.fit(self.X_train, self.y_train)

            y_train_pred = model.predict(self.X_train_scaled) if name == "逻辑回归" else model.predict(self.X_train)
            y_test_pred = model.predict(self.X_test_scaled) if name == "逻辑回归" else model.predict(self.X_test)

            train_accuracy = accuracy_score(self.y_train, y_train_pred)
            test_accuracy = accuracy_score(self.y_test, y_test_pred)
            test_precision = precision_score(self.y_test, y_test_pred, average="weighted", zero_division=0)
            test_recall = recall_score(self.y_test, y_test_pred, average="weighted", zero_division=0)
            test_f1 = f1_score(self.y_test, y_test_pred, average="weighted", zero_division=0)

            self.all_metrics[name] = {
                "训练集准确率": train_accuracy,
                "测试集准确率": test_accuracy,
                "测试集精确率": test_precision,
                "测试集召回率": test_recall,
                "测试集F1分数": test_f1,
                "混淆矩阵": confusion_matrix(self.y_test, y_test_pred),
                "y_test_pred": y_test_pred
            }

            print(f"  训练集准确率: {train_accuracy:.4f}")
            print(f"  测试集准确率: {test_accuracy:.4f}")
            print(f"  测试集F1分数: {test_f1:.4f}")

        self._select_best_model()

        return self.all_metrics

    def _select_best_model(self):

        print("\n" + "-" * 60)
        print("模型对比与选择")
        print("-" * 60)

        metrics_df = pd.DataFrame({
            name: {
                "训练集准确率": m["训练集准确率"],
                "测试集准确率": m["测试集准确率"],
                "测试集F1分数": m["测试集F1分数"],
                "过拟合程度": m["训练集准确率"] - m["测试集准确率"]
            }
            for name, m in self.all_metrics.items()
        }).T

        metrics_df = metrics_df.sort_values("测试集准确率", ascending=False)

        print(f"\n模型对比表:")
        print("-" * 60)
        print(metrics_df.to_string())

        self.best_model_name = metrics_df.index[0]
        self.best_model = self.models[self.best_model_name]
        self.best_model_metrics = self.all_metrics[self.best_model_name]

        print(f"\n✓ 最优模型: {self.best_model_name}")
        print(f"  测试集准确率: {self.best_model_metrics['测试集准确率']:.4f}")
        print(f"  测试集F1分数: {self.best_model_metrics['测试集F1分数']:.4f}")

        print("\n模型选择理由:")
        for name, row in metrics_df.iterrows():
            is_best = name == self.best_model_name
            marker = "★" if is_best else " "
            print(f"  {marker} {name}:")
            print(f"    准确率: {row['测试集准确率']:.4f}")
            print(f"    F1分数: {row['测试集F1分数']:.4f}")
            print(f"    过拟合程度: {row['过拟合程度']:.4f}")

        return self.best_model

    def evaluate(self):

        print("\n" + "=" * 60)
        print(f"Step 4: {self.best_model_name} 详细评估")
        print("=" * 60)

        m = self.best_model_metrics

        print(f"\n测试集评估指标:")
        print(f"  准确率 (Accuracy):  {m['测试集准确率']:.4f} ({m['测试集准确率']*100:.2f}%)")
        print(f"  精确率 (Precision): {m['测试集精确率']:.4f}")
        print(f"  召回率 (Recall):    {m['测试集召回率']:.4f}")
        print(f"  F1分数 (F1-Score):  {m['测试集F1分数']:.4f}")

        print("\n" + "-" * 60)
        print("评估指标解释：")
        print("-" * 60)
        print(f"准确率 (Accuracy) = {m['测试集准确率']:.4f}")
        print("  含义: 预测正确的样本数占总样本数的比例")
        print(f"  解释: 模型在测试集上{m['测试集准确率']*100:.2f}%的预测是正确的")
        print("  范围: [0, 1]，越大越好，1表示全部预测正确")
        print()
        print(f"精确率 (Precision) = {m['测试集精确率']:.4f}")
        print("  含义: 预测为某类的样本中实际为该类的比例(加权平均)")
        print(f"  解释: 模型预测为某类时，{m['测试集精确率']*100:.2f}%的预测是准确的")
        print("  范围: [0, 1]，越大越好")
        print()
        print(f"召回率 (Recall) = {m['测试集召回率']:.4f}")
        print("  含义: 实际为某类的样本中被正确预测的比例(加权平均)")
        print(f"  解释: 实际某类的样本中，{m['测试集召回率']*100:.2f}%被正确预测")
        print("  范围: [0, 1]，越大越好")
        print()
        print(f"F1分数 (F1-Score) = {m['测试集F1分数']:.4f}")
        print("  含义: 精确率和召回率的调和平均数")
        print(f"  解释: 综合评价指标，{m['测试集F1分数']*100:.2f}%")
        print("  范围: [0, 1]，越大越好，1表示完美分类")
        print()

        overfitting = m['训练集准确率'] - m['测试集准确率']
        print(f"过拟合分析:")
        print(f"  训练集准确率 - 测试集准确率 = {overfitting:.4f}")
        if overfitting > 0.2:
            print("  结论: 差值较大，模型存在明显过拟合")
        elif overfitting > 0.1:
            print("  结论: 差值适中，模型有轻微过拟合")
        else:
            print("  结论: 差值较小，模型泛化能力良好")

        cm = m["混淆矩阵"]
        self.confusion_matrix = cm
        cm_df = pd.DataFrame(cm, index=[f"实际_{c}" for c in self.best_model.classes_],
                            columns=[f"预测_{c}" for c in self.best_model.classes_])
        self.confusion_matrix_df = cm_df

        print(f"\n混淆矩阵:")
        print("-" * 40)
        print(cm_df)

        print(f"\n详细分类报告:")
        print("-" * 40)
        y_test_pred = m["y_test_pred"]
        print(classification_report(self.y_test, y_test_pred, zero_division=0))

        self.y_test_pred = y_test_pred

        self.plot_confusion_matrix()
        self.plot_model_comparison()

        return m

    def plot_confusion_matrix(self):

        print("\n绘制混淆矩阵图")

        setup_chinese_font()

        plt.figure(figsize=(10, 8))

        sns.heatmap(
            self.confusion_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.best_model.classes_,
            yticklabels=self.best_model.classes_,
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

        print("绘制模型对比图")

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

    def predict(self, home_team, away_team):

        print("\n" + "=" * 60)
        print(f"预测比赛结果: {home_team} vs {away_team}")
        print("=" * 60)

        history_dict = self.team_history.set_index("队伍名称").to_dict("index")

        home_hist = history_dict.get(home_team)
        away_hist = history_dict.get(away_team)

        if home_hist is None:
            print(f"警告: 队伍 '{home_team}' 无历史数据，使用默认值")
            home_hist = {
                "历史参赛次数": 0, "历史比赛场次": 0,
                "历史场均进球": 0, "历史场均失球": 0,
                "历史净胜球": 0, "历史成绩排名": 0,
                "近3届场均进球": 0, "近3届胜率": 0,
                "交锋胜场": 0, "交锋平局": 0, "交锋负场": 0, "交锋净胜球": 0
            }

        if away_hist is None:
            print(f"警告: 队伍 '{away_team}' 无历史数据，使用默认值")
            away_hist = {
                "历史参赛次数": 0, "历史比赛场次": 0,
                "历史场均进球": 0, "历史场均失球": 0,
                "历史净胜球": 0, "历史成绩排名": 0,
                "近3届场均进球": 0, "近3届胜率": 0,
                "交锋胜场": 0, "交锋平局": 0, "交锋负场": 0, "交锋净胜球": 0
            }

        input_data = {
            "主队历史参赛次数": home_hist["历史参赛次数"],
            "主队历史比赛场次": home_hist["历史比赛场次"],
            "主队历史场均进球": home_hist["历史场均进球"],
            "主队历史场均失球": home_hist["历史场均失球"],
            "主队历史净胜球": home_hist["历史净胜球"],
            "主队历史成绩排名": home_hist["历史成绩排名"],
            "主队近3届场均进球": home_hist.get("近3届场均进球", 0),
            "主队近3届胜率": home_hist.get("近3届胜率", 0),
            "客队历史参赛次数": away_hist["历史参赛次数"],
            "客队历史比赛场次": away_hist["历史比赛场次"],
            "客队历史场均进球": away_hist["历史场均进球"],
            "客队历史场均失球": away_hist["历史场均失球"],
            "客队历史净胜球": away_hist["历史净胜球"],
            "客队历史成绩排名": away_hist["历史成绩排名"],
            "客队近3届场均进球": away_hist.get("近3届场均进球", 0),
            "客队近3届胜率": away_hist.get("近3届胜率", 0),
            "参赛次数差": home_hist["历史参赛次数"] - away_hist["历史参赛次数"],
            "场均进球差": home_hist["历史场均进球"] - away_hist["历史场均进球"],
            "场均失球差": home_hist["历史场均失球"] - away_hist["历史场均失球"],
            "净胜球差": home_hist["历史净胜球"] - away_hist["历史净胜球"],
            "成绩排名差": home_hist["历史成绩排名"] - away_hist["历史成绩排名"],
            "近3届场均进球差": home_hist.get("近3届场均进球", 0) - away_hist.get("近3届场均进球", 0),
            "近3届胜率差": home_hist.get("近3届胜率", 0) - away_hist.get("近3届胜率", 0),
            "交锋胜场": home_hist.get("交锋胜场", 0),
            "交锋平局": home_hist.get("交锋平局", 0),
            "交锋负场": home_hist.get("交锋负场", 0),
            "交锋净胜球": home_hist.get("交锋净胜球", 0)
        }

        input_df = pd.DataFrame([input_data])
        input_df = input_df[self.feature_columns].fillna(0)

        if self.best_model_name == "逻辑回归":
            input_scaled = self.scaler.transform(input_df)
            prediction = self.best_model.predict(input_scaled)[0]
            probabilities = self.best_model.predict_proba(input_scaled)[0]
        else:
            prediction = self.best_model.predict(input_df)[0]
            probabilities = self.best_model.predict_proba(input_df)[0]

        classes = self.best_model.classes_

        print(f"\n两队特征对比:")
        print("-" * 70)
        print(f"{'特征':<20} {home_team:<15} {away_team:<15} {'差距':<15}")
        print("-" * 70)
        print(f"{'历史参赛次数':<20} {home_hist['历史参赛次数']:<15} {away_hist['历史参赛次数']:<15} {input_data['参赛次数差']:<15}")
        print(f"{'历史比赛场次':<20} {home_hist['历史比赛场次']:<15} {away_hist['历史比赛场次']:<15} {home_hist['历史比赛场次'] - away_hist['历史比赛场次']:<15}")
        print(f"{'历史场均进球':<20} {home_hist['历史场均进球']:<15.4f} {away_hist['历史场均进球']:<15.4f} {input_data['场均进球差']:<15.4f}")
        print(f"{'历史场均失球':<20} {home_hist['历史场均失球']:<15.4f} {away_hist['历史场均失球']:<15.4f} {input_data['场均失球差']:<15.4f}")
        print(f"{'历史净胜球':<20} {home_hist['历史净胜球']:<15} {away_hist['历史净胜球']:<15} {input_data['净胜球差']:<15}")
        print(f"{'历史成绩排名':<20} {home_hist['历史成绩排名']:<15} {away_hist['历史成绩排名']:<15} {input_data['成绩排名差']:<15}")
        print(f"{'近3届场均进球':<20} {home_hist.get('近3届场均进球', 0):<15.4f} {away_hist.get('近3届场均进球', 0):<15.4f} {input_data['近3届场均进球差']:<15.4f}")
        print(f"{'近3届胜率':<20} {home_hist.get('近3届胜率', 0):<15.4f} {away_hist.get('近3届胜率', 0):<15.4f} {input_data['近3届胜率差']:<15.4f}")
        if home_hist.get("交锋胜场", 0) > 0 or home_hist.get("交锋平局", 0) > 0 or home_hist.get("交锋负场", 0) > 0:
            total_h2h = home_hist.get("交锋胜场", 0) + home_hist.get("交锋平局", 0) + home_hist.get("交锋负场", 0)
            print(f"{'交锋记录':<20} {home_hist.get('交锋胜场', 0)}胜{home_hist.get('交锋平局', 0)}平{home_hist.get('交锋负场', 0)}负")

        print(f"\n预测结果:")
        print("-" * 40)
        print(f"  比赛结果: {prediction}")
        print(f"\n各结果概率:")
        for cls, prob in zip(classes, probabilities):
            print(f"  {cls}: {prob*100:.2f}%")

        max_prob = max(probabilities)
        confidence = "高" if max_prob > 0.7 else "中" if max_prob > 0.5 else "低"
        print(f"\n预测置信度: {confidence} (最高概率 {max_prob*100:.2f}%)")

        result = {
            "主队": home_team,
            "客队": away_team,
            "预测结果": prediction,
            "主队胜概率": dict(zip(classes, probabilities)).get("主队胜", 0),
            "平局概率": dict(zip(classes, probabilities)).get("平局", 0),
            "客队胜概率": dict(zip(classes, probabilities)).get("客队胜", 0),
            "置信度": confidence,
            "使用模型": self.best_model_name
        }

        return result

    def predict_2026(self, home_team, away_team):

        print("\n" + "=" * 60)
        print("预测2026年世界杯比赛结果")
        print("=" * 60)

        return self.predict(home_team, away_team)

    def save_results(self):

        print("\n" + "=" * 60)
        print("Step 5: 保存模型与结果")
        print("=" * 60)

        model_path = MODEL_DIR / f"task3_{self.best_model_name}_model.pkl"
        joblib.dump(self.best_model, model_path)
        print(f"✓ {self.best_model_name}模型已保存到: {model_path}")

        scaler_path = MODEL_DIR / "task3_scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
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

        cm_path = TABLE_DIR / f"task3_confusion_matrix_{self.best_model_name}.csv"
        self.confusion_matrix_df.to_csv(cm_path, encoding="utf-8-sig")
        print(f"✓ 混淆矩阵已保存到: {cm_path}")

        test_result_df = pd.DataFrame({
            "实际结果": self.y_test.values,
            "预测结果": self.y_test_pred
        })
        test_result_path = TABLE_DIR / f"task3_test_predictions_{self.best_model_name}.csv"
        test_result_df.to_csv(test_result_path, index=False, encoding="utf-8-sig")
        print(f"✓ 测试集预测结果已保存到: {test_result_path}")

        self.generate_report()

        return True

    def generate_report(self):

        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("Task3 第二小问：比赛胜负分类预测模型报告")
        report_lines.append("=" * 70)
        report_lines.append("")

        report_lines.append("一、数据集划分")
        report_lines.append("-" * 40)
        report_lines.append(f"随机种子: {RANDOM_SEED}")
        report_lines.append("训练集: 1930-2010年世界杯比赛数据")
        report_lines.append(f"训练集样本数: {len(self.X_train)}")
        report_lines.append("测试集: 2014年世界杯比赛数据")
        report_lines.append(f"测试集样本数: {len(self.X_test)}")
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

        report_lines.append("三、最优模型参数")
        report_lines.append("-" * 40)
        report_lines.append(f"模型类型: {self.best_model_name}")
        if self.best_model_name == "逻辑回归":
            report_lines.append(f"求解器: {self.best_model.solver}")
            report_lines.append(f"最大迭代次数: {self.best_model.max_iter}")
        elif self.best_model_name == "随机森林":
            report_lines.append(f"决策树数量: {self.best_model.n_estimators}")
            report_lines.append(f"最大深度: {self.best_model.max_depth}")
            report_lines.append(f"最小样本分割数: {self.best_model.min_samples_split}")
        elif self.best_model_name == "梯度提升":
            report_lines.append(f"弱学习器数量: {self.best_model.n_estimators}")
            report_lines.append(f"最大深度: {self.best_model.max_depth}")
            report_lines.append(f"学习率: {self.best_model.learning_rate}")
        report_lines.append(f"类别: {list(self.best_model.classes_)}")
        report_lines.append("")

        if self.best_model_name == "逻辑回归":
            report_lines.append("模型系数:")
            for i, class_name in enumerate(self.best_model.classes_):
                report_lines.append(f"类别 '{class_name}' 的系数:")
                for feature, coef in zip(self.feature_columns, self.best_model.coef_[i]):
                    report_lines.append(f"  {feature}: {coef:.4f}")
                report_lines.append(f"  截距: {self.best_model.intercept_[i]:.4f}")
            report_lines.append("")

        report_lines.append("四、评估指标")
        report_lines.append("-" * 40)
        m = self.best_model_metrics
        report_lines.append(f"训练集准确率: {m['训练集准确率']:.4f} ({m['训练集准确率']*100:.2f}%)")
        report_lines.append(f"测试集准确率: {m['测试集准确率']:.4f} ({m['测试集准确率']*100:.2f}%)")
        report_lines.append(f"测试集精确率: {m['测试集精确率']:.4f}")
        report_lines.append(f"测试集召回率: {m['测试集召回率']:.4f}")
        report_lines.append(f"测试集F1分数: {m['测试集F1分数']:.4f}")
        report_lines.append("")

        report_lines.append("评估指标详细解释:")
        report_lines.append("-" * 40)
        report_lines.append(f"准确率 (Accuracy) = {m['测试集准确率']:.4f}")
        report_lines.append("  计算公式: Accuracy = (TP + TN) / (TP + TN + FP + FN)")
        report_lines.append("  含义: 预测正确的样本数占总样本数的比例")
        report_lines.append(f"  解释: 模型在测试集上{m['测试集准确率']*100:.2f}%的预测是正确的")
        report_lines.append("  范围: [0, 1]，越大越好，1表示全部预测正确")
        report_lines.append("")
        report_lines.append(f"精确率 (Precision) = {m['测试集精确率']:.4f}")
        report_lines.append("  计算公式: Precision = TP / (TP + FP)")
        report_lines.append("  含义: 预测为某类的样本中实际为该类的比例(加权平均)")
        report_lines.append(f"  解释: 模型预测为某类时，{m['测试集精确率']*100:.2f}%的预测是准确的")
        report_lines.append("  范围: [0, 1]，越大越好")
        report_lines.append("")
        report_lines.append(f"召回率 (Recall) = {m['测试集召回率']:.4f}")
        report_lines.append("  计算公式: Recall = TP / (TP + FN)")
        report_lines.append("  含义: 实际为某类的样本中被正确预测的比例(加权平均)")
        report_lines.append(f"  解释: 实际某类的样本中，{m['测试集召回率']*100:.2f}%被正确预测")
        report_lines.append("  范围: [0, 1]，越大越好")
        report_lines.append("")
        report_lines.append(f"F1分数 (F1-Score) = {m['测试集F1分数']:.4f}")
        report_lines.append("  计算公式: F1 = 2 * (Precision * Recall) / (Precision + Recall)")
        report_lines.append("  含义: 精确率和召回率的调和平均数")
        report_lines.append(f"  解释: 综合评价指标，{m['测试集F1分数']*100:.2f}%")
        report_lines.append("  范围: [0, 1]，越大越好，1表示完美分类")
        report_lines.append("")

        overfitting = m['训练集准确率'] - m['测试集准确率']
        report_lines.append("过拟合分析:")
        report_lines.append(f"  训练集准确率 - 测试集准确率 = {overfitting:.4f}")
        if overfitting > 0.2:
            report_lines.append("  结论: 差值较大，模型存在明显过拟合")
        elif overfitting > 0.1:
            report_lines.append("  结论: 差值适中，模型有轻微过拟合")
        else:
            report_lines.append("  结论: 差值较小，模型泛化能力良好")
        report_lines.append("")

        report_lines.append("五、混淆矩阵")
        report_lines.append("-" * 40)
        report_lines.append(self.confusion_matrix_df.to_string())
        report_lines.append("")

        report_lines.append("六、方法局限性")
        report_lines.append("-" * 40)
        report_lines.append("1. 历史数据假设: 假设队伍历史表现能够预测未来表现，")
        report_lines.append("   但队伍实力可能随时间变化(如教练更替、球员换代)。")
        report_lines.append("2. 样本量限制: 足球比赛结果受多种因素影响，历史数据有限。")
        report_lines.append("3. 特征局限性: 仅使用历史统计特征，未考虑临场因素。")
        report_lines.append("4. 赛事规则变化: 不同年份世界杯的赛制、规则可能不同。")
        report_lines.append("5. 新队伍问题: 首次参赛的队伍没有历史数据，使用默认值。")
        report_lines.append("")

        report_lines.append("七、预测函数使用说明")
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

    def run(self):

        self.split_data()
        self.preprocess()
        self.train_all_models()
        self.evaluate()
        self.save_results()

        if "Brazil" in self.team_history["队伍名称"].values:
            self.predict_2026("Brazil", "Germany")
        else:
            top_team = self.team_history.iloc[0]["队伍名称"]
            second_team = self.team_history.iloc[1]["队伍名称"] if len(self.team_history) > 1 else top_team
            self.predict_2026(top_team, second_team)

        self.print_summary()

        self.interactive_predict()

        return self.best_model

    def print_summary(self):

        print("\n" + "=" * 70)
        print("Task3 第二小问：模型预测总结")
        print("=" * 70)

        print("\n一、预测成功率统计")
        print("-" * 40)
        m = self.best_model_metrics
        print(f"最优模型: {self.best_model_name}")
        print(f"测试集预测成功率(准确率): {m['测试集准确率']*100:.2f}%")
        print(f"测试集精确率: {m['测试集精确率']*100:.2f}%")
        print(f"测试集召回率: {m['测试集召回率']*100:.2f}%")
        print(f"测试集F1分数: {m['测试集F1分数']*100:.2f}%")

        cm = self.confusion_matrix
        tp = cm[0, 0] + cm[1, 1] + cm[2, 2] if cm.shape[0] >= 3 else cm[0, 0] + cm[1, 1]
        total = cm.sum()
        print(f"\n混淆矩阵分析:")
        print(f"  正确预测: {tp} / {total}")
        print(f"  错误预测: {total - tp} / {total}")

        print("\n二、各结果预测成功率")
        print("-" * 40)
        y_test_pred = self.y_test_pred
        for cls in self.best_model.classes_:
            mask = self.y_test == cls
            if mask.sum() > 0:
                correct = (y_test_pred[mask] == cls).sum()
                rate = correct / mask.sum()
                print(f"  {cls}: {correct}/{mask.sum()} ({rate*100:.2f}%)")

        print("\n三、模型对比总结")
        print("-" * 40)
        metrics_df = pd.DataFrame({
            name: {
                "训练集准确率": m["训练集准确率"],
                "测试集准确率": m["测试集准确率"],
                "测试集F1分数": m["测试集F1分数"],
                "过拟合程度": m["训练集准确率"] - m["测试集准确率"]
            }
            for name, m in self.all_metrics.items()
        }).T
        metrics_df = metrics_df.sort_values("测试集准确率", ascending=False)

        print(f"模型对比表:")
        print("-" * 60)
        print(metrics_df.to_string())

        print(f"\n模型选择分析:")
        print("-" * 40)
        for i, (name, row) in enumerate(metrics_df.iterrows()):
            is_best = name == self.best_model_name
            marker = "★" if is_best else f" {i+1}."
            print(f"{marker} {name}:")
            print(f"    测试集准确率: {row['测试集准确率']*100:.2f}%")
            print(f"    测试集F1分数: {row['测试集F1分数']*100:.2f}%")
            print(f"    过拟合程度: {row['过拟合程度']*100:.2f}%")
            if is_best:
                print("    ✓ 选择理由: 测试集准确率最高，综合性能最优")
            else:
                print("    ✗ 淘汰理由: 性能不如最优模型")

        print("\n四、模型优缺点分析")
        print("-" * 40)
        model_analysis = {
            "逻辑回归": {
                "优点": ["模型简单透明，可解释性强", "训练速度快，计算效率高", "不易过拟合"],
                "缺点": ["假设线性关系，可能无法捕捉复杂交互", "对特征尺度敏感", "预测能力有限"]
            },
            "随机森林": {
                "优点": ["能捕捉非线性关系和特征交互", "对异常值不敏感", "无需特征标准化"],
                "缺点": ["模型复杂，可解释性较差", "容易过拟合", "训练速度较慢"]
            },
            "梯度提升": {
                "优点": ["精度高，能捕捉复杂模式", "可以处理混合类型数据", "特征重要性可解释"],
                "缺点": ["对参数敏感，需要调参", "容易过拟合", "训练速度较慢"]
            }
        }

        for name, analysis in model_analysis.items():
            is_best = name == self.best_model_name
            marker = "★" if is_best else " "
            print(f"\n{marker} {name}:")
            print("  优点:")
            for advantage in analysis["优点"]:
                print(f"    - {advantage}")
            print("  缺点:")
            for disadvantage in analysis["缺点"]:
                print(f"    - {disadvantage}")

        print("\n" + "=" * 70)
        print("预测模型总结完毕")
        print("=" * 70)

    def interactive_predict(self):

        print("\n\n" + "=" * 70)
        print("交互式预测功能")
        print("=" * 70)
        print("说明: 输入两队名称，预测比赛结果")
        print("提示: 输入 'q' 或 'quit' 退出")
        print("可用队伍示例: Brazil, Germany, Argentina, Italy, Spain, France等")
        print("=" * 70)

        available_teams = sorted(self.team_history["队伍名称"].unique())
        print(f"\n可用队伍列表(前20个):")
        print(", ".join(available_teams[:20]))

        try:
            import sys
            if not sys.stdin.isatty():
                print("\n注意: 当前为非交互式模式，跳过手动输入")
                print("若要使用交互式预测，请直接运行脚本并在终端中输入")
                return
        except:
            pass

        while True:
            print("\n" + "-" * 50)
            try:
                home_team = input("请输入主队名称: ").strip()
            except EOFError:
                print("\n检测到EOF，退出交互式预测")
                break

            if home_team.lower() in ['q', 'quit', 'exit']:
                print("退出交互式预测")
                break

            try:
                away_team = input("请输入客队名称: ").strip()
            except EOFError:
                print("\n检测到EOF，退出交互式预测")
                break

            if away_team.lower() in ['q', 'quit', 'exit']:
                print("退出交互式预测")
                break

            if not home_team or not away_team:
                print("错误: 队伍名称不能为空")
                continue

            try:
                result = self.predict(home_team, away_team)

                print("\n" + "-" * 50)
                print("预测结果总结:")
                print(f"  比赛: {result['主队']} vs {result['客队']}")
                print(f"  预测: {result['预测结果']}")
                print(f"  主队胜概率: {result['主队胜概率']*100:.2f}%")
                print(f"  平局概率: {result['平局概率']*100:.2f}%")
                print(f"  客队胜概率: {result['客队胜概率']*100:.2f}%")
                print(f"  置信度: {result['置信度']}")
                print(f"  使用模型: {result['使用模型']}")

            except Exception as e:
                print(f"预测出错: {e}")
                print("请检查队伍名称是否正确")