"""
regression.py
==========================
Task1 回归预测模块（优化版）
包含：多模型对比、特征工程、集成学习
==========================
"""

import pandas as pd
import numpy as np
import random
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    VotingRegressor, StackingRegressor
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from src.config import MODEL_DIR, REPORT_DIR, FIGURE_DIR

RANDOM_SEED = 42


def setup_random_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)


class OptimizedRegressionModel:

    def __init__(self, df):

        setup_random_seed()

        self.df = df

        self.target_column = "总进球数"

        self.basic_features = [
            "参赛队伍数量",
            "总比赛场次",
            "总观众人数"
        ]

        self.models = {}

        self.best_model = None

        self.best_model_name = None

        self.best_model_metrics = None

        self.prediction_result = None

    def clean_and_preprocess(self):

        print("=" * 60)
        print("数据清洗与特征预处理")
        print("=" * 60)

        self.processed_df = self.df.copy()

        if "总观众人数" in self.processed_df.columns:

            self.processed_df["总观众人数"] = self.processed_df["总观众人数"].astype(str).str.replace(".", "", regex=False)

            self.processed_df["总观众人数"] = pd.to_numeric(self.processed_df["总观众人数"], errors="coerce")

        for col in self.basic_features + [self.target_column]:

            if col in self.processed_df.columns:

                self.processed_df[col] = pd.to_numeric(self.processed_df[col], errors="coerce")

        self.processed_df = self.processed_df.dropna(subset=self.basic_features + [self.target_column])

        self.processed_df = self.processed_df.sort_values("年份").reset_index(drop=True)

        self._create_features()

        print(f"有效样本数：{len(self.processed_df)}")
        print(f"特征数量：{len(self.feature_columns)}")

        print("\n特征数据预览：")
        print(self.processed_df[self.feature_columns + [self.target_column]].head())

        return self.processed_df

    def _create_features(self):

        year_mean = self.processed_df["年份"].mean()
        year_std = self.processed_df["年份"].std()

        self.processed_df["年份_归一化"] = (self.processed_df["年份"] - year_mean) / year_std

        self.processed_df["前一届进球数"] = self.processed_df["总进球数"].shift(1)
        self.processed_df["前两届进球数"] = self.processed_df["总进球数"].shift(2)
        self.processed_df["前一届场均进球"] = self.processed_df["场均进球"].shift(1)

        self.processed_df["队伍场次比"] = self.processed_df["参赛队伍数量"] / self.processed_df["总比赛场次"]

        self.feature_columns = [
            "参赛队伍数量",
            "总比赛场次",
            "总观众人数",
            "年份_归一化",
            "前一届进球数",
            "前两届进球数",
            "前一届场均进球",
            "队伍场次比"
        ]

        for col in ["前一届进球数", "前两届进球数", "前一届场均进球"]:
            if col in self.processed_df.columns:
                self.processed_df[col] = self.processed_df[col].fillna(self.processed_df[col].mean())

        self.processed_df = self.processed_df.dropna(subset=self.feature_columns + [self.target_column])

    def train_models(self):

        print("\n" + "=" * 60)
        print("训练回归模型")
        print("=" * 60)

        X = self.processed_df[self.feature_columns]
        y = self.processed_df[self.target_column]

        self.models_info = []

        models_to_train = [
            ("线性回归", LinearRegression()),
            ("Ridge回归", Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=10.0))])),
            ("Lasso回归", Pipeline([('scaler', StandardScaler()), ('lasso', Lasso(alpha=1.0))])),
            ("ElasticNet", Pipeline([('scaler', StandardScaler()), ('elastic', ElasticNet(alpha=1.0, l1_ratio=0.5))])),
            ("随机森林", RandomForestRegressor(n_estimators=100, max_depth=7, min_samples_leaf=3, random_state=42)),
            ("梯度提升", GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42))
        ]

        stacking_estimators = [
            ('lr', LinearRegression()),
            ('ridge', Ridge(alpha=10.0)),
            ('rf', RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42))
        ]

        all_models = models_to_train + [
            ("Stacking集成", StackingRegressor(estimators=stacking_estimators,
                                              final_estimator=Ridge(alpha=10.0))),
            ("Voting集成", VotingRegressor(estimators=[
                ('gb', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)),
                ('rf', RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42)),
                ('ridge', Ridge(alpha=10.0))
            ]))
        ]

        for name, model in all_models:

            print(f"\n训练 {name}...")

            model.fit(X, y)

            y_pred = model.predict(X)

            r2 = r2_score(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))

            cv_scores = cross_val_score(model, X, y, cv=TimeSeriesSplit(n_splits=5), scoring='r2')
            cv_mae = cross_val_score(model, X, y, cv=TimeSeriesSplit(n_splits=5), scoring='neg_mean_absolute_error')

            self.models[name] = model
            self.models_info.append({
                "模型": name,
                "R2": r2,
                "MAE": mae,
                "RMSE": rmse,
                "CV_R2": cv_scores.mean(),
                "CV_MAE": -cv_mae.mean(),
                "CV_R2_std": cv_scores.std()
            })

            print(f"  R2={r2:.4f}, MAE={mae:.2f}, CV_R2={cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

        print("\n" + "-" * 60)
        print("评估指标解释：")
        print("-" * 60)
        print("R² (决定系数): 衡量模型解释目标变量方差的比例")
        print("  范围: (-∞, 1]，越接近1拟合越好，1为完美拟合")
        print("MAE (平均绝对误差): 预测值与实际值之差的绝对值的平均")
        print("  范围: [0, +∞)，越小越好，单位与目标变量一致(球)")
        print("RMSE (均方根误差): 均方误差的平方根，对异常值敏感")
        print("  范围: [0, +∞)，越小越好，单位与目标变量一致(球)")
        print("CV_R2 (交叉验证R²): 5折时序交叉验证的平均R²，衡量泛化能力")
        print("CV_MAE (交叉验证MAE): 5折时序交叉验证的平均MAE，衡量实际预测误差")
        print("CV_R2_std: 交叉验证R²的标准差，越小说明模型越稳定")
        print("-" * 60)

        self._select_best_model()

    def _select_best_model(self):

        models_df = pd.DataFrame(self.models_info)

        models_df["综合评分"] = models_df["CV_R2"] - models_df["CV_MAE"] / 100 - models_df["CV_R2_std"] * 0.3

        self.best_model_name = models_df.loc[models_df["综合评分"].idxmax(), "模型"]
        self.best_model = self.models[self.best_model_name]
        self.best_model_metrics = models_df.loc[models_df["模型"] == self.best_model_name].iloc[0].to_dict()

        print(f"\n✓ 最优模型：{self.best_model_name}")
        print(f"  CV_R2: {self.best_model_metrics['CV_R2']:.4f}")
        print(f"  CV_MAE: {self.best_model_metrics['CV_MAE']:.2f}")

        self._interpret_model_metrics()

    def _interpret_model_metrics(self):

        print("\n" + "-" * 60)
        print("模型评估解读")
        print("-" * 60)

        r2 = self.best_model_metrics['CV_R2']
        mae = self.best_model_metrics['CV_MAE']

        if r2 >= 0.8:
            r2_level = "极好"
            r2_desc = "模型能够解释80%以上的方差，拟合效果非常优秀"
        elif r2 >= 0.6:
            r2_level = "良好"
            r2_desc = "模型能够解释60%-80%的方差，拟合效果较好"
        elif r2 >= 0.4:
            r2_level = "一般"
            r2_desc = "模型能够解释40%-60%的方差，拟合效果一般"
        elif r2 >= 0.2:
            r2_level = "较弱"
            r2_desc = "模型能够解释20%-40%的方差，拟合效果较弱"
        else:
            r2_level = "很差"
            r2_desc = "模型解释能力不足20%，拟合效果很差"

        print(f"✓ R²解读: {r2_level} ({r2:.4f})")
        print(f"  {r2_desc}")
        print(f"✓ MAE解读: 平均预测误差约 {mae:.1f} 个进球")

        train_r2 = self.best_model_metrics['R2']
        if train_r2 - r2 > 0.5:
            print("⚠️ 过拟合警告: 训练集R²与交叉验证R²差距较大，模型可能存在过拟合")
        else:
            print("✓ 模型泛化能力: 训练集与交叉验证表现接近，泛化能力较好")

    def predict_2018(self):

        print("\n" + "=" * 60)
        print("预测2018年俄罗斯世界杯")
        print("=" * 60)

        latest_year = self.processed_df["年份"].max()
        latest_data = self.processed_df[self.processed_df["年份"] == latest_year].iloc[0]

        year_mean = self.processed_df["年份"].mean()
        year_std = self.processed_df["年份"].std()

        prediction_data = pd.DataFrame({
            "参赛队伍数量": [32],
            "总比赛场次": [64],
            "总观众人数": [3031768],
            "年份_归一化": [(2018 - year_mean) / year_std],
            "前一届进球数": [latest_data["总进球数"]],
            "前两届进球数": [self.processed_df[self.processed_df["年份"] == 2010]["总进球数"].values[0]],
            "前一届场均进球": [latest_data["场均进球"]],
            "队伍场次比": [32 / 64]
        })

        prediction_data = prediction_data[self.feature_columns]

        prediction = self.best_model.predict(prediction_data)[0]

        actual = 169

        absolute_error = abs(prediction - actual)
        relative_error = absolute_error / actual * 100

        print(f"预测值: {prediction:.1f}")
        print(f"实际值: {actual}")
        print(f"绝对误差: {absolute_error:.1f}")
        print(f"相对误差: {relative_error:.2f}%")

        self.prediction_result = {
            "prediction": round(prediction),
            "actual": actual,
            "absolute_error": absolute_error,
            "relative_error": relative_error
        }

        return self.prediction_result

    def generate_report(self):

        report_lines = []

        report_lines.append("=" * 60)
        report_lines.append("Task1 回归预测模型报告")
        report_lines.append("=" * 60)
        report_lines.append("")

        report_lines.append("一、模型配置")
        report_lines.append("-" * 40)
        report_lines.append(f"目标变量: {self.target_column}")
        report_lines.append(f"特征数量: {len(self.feature_columns)}")
        report_lines.append(f"特征列表: {', '.join(self.feature_columns)}")
        report_lines.append(f"样本数量: {len(self.processed_df)}")
        report_lines.append("")

        report_lines.append("二、模型对比")
        report_lines.append("-" * 40)
        models_df = pd.DataFrame(self.models_info)
        report_lines.append(models_df.to_string(index=False))
        report_lines.append("")

        report_lines.append("三、最优模型")
        report_lines.append("-" * 40)
        report_lines.append(f"模型名称: {self.best_model_name}")
        for key, value in self.best_model_metrics.items():
            report_lines.append(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
        report_lines.append("")

        report_lines.append("四、2018年世界杯预测")
        report_lines.append("-" * 40)
        if self.prediction_result:
            report_lines.append(f"预测总进球数: {self.prediction_result['prediction']}")
            report_lines.append(f"实际总进球数: {self.prediction_result['actual']}")
            report_lines.append(f"绝对误差: {self.prediction_result['absolute_error']:.1f}")
            report_lines.append(f"相对误差: {self.prediction_result['relative_error']:.2f}%")
        report_lines.append("")

        report_lines.append("=" * 60)

        report_path = REPORT_DIR / "regression_model_log.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

        print(f"\n✓ 报告已保存到: {report_path}")

    def run(self):

        self.clean_and_preprocess()

        self.train_models()

        self.predict_2018()

        self.generate_report()

        return self.best_model