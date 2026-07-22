"""
prediction.py
==========================
Task2 预测模型模块
==========================
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.config import REPORT_DIR, MODEL_DIR
import joblib


class MatchPredictionModel:
    """
    比赛进球预测模型
    """

    def __init__(self, match_df, worldcups_df=None):

        self.match_df = match_df
        self.worldcups_df = worldcups_df
        self.target_column = "总进球数"
        # 仅使用赛前可获得特征，避免泄露比赛结果信息
        self.features = [
            "年份", "阶段", "观众人数", "主队胜场", "客队胜场", "平局场数"
        ]

        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_r2 = -np.inf

    def prepare_features(self):

        print("\n" + "=" * 60)
        print("准备预测特征")
        print("=" * 60)

        df = self.match_df.copy()

        df["年份"] = pd.to_numeric(df["年份"], errors="coerce")
        df["观众人数"] = pd.to_numeric(df["观众人数"], errors="coerce")
        df["主队胜场"] = pd.to_numeric(df["主队胜场"], errors="coerce")
        df["客队胜场"] = pd.to_numeric(df["客队胜场"], errors="coerce")
        df["平局场数"] = pd.to_numeric(df["平局场数"], errors="coerce")

        # 阶段编码只作为类别标识，不引入未来统计量
        df["阶段编码"] = df["阶段"].astype("category").cat.codes

        feature_cols = [
            "年份", "阶段编码", "观众人数", "主队胜场", "客队胜场", "平局场数"
        ]

        feature_cols = [col for col in feature_cols if col in df.columns]

        self.feature_cols = feature_cols
        self.X = df[feature_cols]
        self.y = pd.to_numeric(df[self.target_column], errors="coerce")

        valid_mask = self.X.notna().all(axis=1) & self.y.notna()
        self.X = self.X.loc[valid_mask].reset_index(drop=True)
        self.y = self.y.loc[valid_mask].reset_index(drop=True)

        if "年份" in self.X.columns:
            order = self.X["年份"].sort_values().index
            self.X = self.X.loc[order].reset_index(drop=True)
            self.y = self.y.loc[order].reset_index(drop=True)

        print(f"特征数量：{len(feature_cols)}")
        print(f"特征列表：{feature_cols}")
        print(f"样本数量：{len(self.X)}")

        return df

    def train_models(self):

        print("\n" + "=" * 60)
        print("训练回归模型")
        print("=" * 60)

        tscv = TimeSeriesSplit(n_splits=5)

        model_list = [
            ("LinearRegression", LinearRegression()),
            ("Ridge", Ridge(alpha=1.0)),
            ("Lasso", Lasso(alpha=0.1)),
            ("ElasticNet", ElasticNet(alpha=0.1, l1_ratio=0.5)),
            ("RandomForest", RandomForestRegressor(
                n_estimators=100, max_depth=7, random_state=42)),
            ("GradientBoosting", GradientBoostingRegressor(
                n_estimators=100, max_depth=5, random_state=42)),
            ("SVR", SVR(kernel="rbf", C=100, gamma=0.1)),
            ("KNeighbors", KNeighborsRegressor(n_neighbors=5)),
            ("DecisionTree", DecisionTreeRegressor(max_depth=5, random_state=42))
        ]

        results = []

        for name, model in model_list:
            print(f"\n训练 {name}...")

            r2_scores = cross_val_score(model, self.X, self.y, cv=tscv, scoring="r2")
            mae_scores = cross_val_score(model, self.X, self.y, cv=tscv, scoring="neg_mean_absolute_error")
            rmse_scores = cross_val_score(model, self.X, self.y, cv=tscv, scoring="neg_root_mean_squared_error")

            model.fit(self.X, self.y)
            y_pred = model.predict(self.X)

            result = {
                "模型名称": name,
                "R2均值": np.mean(r2_scores),
                "R2标准差": np.std(r2_scores),
                "MAE均值": -np.mean(mae_scores),
                "RMSE均值": -np.mean(rmse_scores),
                "训练R2": r2_score(self.y, y_pred)
            }

            results.append(result)

            self.models[name] = model

            if result["R2均值"] > self.best_r2:
                self.best_r2 = result["R2均值"]
                self.best_model = model
                self.best_model_name = name

            print(f"  R2: {result['R2均值']:.4f} (±{result['R2标准差']:.4f})")
            print(f"  MAE: {result['MAE均值']:.4f}")
            print(f"  RMSE: {result['RMSE均值']:.4f}")

        self.results_df = pd.DataFrame(results)
        self.results_df.sort_values("R2均值", ascending=False, inplace=True)

        print("\n" + "=" * 60)
        print("模型对比结果：")
        print("=" * 60)
        print(self.results_df.to_string(index=False))

        print(f"\n✓ 最佳模型: {self.best_model_name} (R2={self.best_r2:.4f})")

        return self.results_df

    def train_ensemble(self):

        print("\n" + "=" * 60)
        print("训练集成模型")
        print("=" * 60)

        base_models = [
            ("rf", RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42)),
            ("gb", GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)),
            ("lr", LinearRegression())
        ]

        ensemble = VotingRegressor(estimators=base_models)

        tscv = TimeSeriesSplit(n_splits=5)
        r2_scores = cross_val_score(ensemble, self.X, self.y, cv=tscv, scoring="r2")
        mae_scores = cross_val_score(ensemble, self.X, self.y, cv=tscv, scoring="neg_mean_absolute_error")
        rmse_scores = cross_val_score(ensemble, self.X, self.y, cv=tscv, scoring="neg_root_mean_squared_error")

        ensemble.fit(self.X, self.y)
        y_pred = ensemble.predict(self.X)

        result = {
            "模型名称": "VotingEnsemble",
            "R2均值": np.mean(r2_scores),
            "R2标准差": np.std(r2_scores),
            "MAE均值": -np.mean(mae_scores),
            "RMSE均值": -np.mean(rmse_scores),
            "训练R2": r2_score(self.y, y_pred)
        }

        self.models["VotingEnsemble"] = ensemble

        if result["R2均值"] > self.best_r2:
            self.best_r2 = result["R2均值"]
            self.best_model = ensemble
            self.best_model_name = "VotingEnsemble"

        print(f"\n集成模型结果：")
        print(f"  R2: {result['R2均值']:.4f} (±{result['R2标准差']:.4f})")
        print(f"  MAE: {result['MAE均值']:.4f}")
        print(f"  RMSE: {result['RMSE均值']:.4f}")

        self.results_df = pd.concat([self.results_df, pd.DataFrame([result])], ignore_index=True)
        self.results_df.sort_values("R2均值", ascending=False, inplace=True)

        print(f"\n✓ 最佳模型更新为: {self.best_model_name} (R2={self.best_r2:.4f})")

        return result

    def analyze_feature_importance(self):

        print("\n" + "=" * 60)
        print("特征重要性分析")
        print("=" * 60)

        importances = []

        for name, model in self.models.items():
            if hasattr(model, "feature_importances_"):
                imp = pd.DataFrame({
                    "特征": self.feature_cols,
                    "重要性": model.feature_importances_,
                    "模型": name
                })
                importances.append(imp)

        if importances:
            self.feature_importance_df = pd.concat(importances, ignore_index=True)

            avg_importance = self.feature_importance_df.groupby("特征")["重要性"].mean().sort_values(ascending=False)

            print("\n平均特征重要性：")
            print("-" * 40)
            for feature, importance in avg_importance.items():
                print(f"{feature}: {importance:.4f}")

            return avg_importance
        else:
            print("没有支持特征重要性的模型")
            return None

    def generate_statistics_report(self):

        print("\n" + "=" * 60)
        print("生成统计报告")
        print("=" * 60)

        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("Task2 预测模型统计报告")
        report_lines.append("=" * 70)
        report_lines.append("")

        report_lines.append("一、数据概况")
        report_lines.append("-" * 30)
        report_lines.append(f"样本数量: {len(self.X)}")
        report_lines.append(f"特征数量: {len(self.feature_cols)}")
        report_lines.append(f"特征列表: {', '.join(self.feature_cols)}")
        report_lines.append("")

        report_lines.append("二、模型训练结果")
        report_lines.append("-" * 30)
        report_lines.append(self.results_df.to_string(index=False))
        report_lines.append("")

        report_lines.append(f"三、最佳模型: {self.best_model_name}")
        report_lines.append("-" * 30)

        if hasattr(self.best_model, "coef_"):
            report_lines.append("模型系数:")
            for feature, coef in zip(self.feature_cols, self.best_model.coef_):
                report_lines.append(f"  {feature}: {coef:.4f}")
            if hasattr(self.best_model, "intercept_"):
                report_lines.append(f"截距: {self.best_model.intercept_:.4f}")

        if hasattr(self.best_model, "feature_importances_"):
            report_lines.append("\n特征重要性:")
            for feature, importance in sorted(zip(self.feature_cols, self.best_model.feature_importances_),
                                              key=lambda x: -x[1]):
                report_lines.append(f"  {feature}: {importance:.4f}")

        report_lines.append(f"\n最佳R2得分: {self.best_r2:.4f}")
        report_lines.append("")

        report_lines.append("四、统计检验")
        report_lines.append("-" * 30)

        X_sm = sm.add_constant(self.X)
        model_sm = sm.OLS(self.y, X_sm).fit()
        report_lines.append("\nOLS回归结果:")
        report_lines.append(str(model_sm.summary()))

        report_lines.append("\n" + "=" * 70)
        report_lines.append("报告生成完毕")
        report_lines.append("=" * 70)

        self.report = "\n".join(report_lines)

        report_path = REPORT_DIR / "task2_prediction_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(self.report)

        print(f"\n✓ 预测报告已保存到: {report_path}")

        return report_path

    def save_models(self):

        for name, model in self.models.items():
            model_path = MODEL_DIR / f"task2_{name.lower().replace(' ', '_')}_model.pkl"
            joblib.dump(model, model_path)
            print(f"✓ 模型 {name} 已保存到: {model_path}")

        return True

    def predict(self, X_new):

        return self.best_model.predict(X_new)

    def run(self):

        self.prepare_features()
        self.train_models()
        self.train_ensemble()
        self.analyze_feature_importance()
        self.generate_statistics_report()
        self.save_models()

        return self.best_model


class YearlyTotalGoalsPredictor:
    pass
