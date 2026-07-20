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
from config import REPORT_DIR, MODEL_DIR
import joblib


class MatchPredictionModel:
    """
    比赛进球预测模型
    """

    def __init__(self, match_df, worldcups_df=None):

        self.match_df = match_df
        self.worldcups_df = worldcups_df
        self.target_column = "总进球数"
        self.features = [
            "主队进球", "客队进球", "半场总进球", "观众人数",
            "年份", "主队胜场", "客队胜场", "平局场数"
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

        df["进球趋势"] = df.groupby("年份")["总进球数"].transform("mean")

        df["阶段编码"] = df["阶段"].astype("category").cat.codes

        df["主队优势"] = df.groupby("主队名称")["主队进球"].transform("mean")
        df["客队防守"] = df.groupby("客队名称")["客队进球"].transform("mean")

        df["进球累计"] = df.groupby("年份")["总进球数"].cumsum()

        df["比赛日类型"] = df["比赛日"].apply(lambda x: "周末" if x in [6, 7] else "工作日")
        df["比赛日类型编码"] = df["比赛日类型"].astype("category").cat.codes

        feature_cols = [
            "主队进球", "客队进球", "半场总进球", "观众人数",
            "阶段编码", "主队优势", "客队防守", "进球累计",
            "比赛日类型编码", "年份"
        ]

        feature_cols = [col for col in feature_cols if col in df.columns]

        self.feature_cols = feature_cols
        self.X = df[feature_cols]
        self.y = df[self.target_column]

        print(f"特征数量：{len(feature_cols)}")
        print(f"特征列表：{feature_cols}")
        print(f"样本数量：{len(df)}")

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
    """
    年度总进球预测器
    使用 WorldCups_preview.csv 数据
    """

    def __init__(self, worldcups_df):

        self.df = worldcups_df
        self.target_column = "总进球数"
        self.feature_columns = ["参赛队伍数量", "总比赛场次", "总观众人数"]

        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_r2 = -np.inf

    def prepare_features(self):

        print("\n" + "=" * 60)
        print("准备年度预测特征")
        print("=" * 60)

        df = self.df.copy()

        df["年份趋势"] = df["年份"] - df["年份"].min()

        df["场均进球"] = df["总进球数"] / df["总比赛场次"]

        df["每队比赛场次"] = df["总比赛场次"] / df["参赛队伍数量"]

        df["观众场均"] = df["总观众人数"] / df["总比赛场次"]

        df["历史平均进球"] = df["总进球数"].rolling(window=3, min_periods=1).mean().shift(1)
        df["历史平均进球"].fillna(df["总进球数"].mean(), inplace=True)

        df["历史参赛队伍"] = df["参赛队伍数量"].shift(1)
        df["历史参赛队伍"].fillna(df["参赛队伍数量"].mean(), inplace=True)

        df["历史比赛场次"] = df["总比赛场次"].shift(1)
        df["历史比赛场次"].fillna(df["总比赛场次"].mean(), inplace=True)

        self.feature_cols = [
            "参赛队伍数量", "总比赛场次", "总观众人数",
            "年份趋势", "每队比赛场次", "观众场均",
            "历史平均进球", "历史参赛队伍", "历史比赛场次"
        ]

        self.X = df[self.feature_cols]
        self.y = df[self.target_column]

        print(f"特征数量：{len(self.feature_cols)}")
        print(f"特征列表：{self.feature_cols}")
        print(f"样本数量：{len(df)}")

        return df

    def train_models(self):

        print("\n" + "=" * 60)
        print("训练年度总进球预测模型")
        print("=" * 60)

        tscv = TimeSeriesSplit(n_splits=5)

        model_list = [
            ("LinearRegression", LinearRegression()),
            ("Ridge", Ridge(alpha=1.0)),
            ("Lasso", Lasso(alpha=0.1)),
            ("RandomForest", RandomForestRegressor(
                n_estimators=100, max_depth=7, random_state=42)),
            ("GradientBoosting", GradientBoostingRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42))
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

    def predict_2018(self):

        print("\n" + "=" * 60)
        print("预测2018年俄罗斯世界杯总进球")
        print("=" * 60)

        if self.worldcups_df is None:
            raise ValueError("需要WorldCups_preview.csv数据")

        df_2018 = self.worldcups_df[self.worldcups_df["年份"] == 2018]

        if df_2018.empty:
            raise ValueError("没有找到2018年数据")

        actual_goals = df_2018["总进球数"].values[0]

        X_2018 = df_2018[self.feature_cols]

        predictions = {}
        for name, model in self.models.items():
            pred = model.predict(X_2018)[0]
            predictions[name] = pred
            print(f"{name}: {pred:.2f}")

        best_pred = self.best_model.predict(X_2018)[0]

        absolute_error = abs(best_pred - actual_goals)
        relative_error = absolute_error / actual_goals * 100

        print(f"\n实际进球数: {actual_goals}")
        print(f"最佳模型预测: {best_pred:.2f}")
        print(f"绝对误差: {absolute_error:.2f}")
        print(f"相对误差: {relative_error:.2f}%")

        self.prediction_result = {
            "实际进球": actual_goals,
            "预测进球": best_pred,
            "绝对误差": absolute_error,
            "相对误差": relative_error,
            "最佳模型": self.best_model_name,
            "所有预测": predictions
        }

        return self.prediction_result

    def generate_report(self):

        print("\n" + "=" * 60)
        print("生成年度预测报告")
        print("=" * 60)

        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("Task2 年度总进球预测报告")
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

        if hasattr(self, "prediction_result"):
            report_lines.append("四、2018年预测结果")
            report_lines.append("-" * 30)
            report_lines.append(f"实际进球数: {self.prediction_result['实际进球']}")
            report_lines.append(f"预测进球数: {self.prediction_result['预测进球']:.2f}")
            report_lines.append(f"绝对误差: {self.prediction_result['绝对误差']:.2f}")
            report_lines.append(f"相对误差: {self.prediction_result['相对误差']:.2f}%")
            report_lines.append("")

            report_lines.append("各模型预测结果:")
            for name, pred in self.prediction_result["所有预测"].items():
                report_lines.append(f"  {name}: {pred:.2f}")

        report_lines.append("\n" + "=" * 70)
        report_lines.append("报告生成完毕")
        report_lines.append("=" * 70)

        self.report = "\n".join(report_lines)

        report_path = REPORT_DIR / "task2_yearly_prediction_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(self.report)

        print(f"\n✓ 年度预测报告已保存到: {report_path}")

        return report_path

    def save_models(self):

        for name, model in self.models.items():
            model_path = MODEL_DIR / f"task2_yearly_{name.lower().replace(' ', '_')}_model.pkl"
            joblib.dump(model, model_path)
            print(f"✓ 年度模型 {name} 已保存到: {model_path}")

        return True

    def run(self):

        self.prepare_features()
        self.train_models()
        self.predict_2018()
        self.generate_report()
        self.save_models()

        return self.best_model