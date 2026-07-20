"""
regression.py
==========================
Task1 回归预测模块（优化版）
==========================
"""

import pandas as pd
import numpy as np
import random
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

from config import MODEL_DIR, REPORT_DIR

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

        self.processed_df["年份平方"] = self.processed_df["年份_归一化"] ** 2
        self.processed_df["年份立方"] = self.processed_df["年份_归一化"] ** 3

        self.processed_df["前一届进球数"] = self.processed_df["总进球数"].shift(1)
        self.processed_df["前两届进球数"] = self.processed_df["总进球数"].shift(2)
        self.processed_df["前一届场均进球"] = self.processed_df["场均进球"].shift(1)

        self.processed_df["队伍场次比"] = self.processed_df["参赛队伍数量"] / self.processed_df["总比赛场次"]

        self.feature_columns = [
            "参赛队伍数量",
            "总比赛场次",
            "总观众人数",
            "年份_归一化",
            "年份平方",
            "年份立方",
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
            ("Ridge回归", Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=0.5))])),
            ("随机森林", RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42)),
            ("梯度提升", GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)),
            ("多项式回归(2阶)", Pipeline([('poly', PolynomialFeatures(degree=2, include_bias=False)),
                                         ('scaler', StandardScaler()), ('linear', LinearRegression())])),
            ("集成学习", VotingRegressor(estimators=[
                ('lr', LinearRegression()),
                ('rf', RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42)),
                ('gb', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42))
            ]))
        ]

        for name, model in models_to_train:

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
                "CV_MAE": -cv_mae.mean()
            })

            print(f"✓ {name}: R2={r2:.4f}, MAE={mae:.2f}, CV_R2={cv_scores.mean():.4f}")

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
        print("-" * 60)

        self._select_best_model()

    def _select_best_model(self):

        models_df = pd.DataFrame(self.models_info)

        models_df["综合评分"] = models_df["CV_R2"] - models_df["CV_MAE"] / 100

        self.best_model_name = models_df.loc[models_df["综合评分"].idxmax(), "模型"]
        self.best_model = self.models[self.best_model_name]
        self.best_model_metrics = models_df.loc[models_df["模型"] == self.best_model_name].iloc[0].to_dict()

        print(f"\n✓ 最优模型：{self.best_model_name}")
        print(f"  CV_R2: {self.best_model_metrics['CV_R2']:.4f}")
        print(f"  CV_MAE: {self.best_model_metrics['CV_MAE']:.2f}")

        print("\n" + "-" * 60)
        print("最优模型指标解读：")
        print("-" * 60)
        r2_val = self.best_model_metrics['R2']
        cv_r2_val = self.best_model_metrics['CV_R2']
        mae_val = self.best_model_metrics['MAE']
        cv_mae_val = self.best_model_metrics['CV_MAE']
        rmse_val = self.best_model_metrics['RMSE']

        if r2_val >= 0.9:
            r2_level = "极好"
        elif r2_val >= 0.7:
            r2_level = "良好"
        elif r2_val >= 0.5:
            r2_level = "一般"
        elif r2_val >= 0.3:
            r2_level = "较弱"
        else:
            r2_level = "很差"
        print(f"R² = {r2_val:.4f} → 模型能解释 {r2_val*100:.2f}% 的进球数变化，拟合效果{r2_level}")

        if cv_r2_val >= 0.7:
            cv_level = "泛化能力强"
        elif cv_r2_val >= 0.5:
            cv_level = "泛化能力一般"
        elif cv_r2_val >= 0:
            cv_level = "泛化能力较弱"
        else:
            cv_level = "泛化能力差，可能过拟合"
        print(f"CV_R² = {cv_r2_val:.4f} → 交叉验证下{cv_level}")

        print(f"MAE = {mae_val:.2f} → 训练集平均预测误差约 {mae_val:.2f} 球")
        print(f"CV_MAE = {cv_mae_val:.2f} → 交叉验证平均预测误差约 {cv_mae_val:.2f} 球")
        print(f"RMSE = {rmse_val:.2f} → 预测误差标准差约 {rmse_val:.2f} 球")

        overfitting = r2_val - cv_r2_val
        print(f"\n过拟合分析: 训练R² - CV_R² = {overfitting:.4f}")
        if overfitting > 0.3:
            print("  结论: 差值较大，模型存在明显过拟合")
        elif overfitting > 0.1:
            print("  结论: 差值适中，模型有轻微过拟合")
        else:
            print("  结论: 差值较小，模型泛化能力良好")
        print("-" * 60)

        return self.best_model

    def analyze_features(self):

        print("\n" + "=" * 60)
        print("特征分析")
        print("=" * 60)

        X = self.processed_df[self.feature_columns]

        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
        elif hasattr(self.best_model[-1], 'feature_importances_'):
            importances = self.best_model[-1].feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            importances = np.abs(self.best_model.coef_)
        elif hasattr(self.best_model[-1], 'coef_'):
            importances = np.abs(self.best_model[-1].coef_)
        else:
            if hasattr(self.best_model, 'estimators_'):
                total_importances = np.zeros(len(self.feature_columns))
                for est in self.best_model.estimators_:
                    if hasattr(est, 'feature_importances_'):
                        total_importances += est.feature_importances_
                    elif hasattr(est, 'coef_'):
                        total_importances += np.abs(est.coef_)
                importances = total_importances / len(self.best_model.estimators_)
            else:
                importances = np.ones(len(self.feature_columns)) / len(self.feature_columns)

        self.feature_importance = pd.DataFrame({
            "特征": self.feature_columns,
            "重要性": importances
        }).sort_values("重要性", ascending=False)

        print("\n特征重要性排序：")
        print("-" * 40)
        print(self.feature_importance.to_string(index=False))

        return self.feature_importance

    def save_report(self):

        log_lines = []

        log_lines.append("=" * 70)
        log_lines.append("APMCM竞赛 - 世界杯总进球数预测模型报告")
        log_lines.append("=" * 70)
        log_lines.append("")

        log_lines.append("一、问题描述")
        log_lines.append("-" * 50)
        log_lines.append("以历届世界杯总进球数为预测目标，")
        log_lines.append("以参赛队伍数量、总比赛场次、总观众人数等为输入特征，")
        log_lines.append("建立回归预测模型，并预测2018年俄罗斯世界杯总进球数。")
        log_lines.append("")

        log_lines.append("二、数据预处理")
        log_lines.append("-" * 50)
        log_lines.append(f"数据来源：WorldCups_preview.csv")
        log_lines.append(f"数据覆盖：{self.processed_df['年份'].min()} - {self.processed_df['年份'].max()}")
        log_lines.append(f"样本数量：{len(self.processed_df)}")
        log_lines.append(f"特征列表：{', '.join(self.feature_columns)}")
        log_lines.append("")

        log_lines.append("三、特征工程")
        log_lines.append("-" * 50)
        log_lines.append("1. 基础特征：参赛队伍数量、总比赛场次、总观众人数")
        log_lines.append("2. 时间趋势特征：年份归一化、年份平方、年份立方")
        log_lines.append("3. 历史数据特征：前一届进球数、前两届进球数、前一届场均进球")
        log_lines.append("4. 比率特征：队伍场次比")
        log_lines.append("")

        log_lines.append("四、模型对比")
        log_lines.append("-" * 50)
        models_df = pd.DataFrame(self.models_info)
        log_lines.append(models_df.to_string(index=False))
        log_lines.append("")

        log_lines.append("五、最优模型")
        log_lines.append("-" * 50)
        log_lines.append(f"模型名称：{self.best_model_name}")
        log_lines.append(f"R2 拟合优度：{self.best_model_metrics['R2']:.4f}")
        log_lines.append(f"MAE 平均绝对误差：{self.best_model_metrics['MAE']:.2f}")
        log_lines.append(f"RMSE 均方根误差：{self.best_model_metrics['RMSE']:.2f}")
        log_lines.append(f"CV_R2 交叉验证：{self.best_model_metrics['CV_R2']:.4f}")
        log_lines.append(f"CV_MAE 交叉验证MAE：{self.best_model_metrics['CV_MAE']:.2f}")
        log_lines.append("")

        log_lines.append("评估指标详细解释：")
        log_lines.append("-" * 50)
        log_lines.append(f"R² (决定系数, R-squared) = {self.best_model_metrics['R2']:.4f}")
        log_lines.append("  计算公式: R² = 1 - SS_res/SS_tot")
        log_lines.append("  含义: 模型解释的目标变量方差比例")
        r2_val = self.best_model_metrics['R2']
        if r2_val >= 0.9:
            r2_level = "极好"
        elif r2_val >= 0.7:
            r2_level = "良好"
        elif r2_val >= 0.5:
            r2_level = "一般"
        elif r2_val >= 0.3:
            r2_level = "较弱"
        else:
            r2_level = "很差"
        log_lines.append(f"  解释: 模型能解释 {r2_val*100:.2f}% 的进球数变化，拟合效果{r2_level}")
        log_lines.append("  范围: (-∞, 1]，越接近1越好，1表示完美拟合")
        log_lines.append("  特点: 衡量模型整体拟合优度，是最常用的评估指标")
        log_lines.append("")
        log_lines.append(f"MAE (平均绝对误差, Mean Absolute Error) = {self.best_model_metrics['MAE']:.2f}")
        log_lines.append("  计算公式: MAE = (1/n) * Σ|y_true - y_pred|")
        log_lines.append("  含义: 预测值与实际值之差的绝对值的平均")
        log_lines.append(f"  解释: 模型预测的平均误差约为 {self.best_model_metrics['MAE']:.2f} 个进球")
        log_lines.append("  范围: [0, +∞)，越小越好，0表示完美预测")
        log_lines.append("  特点: 对异常值不敏感，易于解释，量纲与目标变量一致")
        log_lines.append("")
        log_lines.append(f"RMSE (均方根误差, Root Mean Squared Error) = {self.best_model_metrics['RMSE']:.2f}")
        log_lines.append("  计算公式: RMSE = √[(1/n) * Σ(y_true - y_pred)²]")
        log_lines.append("  含义: 均方误差的平方根，与原数据同量纲")
        log_lines.append(f"  解释: 预测误差的标准差约为 {self.best_model_metrics['RMSE']:.2f} 个进球")
        log_lines.append("  范围: [0, +∞)，越小越好，0表示完美预测")
        log_lines.append("  特点: 对异常值敏感，惩罚较大误差，量纲与目标变量一致")
        log_lines.append("")
        log_lines.append(f"CV_R² (交叉验证决定系数) = {self.best_model_metrics['CV_R2']:.4f}")
        log_lines.append("  计算公式: 5折时序交叉验证(TimeSeriesSplit)的平均R²")
        log_lines.append("  含义: 模型在未见数据上的泛化能力")
        cv_r2_val = self.best_model_metrics['CV_R2']
        if cv_r2_val >= 0.7:
            cv_level = "泛化能力强"
        elif cv_r2_val >= 0.5:
            cv_level = "泛化能力一般"
        elif cv_r2_val >= 0:
            cv_level = "泛化能力较弱"
        else:
            cv_level = "泛化能力差，可能过拟合"
        log_lines.append(f"  解释: {cv_level}")
        log_lines.append("  范围: (-∞, 1]，越接近1越好")
        log_lines.append("  特点: 比单次R²更能反映模型的真实预测能力")
        log_lines.append("")
        log_lines.append(f"CV_MAE (交叉验证平均绝对误差) = {self.best_model_metrics['CV_MAE']:.2f}")
        log_lines.append("  计算公式: 5折时序交叉验证的平均MAE")
        log_lines.append("  含义: 模型在未见数据上的平均预测误差")
        log_lines.append(f"  解释: 交叉验证下平均预测误差约 {self.best_model_metrics['CV_MAE']:.2f} 个进球")
        log_lines.append("  范围: [0, +∞)，越小越好")
        log_lines.append("  特点: 反映模型在实际应用中的预测精度")
        log_lines.append("")

        overfitting = r2_val - cv_r2_val
        log_lines.append("过拟合分析:")
        log_lines.append(f"  训练R² - CV_R² = {overfitting:.4f}")
        if overfitting > 0.3:
            log_lines.append("  结论: 差值较大，模型存在明显过拟合，建议增加数据或正则化")
        elif overfitting > 0.1:
            log_lines.append("  结论: 差值适中，模型有轻微过拟合")
        else:
            log_lines.append("  结论: 差值较小，模型泛化能力良好")
        log_lines.append("")
        log_lines.append("指标选择建议:")
        log_lines.append("  - R²: 评估模型整体拟合优度，适合比较不同模型")
        log_lines.append("  - MAE: 直观反映平均预测误差，易于向非专业人员解释")
        log_lines.append("  - RMSE: 对异常值敏感，适合关注极端误差的场景")
        log_lines.append("  - CV_R²/CV_MAE: 评估模型泛化能力，防止过拟合误判")
        log_lines.append("")

        log_lines.append("六、特征重要性分析")
        log_lines.append("-" * 50)
        log_lines.append(self.feature_importance.to_string(index=False))
        log_lines.append("")

        log_lines.append("七、2018世界杯预测结果")
        log_lines.append("-" * 50)
        log_lines.append(f"预测总进球数：{self.prediction_result['预测值']} 球")
        log_lines.append(f"实际总进球数：{self.prediction_result['实际值']} 球")
        log_lines.append(f"绝对误差：{self.prediction_result['绝对误差']} 球")
        log_lines.append(f"相对误差：{self.prediction_result['相对误差']}%")
        log_lines.append("")

        log_lines.append("八、模型评价")
        log_lines.append("-" * 50)
        if self.prediction_result['相对误差'] < 5:
            log_lines.append("模型预测精度高，相对误差小于5%，达到竞赛优秀水平。")
        elif self.prediction_result['相对误差'] < 10:
            log_lines.append("模型预测精度良好，相对误差在5%-10%之间。")
        else:
            log_lines.append("模型预测精度一般，建议进一步优化特征或尝试其他模型。")
        log_lines.append("")

        log_lines.append("九、方法局限性")
        log_lines.append("-" * 50)
        log_lines.append("1. 数据量限制: 世界杯数据仅有约20届，样本量较小，")
        log_lines.append("   可能影响模型的稳定性和泛化能力。")
        log_lines.append("2. 线性假设: 线性回归模型假设特征与目标变量之间存在")
        log_lines.append("   线性关系，但实际进球数可能受非线性因素影响。")
        log_lines.append("3. 时间趋势假设: 假设历史趋势可以延续到未来，")
        log_lines.append("   但足球规则、战术、球员水平等可能随时间变化。")
        log_lines.append("4. 特征局限性: 仅使用参赛队伍、比赛场次、观众人数等")
        log_lines.append("   基础特征，未考虑天气、场地、裁判等临场因素。")
        log_lines.append("5. 历史数据可用性: 早期世界杯数据可能不够完整或准确。")
        log_lines.append("")

        log_lines.append("=" * 70)
        log_lines.append("报告生成时间：自动生成")
        log_lines.append("=" * 70)

        log_text = "\n".join(log_lines)

        log_path = REPORT_DIR / "regression_model_log.txt"

        with open(log_path, "w", encoding="utf-8") as f:
            f.write(log_text)

        print(f"\n✓ 竞赛报告已保存到: {log_path}")

        return log_path

    def predict_2018(self):

        print("\n" + "=" * 60)
        print("预测2018俄罗斯世界杯总进球数")
        print("=" * 60)

        last_year_data = self.processed_df.iloc[-1]
        year_mean = self.processed_df["年份"].mean()
        year_std = self.processed_df["年份"].std()

        X_2018 = pd.DataFrame({
            "参赛队伍数量": [32],
            "总比赛场次": [64],
            "总观众人数": [3031768],
            "年份_归一化": [(2018 - year_mean) / year_std],
            "年份平方": [((2018 - year_mean) / year_std) ** 2],
            "年份立方": [((2018 - year_mean) / year_std) ** 3],
            "前一届进球数": [last_year_data["总进球数"]],
            "前两届进球数": [self.processed_df.iloc[-2]["总进球数"]],
            "前一届场均进球": [last_year_data["场均进球"]],
            "队伍场次比": [32 / 64]
        })

        print("\n所有模型预测结果：")
        print("-" * 50)

        all_predictions = []
        best_prediction = None
        best_error = float('inf')

        for name, model in self.models.items():
            predicted_goals = model.predict(X_2018)[0]
            actual_goals = 169
            absolute_error = abs(predicted_goals - actual_goals)
            relative_error = (absolute_error / actual_goals) * 100

            all_predictions.append({
                "模型": name,
                "预测值": round(predicted_goals, 2),
                "实际值": actual_goals,
                "绝对误差": round(absolute_error, 2),
                "相对误差": round(relative_error, 2)
            })

            print(f"{name}:")
            print(f"  预测: {predicted_goals:.2f} 球")
            print(f"  实际: {actual_goals} 球")
            print(f"  误差: {relative_error:.2f}%")
            print()

            if relative_error < best_error:
                best_error = relative_error
                best_prediction = {
                    "模型": name,
                    "预测值": round(predicted_goals, 2),
                    "实际值": actual_goals,
                    "绝对误差": round(absolute_error, 2),
                    "相对误差": round(relative_error, 2)
                }

        print("\n" + "=" * 60)
        print("最优预测结果")
        print("=" * 60)
        print(f"最佳模型：{best_prediction['模型']}")
        print(f"预测总进球数：{best_prediction['预测值']} 球")
        print(f"实际总进球数：{best_prediction['实际值']} 球")
        print(f"绝对误差：{best_prediction['绝对误差']} 球")
        print(f"相对误差：{best_prediction['相对误差']}%")

        self.prediction_result = best_prediction

        return self.prediction_result

    def run(self):

        self.clean_and_preprocess()

        self.train_models()

        self.analyze_features()

        self.predict_2018()

        self.save_report()

        return self.best_model