"""
linear_regression.py
==========================
Task2 第二小问：基于线性回归的总进球数预测
==========================
功能：
1. 离散特征"赛事阶段"数值编码
2. 整合上半场进球、观众人数等特征
3. 自动划分训练集/验证集
4. 模型训练与评估（MAE、MSE、R²）
5. 可复用预测函数，支持自定义特征输入
==========================
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from config import REPORT_DIR, MODEL_DIR, TABLE_DIR


class TotalGoalsPredictor:
    """
    基于线性回归的总进球数预测器
    """

    def __init__(self, df):

        self.df = df.copy()

        self.feature_columns = [
            "半场总进球", "观众人数", "阶段编码",
            "半场主队进球", "半场客队进球",
            "主队进球", "客队进球", "进球差"
        ]

        self.target_column = "总进球数"

        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.model = LinearRegression()

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.metrics = {}

    def preprocess(self):

        print("\n" + "=" * 60)
        print("Step 1: 数据预处理与特征工程")
        print("=" * 60)

        df = self.df.copy()

        df["阶段编码"] = self.label_encoder.fit_transform(df["阶段"])

        print(f"\n赛事阶段编码映射：")
        for stage, code in zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))):
            count = (df["阶段"] == stage).sum()
            print(f"  {stage} -> {code} (样本数: {count})")

        feature_cols = [col for col in self.feature_columns if col in df.columns]
        self.feature_columns = feature_cols

        df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
        df = df.dropna(subset=feature_cols + [self.target_column])

        self.processed_df = df

        print(f"\n最终特征列表：{feature_cols}")
        print(f"处理后样本数量：{len(df)}")

        return df

    def split_data(self, test_size=0.2, random_state=42):

        print("\n" + "=" * 60)
        print("Step 2: 数据集划分")
        print("=" * 60)

        X = self.processed_df[self.feature_columns]
        y = self.processed_df[self.target_column]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        print(f"训练集样本数：{len(self.X_train)}")
        print(f"验证集样本数：{len(self.X_test)}")
        print(f"训练集占比：{(1-test_size)*100:.0f}%")
        print(f"验证集占比：{test_size*100:.0f}%")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def train(self):

        print("\n" + "=" * 60)
        print("Step 3: 线性回归模型训练")
        print("=" * 60)

        self.model.fit(self.X_train_scaled, self.y_train)

        print(f"\n模型系数：")
        for feature, coef in zip(self.feature_columns, self.model.coef_):
            print(f"  {feature}: {coef:.4f}")
        print(f"截距: {self.model.intercept_:.4f}")

        return self.model

    def evaluate(self):

        print("\n" + "=" * 60)
        print("Step 4: 模型评估")
        print("=" * 60)

        y_train_pred = self.model.predict(self.X_train_scaled)
        y_test_pred = self.model.predict(self.X_test_scaled)

        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        train_mse = mean_squared_error(self.y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_r2 = r2_score(self.y_train, y_train_pred)

        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        test_mse = mean_squared_error(self.y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(self.y_test, y_test_pred)

        self.metrics = {
            "训练集_MAE": train_mae,
            "训练集_MSE": train_mse,
            "训练集_RMSE": train_rmse,
            "训练集_R2": train_r2,
            "验证集_MAE": test_mae,
            "验证集_MSE": test_mse,
            "验证集_RMSE": test_rmse,
            "验证集_R2": test_r2
        }

        print(f"\n训练集评估指标：")
        print(f"  MAE  (平均绝对误差): {train_mae:.4f}")
        print(f"  MSE  (均方误差):     {train_mse:.4f}")
        print(f"  RMSE (均方根误差):   {train_rmse:.4f}")
        print(f"  R²   (决定系数):     {train_r2:.4f}")

        print(f"\n验证集评估指标：")
        print(f"  MAE  (平均绝对误差): {test_mae:.4f}")
        print(f"  MSE  (均方误差):     {test_mse:.4f}")
        print(f"  RMSE (均方根误差):   {test_rmse:.4f}")
        print(f"  R²   (决定系数):     {test_r2:.4f}")

        print("\n" + "-" * 60)
        print("评估指标解释：")
        print("-" * 60)
        print(f"MAE (平均绝对误差) = {test_mae:.4f}")
        print("  含义: 预测值与实际值之差的绝对值的平均")
        print(f"  解释: 模型预测的平均误差约为 {test_mae:.2f} 个进球")
        print(f"  范围: [0, +∞)，越小越好，0表示完美预测")
        print()
        print(f"MSE (均方误差) = {test_mse:.4f}")
        print("  含义: 预测值与实际值之差的平方的平均")
        print(f"  解释: 放大异常误差的影响，值为 {test_mse:.2f}")
        print(f"  范围: [0, +∞)，越小越好，0表示完美预测")
        print()
        print(f"RMSE (均方根误差) = {test_rmse:.4f}")
        print("  含义: 均方误差的平方根，与原数据同量纲")
        print(f"  解释: 预测误差的标准差约为 {test_rmse:.2f} 个进球")
        print(f"  范围: [0, +∞)，越小越好，0表示完美预测")
        print()
        print(f"R² (决定系数) = {test_r2:.4f}")
        print("  含义: 模型解释的目标变量方差比例")
        if test_r2 >= 0.9:
            r2_level = "极好"
        elif test_r2 >= 0.7:
            r2_level = "良好"
        elif test_r2 >= 0.5:
            r2_level = "一般"
        elif test_r2 >= 0.3:
            r2_level = "较弱"
        else:
            r2_level = "很差"
        print(f"  解释: 模型能解释 {test_r2*100:.2f}% 的进球数变化，拟合效果{r2_level}")
        print(f"  范围: (-∞, 1]，越接近1越好，1表示完美拟合")
        print()

        overfitting = train_r2 - test_r2
        print(f"过拟合分析：")
        print(f"  训练集R² - 验证集R² = {overfitting:.4f}")
        if overfitting > 0.1:
            print(f"  结论: 差值较大，模型可能存在过拟合")
        elif overfitting > 0.05:
            print(f"  结论: 差值适中，模型有轻微过拟合")
        else:
            print(f"  结论: 差值较小，模型泛化能力良好")

        comparison_df = pd.DataFrame({
            "实际值": self.y_test.values,
            "预测值": y_test_pred,
            "误差": self.y_test.values - y_test_pred
        })
        self.comparison_df = comparison_df

        print(f"\n验证集预测对比（前10条）：")
        print("-" * 40)
        print(comparison_df.head(10).to_string(index=False))

        return self.metrics

    def predict(self, half_time_goals, attendance, stage,
                half_home_goals=None, half_away_goals=None,
                home_goals=None, away_goals=None):

        if stage in self.label_encoder.classes_:
            stage_code = self.label_encoder.transform([stage])[0]
        else:
            stage_code = len(self.label_encoder.classes_)
            print(f"警告：阶段 '{stage}' 不在训练数据中，使用默认编码 {stage_code}")

        if half_home_goals is None or half_away_goals is None:
            half_home_goals = half_time_goals // 2
            half_away_goals = half_time_goals - half_home_goals

        if home_goals is None or away_goals is None:
            home_goals = half_home_goals
            away_goals = half_away_goals

        goal_diff = home_goals - away_goals

        input_data = {
            "半场总进球": half_time_goals,
            "观众人数": attendance,
            "阶段编码": stage_code,
            "半场主队进球": half_home_goals,
            "半场客队进球": half_away_goals,
            "主队进球": home_goals,
            "客队进球": away_goals,
            "进球差": goal_diff
        }

        input_df = pd.DataFrame([input_data])
        input_scaled = self.scaler.transform(input_df[self.feature_columns])

        prediction = self.model.predict(input_scaled)[0]

        print("\n" + "=" * 60)
        print("预测结果")
        print("=" * 60)
        print(f"输入特征参数：")
        for key, value in input_data.items():
            print(f"  {key}: {value}")
        print(f"\n预测总进球数: {prediction:.2f}")
        print(f"预测总进球数(四舍五入): {round(prediction)}")

        return prediction

    def predict_2026_group_match(self, half_time_goals=1, attendance=50000,
                                  half_home_goals=None, half_away_goals=None,
                                  home_goals=None, away_goals=None):

        print("\n" + "=" * 60)
        print("预测2026年世界杯小组赛总进球数")
        print("=" * 60)

        return self.predict(
            half_time_goals=half_time_goals,
            attendance=attendance,
            stage="小组赛",
            half_home_goals=half_home_goals,
            half_away_goals=half_away_goals,
            home_goals=home_goals,
            away_goals=away_goals
        )

    def save_results(self):

        print("\n" + "=" * 60)
        print("Step 5: 保存模型与结果")
        print("=" * 60)

        model_path = MODEL_DIR / "task2_linear_regression_model.pkl"
        joblib.dump(self.model, model_path)
        print(f"✓ 模型已保存到: {model_path}")

        scaler_path = MODEL_DIR / "task2_linear_regression_scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        print(f"✓ 标准化器已保存到: {scaler_path}")

        encoder_path = MODEL_DIR / "task2_linear_regression_encoder.pkl"
        joblib.dump(self.label_encoder, encoder_path)
        print(f"✓ 编码器已保存到: {encoder_path}")

        metrics_df = pd.DataFrame([self.metrics])
        metrics_path = TABLE_DIR / "task2_linear_regression_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
        print(f"✓ 评估指标已保存到: {metrics_path}")

        comparison_path = TABLE_DIR / "task2_linear_regression_comparison.csv"
        self.comparison_df.to_csv(comparison_path, index=False, encoding="utf-8-sig")
        print(f"✓ 预测对比已保存到: {comparison_path}")

        self.generate_report()

        return True

    def generate_report(self):

        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("Task2 第二小问：线性回归预测模型报告")
        report_lines.append("=" * 70)
        report_lines.append("")

        report_lines.append("一、数据预处理与特征工程")
        report_lines.append("-" * 40)
        report_lines.append(f"样本数量: {len(self.processed_df)}")
        report_lines.append(f"特征数量: {len(self.feature_columns)}")
        report_lines.append(f"特征列表: {', '.join(self.feature_columns)}")
        report_lines.append("")
        report_lines.append("赛事阶段编码映射:")
        for stage, code in zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))):
            report_lines.append(f"  {stage} -> {code}")
        report_lines.append("")

        report_lines.append("二、数据集划分")
        report_lines.append("-" * 40)
        report_lines.append(f"训练集样本数: {len(self.X_train)}")
        report_lines.append(f"验证集样本数: {len(self.X_test)}")
        report_lines.append("")

        report_lines.append("三、模型参数")
        report_lines.append("-" * 40)
        report_lines.append("模型系数:")
        for feature, coef in zip(self.feature_columns, self.model.coef_):
            report_lines.append(f"  {feature}: {coef:.4f}")
        report_lines.append(f"截距: {self.model.intercept_:.4f}")
        report_lines.append("")

        report_lines.append("四、评估指标")
        report_lines.append("-" * 40)
        report_lines.append("训练集:")
        report_lines.append(f"  MAE:  {self.metrics['训练集_MAE']:.4f}")
        report_lines.append(f"  MSE:  {self.metrics['训练集_MSE']:.4f}")
        report_lines.append(f"  RMSE: {self.metrics['训练集_RMSE']:.4f}")
        report_lines.append(f"  R²:   {self.metrics['训练集_R2']:.4f}")
        report_lines.append("")
        report_lines.append("验证集:")
        report_lines.append(f"  MAE:  {self.metrics['验证集_MAE']:.4f}")
        report_lines.append(f"  MSE:  {self.metrics['验证集_MSE']:.4f}")
        report_lines.append(f"  RMSE: {self.metrics['验证集_RMSE']:.4f}")
        report_lines.append(f"  R²:   {self.metrics['验证集_R2']:.4f}")
        report_lines.append("")

        report_lines.append("评估指标详细解释:")
        report_lines.append("-" * 40)
        report_lines.append(f"MAE (平均绝对误差, Mean Absolute Error) = {self.metrics['验证集_MAE']:.4f}")
        report_lines.append("  计算公式: MAE = (1/n) * Σ|y_true - y_pred|")
        report_lines.append("  含义: 预测值与实际值之差的绝对值的平均")
        report_lines.append(f"  解释: 模型预测的平均误差约为 {self.metrics['验证集_MAE']:.2f} 个进球")
        report_lines.append("  范围: [0, +∞)，越小越好，0表示完美预测")
        report_lines.append("  特点: 对异常值不敏感，易于解释")
        report_lines.append("")
        report_lines.append(f"MSE (均方误差, Mean Squared Error) = {self.metrics['验证集_MSE']:.4f}")
        report_lines.append("  计算公式: MSE = (1/n) * Σ(y_true - y_pred)²")
        report_lines.append("  含义: 预测值与实际值之差的平方的平均")
        report_lines.append(f"  解释: 放大异常误差的影响，值为 {self.metrics['验证集_MSE']:.2f}")
        report_lines.append("  范围: [0, +∞)，越小越好，0表示完美预测")
        report_lines.append("  特点: 对异常值敏感，惩罚较大误差")
        report_lines.append("")
        report_lines.append(f"RMSE (均方根误差, Root Mean Squared Error) = {self.metrics['验证集_RMSE']:.4f}")
        report_lines.append("  计算公式: RMSE = √MSE")
        report_lines.append("  含义: 均方误差的平方根，与原数据同量纲")
        report_lines.append(f"  解释: 预测误差的标准差约为 {self.metrics['验证集_RMSE']:.2f} 个进球")
        report_lines.append("  范围: [0, +∞)，越小越好，0表示完美预测")
        report_lines.append("  特点: 量纲与目标变量一致，直观易理解")
        report_lines.append("")
        report_lines.append(f"R² (决定系数, R-squared) = {self.metrics['验证集_R2']:.4f}")
        report_lines.append("  计算公式: R² = 1 - SS_res/SS_tot")
        report_lines.append("  含义: 模型解释的目标变量方差比例")
        r2_val = self.metrics['验证集_R2']
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
        report_lines.append(f"  解释: 模型能解释 {r2_val*100:.2f}% 的进球数变化，拟合效果{r2_level}")
        report_lines.append("  范围: (-∞, 1]，越接近1越好，1表示完美拟合")
        report_lines.append("  特点: 衡量模型整体拟合优度，是最常用的评估指标")
        report_lines.append("")

        overfitting = self.metrics['训练集_R2'] - self.metrics['验证集_R2']
        report_lines.append("过拟合分析:")
        report_lines.append(f"  训练集R² - 验证集R² = {overfitting:.4f}")
        if overfitting > 0.1:
            report_lines.append("  结论: 差值较大，模型可能存在过拟合，建议增加数据或正则化")
        elif overfitting > 0.05:
            report_lines.append("  结论: 差值适中，模型有轻微过拟合")
        else:
            report_lines.append("  结论: 差值较小，模型泛化能力良好")
        report_lines.append("")

        report_lines.append("五、预测函数使用说明")
        report_lines.append("-" * 40)
        report_lines.append("可通过 predict() 函数输入自定义特征参数进行预测:")
        report_lines.append("参数说明:")
        report_lines.append("  half_time_goals: 上半场总进球数")
        report_lines.append("  attendance: 观众人数")
        report_lines.append("  stage: 赛事阶段(如: 小组赛、1/8决赛、决赛等)")
        report_lines.append("  half_home_goals: 半场主队进球(可选)")
        report_lines.append("  half_away_goals: 半场客队进球(可选)")
        report_lines.append("  home_goals: 主队进球(可选)")
        report_lines.append("  away_goals: 客队进球(可选)")
        report_lines.append("")

        report_lines.append("=" * 70)
        report_lines.append("报告生成完毕")
        report_lines.append("=" * 70)

        report = "\n".join(report_lines)

        report_path = REPORT_DIR / "task2_linear_regression_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"✓ 预测报告已保存到: {report_path}")

        return report_path

    def run(self):

        self.preprocess()
        self.split_data()
        self.train()
        self.evaluate()
        self.save_results()

        self.predict_2026_group_match(
            half_time_goals=1,
            attendance=50000,
            half_home_goals=1,
            half_away_goals=0,
            home_goals=2,
            away_goals=1
        )

        return self.model
