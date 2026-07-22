"""
debug_trainer.py
调试训练逻辑问题
"""

import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from task2.loader import MatchLoader
from task3.feature_engineering import TeamFeatureEngineering

print("=" * 60)
print("调试训练逻辑")
print("=" * 60)

# 1. 数据加载
print("\n1. 数据加载...")
loader = MatchLoader()
df = loader.load()
print(f"   数据规模: {len(df)}行")

# 2. 特征工程
print("\n2. 特征工程...")
feature_eng = TeamFeatureEngineering(df)
feature_eng.run()
print(f"   特征矩阵规模: {len(feature_eng.feature_matrix)}行")
print(f"   特征列: {feature_eng.feature_matrix.columns.tolist()}")

# 3. 手动测试训练
print("\n3. 手动测试训练...")
feature_columns = [
    "阶段类型",
    "参赛次数差", "比赛场次差", "场均进球差", "场均失球差",
    "净胜球差", "成绩排名差", "近3届场均进球差", "近3届胜率差",
    "场均净胜球差", "淘汰赛胜率差", "小组赛胜率差",
    "场均半场进球差", "半场胜率差",
    "交锋胜场", "交锋平局", "交锋负场", "交锋净胜球",
    "交锋总场次", "交锋胜率"
]
feature_columns = [c for c in feature_columns if c in feature_eng.feature_matrix.columns]

target_column = "胜负结果"

X = feature_eng.feature_matrix[feature_columns]
y = feature_eng.feature_matrix[target_column]

print(f"   X形状: {X.shape}")
print(f"   y分布:\n{y.value_counts()}")

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\n   训练集: {len(X_train)}样本")
print(f"   测试集: {len(X_test)}样本")

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练逻辑回归
print("\n4. 训练逻辑回归...")
model = LogisticRegression(solver="lbfgs", max_iter=1000, class_weight="balanced", random_state=42)
model.fit(X_train_scaled, y_train)

# 评估
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"   训练集准确率: {train_acc:.4f}")
print(f"   测试集准确率: {test_acc:.4f}")
print(f"   预测结果分布:\n{pd.Series(y_test_pred).value_counts()}")

# 查看特征值范围
print("\n5. 特征值统计...")
print(X_train.describe())

print("\n" + "=" * 60)
print("调试完成!")
print("=" * 60)
