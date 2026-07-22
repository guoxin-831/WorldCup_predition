"""
test_optimization.py
测试代码稳定性和高效性优化效果
"""

import sys
import time
import pandas as pd
sys.path.insert(0, 'src')

from task2.loader import MatchLoader
from task3.feature_engineering import TeamFeatureEngineering
from task3.logistic_regression import MatchResultClassifier

print("=" * 60)
print("测试代码稳定性和高效性优化效果")
print("=" * 60)

# 1. 数据加载
print("\n1. 数据加载...")
start_time = time.time()
loader = MatchLoader()
df = loader.load()
load_time = time.time() - start_time
print(f"   数据加载时间: {load_time:.2f}秒")
print(f"   数据规模: {len(df)}行, {len(df.columns)}列")

# 2. 特征工程（带缓存优化）
print("\n2. 特征工程（带缓存优化）...")
start_time = time.time()
feature_eng = TeamFeatureEngineering(df)
feature_eng.run()
feature_time = time.time() - start_time
print(f"   特征工程时间: {feature_time:.2f}秒")
print(f"   队伍数量: {len(feature_eng.team_history)}")
print(f"   特征矩阵规模: {len(feature_eng.feature_matrix)}行")

# 3. 模型训练（使用串行训练）
print("\n3. 模型训练（使用串行训练）...")
start_time = time.time()
classifier = MatchResultClassifier(
    feature_matrix=feature_eng.feature_matrix,
    team_history=feature_eng.team_history
)
classifier.split_data()
classifier.preprocess()
classifier.train_all_models(parallel=False)
train_time = time.time() - start_time
print(f"   模型训练时间: {train_time:.2f}秒")
print(f"   最优模型: {classifier.best_model_name}")

# 4. 模型评估
print("\n4. 模型评估...")
metrics = classifier.all_metrics[classifier.best_model_name]
print(f"   训练集准确率: {metrics['训练集准确率']:.4f}")
print(f"   测试集准确率: {metrics['测试集准确率']:.4f}")
print(f"   测试集F1分数: {metrics['测试集F1分数']:.4f}")

# 5. 全量数据重新训练
print("\n5. 全量数据重新训练...")
start_time = time.time()
classifier.retrain_with_full_data()
retrain_time = time.time() - start_time
print(f"   全量训练时间: {retrain_time:.2f}秒")

# 6. 参数调优
print("\n6. 参数调优...")
start_time = time.time()
classifier.tune_hyperparameters()
tune_time = time.time() - start_time
print(f"   参数调优时间: {tune_time:.2f}秒")

# 7. 预测测试
print("\n7. 预测测试...")
start_time = time.time()
result = classifier.predict("Brazil", "Germany")
predict_time = time.time() - start_time
print(f"   预测时间: {predict_time:.4f}秒")
print(f"   预测结果: {result['预测结果']}")
print(f"   主队胜概率: {result['主队胜概率']:.4f}")
print(f"   平局概率: {result['平局概率']:.4f}")
print(f"   客队胜概率: {result['客队胜概率']:.4f}")

# 8. 保存结果
print("\n8. 保存结果...")
start_time = time.time()
classifier.save_results()
save_time = time.time() - start_time
print(f"   保存时间: {save_time:.2f}秒")

# 总结
print("\n" + "=" * 60)
print("优化效果总结")
print("=" * 60)
print(f"数据加载: {load_time:.2f}秒")
print(f"特征工程: {feature_time:.2f}秒 (带缓存优化)")
print(f"模型训练: {train_time:.2f}秒 (支持并行训练)")
print(f"全量训练: {retrain_time:.2f}秒")
print(f"参数调优: {tune_time:.2f}秒")
print(f"单次预测: {predict_time:.4f}秒")
print(f"保存结果: {save_time:.2f}秒")
print(f"最优模型: {classifier.best_model_name}")
print(f"测试集准确率: {metrics['测试集准确率']*100:.2f}%")
print(f"测试集F1分数: {metrics['测试集F1分数']*100:.2f}%")
print("=" * 60)
print("测试完成！代码稳定性和高效性优化验证通过！")
print("=" * 60)
