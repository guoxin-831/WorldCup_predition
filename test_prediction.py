import sys
sys.path.insert(0, 'src')
from task2.loader import MatchLoader
from task3.feature_engineering import TeamFeatureEngineering
from task3.logistic_regression import MatchResultClassifier
import warnings
warnings.filterwarnings('ignore')

loader = MatchLoader()
df = loader.load()

feature_eng = TeamFeatureEngineering(df)
feature_eng.run()

classifier = MatchResultClassifier(feature_eng.feature_matrix, feature_eng.team_history)
classifier.run()

print('\n=== 测试预测 ===')
print(f'当前最优模型: {classifier.best_model_name}')
print(f'模型类别: {classifier.best_model.classes_}')

test_cases = [
    ('France', 'England'), 
    ('England', 'France'),
    ('Brazil', 'Germany'), 
    ('Germany', 'Brazil'),
    ('Argentina', 'Spain'), 
    ('Italy', 'France')
]
for home, away in test_cases:
    result = classifier.predict(home, away)
    pred = result['预测结果']
    h_prob = result['主队胜概率']
    d_prob = result['平局概率']
    a_prob = result['客队胜概率']
    print(f'{home} vs {away}: {pred}')
    print(f'  概率: 主队胜={h_prob*100:.1f}%, 平局={d_prob*100:.1f}%, 客队胜={a_prob*100:.1f}%')
    print()
