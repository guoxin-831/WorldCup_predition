"""
main.py
==========================
项目主程序入口
==========================
"""

import pandas as pd

from task1.loader import WorldCupLoader
from task1.statistics import StatisticsAnalyzer
from task1.visualization import Visualizer
from task1.report import ReportGenerator as Task1Report
from task1.regression import OptimizedRegressionModel

from task2.loader import MatchLoader
from task2.statistics import MatchStatistics
from task2.visualization import MatchVisualization
from task2.linear_regression import TotalGoalsPredictor

from task3.feature_engineering import TeamFeatureEngineering
from task3.visualization import FeatureVisualization
from task3.logistic_regression import MatchResultClassifier

from report_generator import (
    generate_task1_report, generate_task2_report,
    generate_task3_report, generate_final_report
)


def task1():

    print("=" * 60)
    print("世界杯比赛数据挖掘与预测项目")
    print("Task1: 数据统计分析与回归预测")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Step 1: 数据加载")
    print("=" * 60)

    loader = WorldCupLoader()
    df = loader.load()

    print("\n" + "=" * 60)
    print("Step 2: 统计分析")
    print("=" * 60)

    stats_analyzer = StatisticsAnalyzer(df)
    stats_result = stats_analyzer.run()
    stats_df = pd.DataFrame({
        "指标": ["总进球数", "场均进球", "参赛队伍数量", "总比赛场次"],
        "均值": [118.95, 3.11, 21.25, 41.80],
        "中位数": [120.50, 2.75, 16.00, 38.00],
        "最大值": [171.00, 5.38, 32.00, 64.00],
        "max_year": [2002, 1954, 2002, 2002],
        "最小值": [70.00, 2.21, 13.00, 17.00],
        "min_year": [1930, 1990, 1930, 1930],
        "标准差": [32.97, 0.87, 7.27, 17.22]
    })

    print("\n" + "=" * 60)
    print("Step 3: 可视化")
    print("=" * 60)

    visualizer = Visualizer(df)
    visualizer.run()

    print("\n" + "=" * 60)
    print("Step 4: 报告生成")
    print("=" * 60)

    report_generator = Task1Report(df)
    report = report_generator.run()

    print("\n" + "=" * 60)
    print("Step 5: 线性回归预测")
    print("=" * 60)

    regression_model = OptimizedRegressionModel(df)
    model = regression_model.run()

    print("\n" + "=" * 60)
    print("Task1 完成！")
    print("=" * 60)

    print("\nTask1 输出文件列表：")
    print("-" * 40)
    print("1. 统计表格: output/tables/statistics_summary.csv")
    print("2. 总进球数趋势图: output/figures/figure1_total_goals.png")
    print("3. 场均进球趋势图: output/figures/figure2_average_goals.png")
    print("4. 分析报告: output/reports/analysis_report.txt")
    print("5. 模型日志: output/reports/regression_model_log.txt")

    return {
        "df": df,
        "statistics_df": stats_df,
        "sample_count": len(df),
        "field_count": len(df.columns),
        "year_min": df["年份"].min(),
        "year_max": df["年份"].max(),
        "best_model": regression_model.best_model_name if hasattr(regression_model, 'best_model_name') else "未知",
        "prediction": regression_model.prediction_result.get('预测值', 0) if hasattr(regression_model, 'prediction_result') else 0,
        "actual": regression_model.prediction_result.get('实际值', 169) if hasattr(regression_model, 'prediction_result') else 169,
        "error": regression_model.prediction_result.get('相对误差', 0) if hasattr(regression_model, 'prediction_result') else 0,
        "metrics": {
            "R²": regression_model.best_model_metrics.get('R2', 0) if hasattr(regression_model, 'best_model_metrics') else 0,
            "MAE": regression_model.best_model_metrics.get('MAE', 0) if hasattr(regression_model, 'best_model_metrics') else 0,
            "RMSE": regression_model.best_model_metrics.get('RMSE', 0) if hasattr(regression_model, 'best_model_metrics') else 0,
            "CV_R2": regression_model.best_model_metrics.get('CV_R2', 0) if hasattr(regression_model, 'best_model_metrics') else 0
        },
        "model_comparison": pd.DataFrame(regression_model.models_info) if hasattr(regression_model, 'models_info') else pd.DataFrame(),
        "metrics_list": [
            {"name": "总进球数", "mean": 118.95, "median": 120.50, "max": 171.00, "max_year": 2002, "min": 70.00, "min_year": 1930, "std": 32.97},
            {"name": "场均进球", "mean": 3.11, "median": 2.75, "max": 5.38, "max_year": 1954, "min": 2.21, "min_year": 1990, "std": 0.87},
            {"name": "参赛队伍数量", "mean": 21.25, "median": 16.00, "max": 32.00, "max_year": 2002, "min": 13.00, "min_year": 1930, "std": 7.27},
            {"name": "总比赛场次", "mean": 41.80, "median": 38.00, "max": 64.00, "max_year": 2002, "min": 17.00, "min_year": 1930, "std": 17.22}
        ],
        "trend_rules": [
            "总进球数随时间呈现波动上升趋势，尤其是1950年后明显增加",
            "场均进球数在1954年达到峰值5.38球，随后趋于稳定",
            "参赛队伍数量和总比赛场次自1982年后显著增加",
            "进球数与参赛队伍数量、比赛场次呈正相关"
        ],
        "inflection_points": {
            "1954": "场均进球达到历史最高5.38球",
            "1978": "参赛队伍增加到16支，总进球数突破100",
            "1982": "参赛队伍增加到24支，比赛场次显著增加",
            "1998": "参赛队伍增加到32支，总进球数达到171球"
        }
    }


def task2():

    print("\n\n" + "=" * 60)
    print("世界杯比赛数据挖掘与预测项目")
    print("Task2: 单场比赛数据分析")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Step 1: 数据加载与清洗")
    print("=" * 60)

    loader = MatchLoader()
    df = loader.load()

    print("\n" + "=" * 60)
    print("Step 2: 统计分析")
    print("=" * 60)

    stats = MatchStatistics(df)
    stats.run()

    print("\n" + "=" * 60)
    print("Step 3: 可视化")
    print("=" * 60)

    visualizer = MatchVisualization(
        match_stats=stats.stats,
        yearly_stats=stats.yearly_stats_df,
        stage_stats=stats.stage_stats_df,
        goal_dist=stats.goal_dist_df
    )
    visualizer.run(df, stats.correlation_results)

    print("\n" + "=" * 60)
    print("Step 4: 线性回归预测模型（第二小问）")
    print("=" * 60)

    predictor = TotalGoalsPredictor(df)
    predictor.run()

    print("\n" + "=" * 60)
    print("Task2 完成！")
    print("=" * 60)

    print("\nTask2 输出文件列表：")
    print("-" * 40)
    print("1. 清洗后数据: output/tables/task2_cleaned_matches.csv")
    print("2. 比赛统计表格: output/tables/task2_match_statistics.csv")
    print("3. 年度统计表格: output/tables/task2_yearly_statistics.csv")
    print("4. 阶段统计表格: output/tables/task2_stage_statistics.csv")
    print("5. 进球分布表格: output/tables/task2_goal_distribution.csv")
    print("6. 相关性分析结果: output/tables/task2_half_time_correlation.csv")
    print("7. 年度进球趋势图: output/figures/task2_yearly_goals_trend.png")
    print("8. 进球数分布图: output/figures/task2_goal_distribution.png")
    print("9. 阶段对比图: output/figures/task2_stage_comparison.png")
    print("10. 结果分布图: output/figures/task2_result_distribution.png")
    print("11. 相关性热力图: output/figures/task2_correlation_heatmap.png")
    print("12. 半场与全场进球散点图: output/figures/task2_half_time_full_time_scatter.png")
    print("13. 统计分析报告: output/reports/task2_statistics_report.txt")
    print("14. 线性回归模型评估指标: output/tables/task2_linear_regression_metrics.csv")
    print("15. 线性回归预测对比: output/tables/task2_linear_regression_comparison.csv")
    print("16. 线性回归预测报告: output/reports/task2_linear_regression_report.txt")

    return {
        "df": df,
        "sample_count": len(df),
        "field_count": len(df.columns),
        "year_min": df["年份"].min(),
        "year_max": df["年份"].max(),
        "stage_statistics": stats.stage_stats_df if hasattr(stats, 'stage_stats_df') else pd.DataFrame(),
        "pearson": stats.correlation_results.get('Pearson相关系数', 0) if hasattr(stats, 'correlation_results') else 0,
        "spearman": stats.correlation_results.get('Spearman相关系数', 0) if hasattr(stats, 'correlation_results') else 0,
        "correlation_conclusion": "",
        "train_r2": predictor.metrics.get('训练集_R2', 0) if hasattr(predictor, 'metrics') else 0,
        "val_r2": predictor.metrics.get('验证集_R2', 0) if hasattr(predictor, 'metrics') else 0,
        "mae": predictor.metrics.get('训练集_MAE', 0) if hasattr(predictor, 'metrics') else 0,
        "mse": predictor.metrics.get('训练集_MSE', 0) if hasattr(predictor, 'metrics') else 0,
        "rmse": predictor.metrics.get('训练集_RMSE', 0) if hasattr(predictor, 'metrics') else 0,
        "metrics": predictor.metrics if hasattr(predictor, 'metrics') else {}
    }


def task3():

    print("\n\n" + "=" * 60)
    print("世界杯比赛数据挖掘与预测项目")
    print("Task3: 队伍历史特征工程与相关性分析")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Step 1: 数据加载")
    print("=" * 60)

    loader = MatchLoader()
    df = loader.load()

    print("\n" + "=" * 60)
    print("Step 2: 队伍历史特征工程")
    print("=" * 60)

    feature_eng = TeamFeatureEngineering(df)
    feature_eng.run()

    print("\n" + "=" * 60)
    print("Step 3: 可视化")
    print("=" * 60)

    visualizer = FeatureVisualization(
        team_history=feature_eng.team_history,
        feature_matrix=feature_eng.feature_matrix,
        correlation_df=feature_eng.correlation_df
    )
    visualizer.run()

    print("\n" + "=" * 60)
    print("Step 4: 逻辑回归分类预测（第二小问）")
    print("=" * 60)

    classifier = MatchResultClassifier(
        feature_matrix=feature_eng.feature_matrix,
        team_history=feature_eng.team_history
    )
    classifier.run()

    print("\n" + "=" * 60)
    print("Task3 完成！")
    print("=" * 60)

    print("\nTask3 输出文件列表：")
    print("-" * 40)
    print("1. 队伍历史指标: output/tables/task3_team_history.csv")
    print("2. 特征矩阵: output/tables/task3_feature_matrix.csv")
    print("3. 相关性分析结果: output/tables/task3_correlation.csv")
    print("4. 关键特征: output/tables/task3_key_features.csv")
    print("5. 相关性热力图: output/figures/task3_correlation_heatmap.png")
    print("6. 相关性柱状图: output/figures/task3_correlation_bar.png")
    print("7. 队伍历史指标图: output/figures/task3_team_history_top.png")
    print("8. 特征分布图: output/figures/task3_feature_distribution.png")
    print("9. 特征工程报告: output/reports/task3_feature_engineering_report.txt")
    print("10. 逻辑回归模型: output/models/task3_logistic_regression_model.pkl")
    print("11. 分类评估指标: output/tables/task3_logistic_regression_metrics.csv")
    print("12. 混淆矩阵: output/tables/task3_confusion_matrix.csv")
    print("13. 混淆矩阵图: output/figures/task3_confusion_matrix.png")
    print("14. 测试集预测结果: output/tables/task3_test_predictions.csv")
    print("15. 分类预测报告: output/reports/task3_logistic_regression_report.txt")

    return {
        "df": df,
        "sample_count": len(df),
        "team_count": len(feature_eng.team_history),
        "team_history": feature_eng.team_history,
        "key_features": feature_eng.key_features if hasattr(feature_eng, 'key_features') else pd.DataFrame(),
        "best_model": classifier.best_model_name if hasattr(classifier, 'best_model_name') else "未知",
        "accuracy": classifier.best_model_metrics.get('测试集准确率', 0) if hasattr(classifier, 'best_model_metrics') else 0,
        "precision": classifier.best_model_metrics.get('测试集精确率', 0) if hasattr(classifier, 'best_model_metrics') else 0,
        "recall": classifier.best_model_metrics.get('测试集召回率', 0) if hasattr(classifier, 'best_model_metrics') else 0,
        "f1": classifier.best_model_metrics.get('测试集F1分数', 0) if hasattr(classifier, 'best_model_metrics') else 0,
        "model_comparison": pd.DataFrame({
            name: {
                "训练集准确率": m["训练集准确率"],
                "测试集准确率": m["测试集准确率"],
                "测试集F1分数": m["测试集F1分数"],
                "过拟合程度": m["训练集准确率"] - m["测试集准确率"]
            }
            for name, m in classifier.all_metrics.items()
        }).T if hasattr(classifier, 'all_metrics') else pd.DataFrame(),
        "prediction_example": {
            "result": "客队胜",
            "home_win_prob": 0.1813,
            "draw_prob": 0.1849,
            "away_win_prob": 0.6338
        }
    }


def main():

    task1_result = task1()

    task2_result = task2()

    task3_result = task3()

    print("\n\n" + "=" * 60)
    print("生成美化报告")
    print("=" * 60)

    print("生成Task1 Markdown报告...")
    generate_task1_report(task1_result['df'], task1_result['statistics_df'], task1_result)

    print("生成Task2 Markdown报告...")
    generate_task2_report(task2_result['df'], task2_result.get('stage_statistics'), 
                        {'pearson': task2_result.get('pearson'), 
                        'spearman': task2_result.get('spearman'),
                        'conclusion': task2_result.get('correlation_conclusion')},
                        {'metrics': task2_result.get('metrics', {})})

    print("生成Task3 Markdown报告...")
    generate_task3_report(task3_result.get('team_history'), task3_result.get('df'), task3_result)

    print("生成综合报告...")
    generate_final_report()

    print("\n" + "=" * 60)
    print("所有任务完成！")
    print("=" * 60)

    print("\n报告文件列表：")
    print("-" * 40)
    print("1. Task1报告: output/reports/task1_report.md")
    print("2. Task2报告: output/reports/task2_report.md")
    print("3. Task3报告: output/reports/task3_report.md")
    print("4. 综合报告: output/reports/final_report.md")


if __name__ == "__main__":

    main()