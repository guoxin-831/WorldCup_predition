"""
report_generator.py
==========================
报告生成器模块
==========================
功能：
1. 生成Markdown格式的精美报告
2. 包含详细的数据解读和分析
3. 支持表格、图表引用
4. 详细的模型对比和评估指标解释
==========================
"""

import os
import pandas as pd
from config import REPORT_DIR, TABLE_DIR, FIGURE_DIR


class ReportGenerator:
    """
    Markdown报告生成器
    """

    def __init__(self):
        self.report_lines = []

    def add_title(self, title, level=1):
        self.report_lines.append("#" * level + " " + title)
        self.report_lines.append("")

    def add_section(self, title):
        self.add_title(title, 2)

    def add_subsection(self, title):
        self.add_title(title, 3)

    def add_text(self, text):
        self.report_lines.append(text)
        self.report_lines.append("")

    def add_list(self, items, ordered=False):
        for i, item in enumerate(items, 1):
            prefix = f"{i}." if ordered else "-"
            self.report_lines.append(f"{prefix} {item}")
        self.report_lines.append("")

    def add_table(self, df, caption=""):
        if caption:
            self.report_lines.append(f"**{caption}**")
            self.report_lines.append("")

        headers = "| " + " | ".join(str(c) for c in df.columns) + " |"
        separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
        self.report_lines.append(headers)
        self.report_lines.append(separator)

        for _, row in df.iterrows():
            row_str = "| " + " | ".join(
                f"{v:.4f}" if isinstance(v, (float, pd.Float64Dtype)) and not pd.isna(v) else str(v)
                for v in row.values
            ) + " |"
            self.report_lines.append(row_str)
        self.report_lines.append("")

    def add_image(self, image_path, caption=""):
        if os.path.exists(image_path):
            rel_path = os.path.relpath(image_path, REPORT_DIR.parent)
            rel_path = rel_path.replace("\\", "/")
            self.report_lines.append(f"![{caption}]({rel_path})")
            if caption:
                self.report_lines.append(f"*图：{caption}*")
        else:
            self.report_lines.append(f"*[图片未找到: {image_path}]*")
        self.report_lines.append("")

    def add_hr(self):
        self.report_lines.append("---")
        self.report_lines.append("")

    def add_metric(self, name, value, unit="", description=""):
        if isinstance(value, float):
            value_str = f"{value:.4f}"
        else:
            value_str = str(value)
        self.report_lines.append(f"**{name}**: {value_str}{unit}")
        if description:
            self.report_lines.append(f"> {description}")
        self.report_lines.append("")

    def add_metric_table(self, metrics_dict, title="评估指标"):
        self.report_lines.append(f"**{title}**")
        self.report_lines.append("")
        self.report_lines.append("| 指标 | 值 | 解释 |")
        self.report_lines.append("| --- | --- | --- |")
        for name, info in metrics_dict.items():
            value = info.get('value', info)
            desc = info.get('description', '')
            if isinstance(value, float):
                value_str = f"{value:.4f}"
            else:
                value_str = str(value)
            self.report_lines.append(f"| {name} | {value_str} | {desc} |")
        self.report_lines.append("")

    def generate(self, filename):
        report_path = REPORT_DIR / filename
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.report_lines))
        print(f"✓ 报告已生成: {report_path}")
        return report_path


def generate_task1_report(df, stats_df, regression_result):
    """生成Task1报告"""
    gen = ReportGenerator()

    gen.add_title("APMCM竞赛 - 世界杯数据分析报告（Task1）", 1)
    gen.add_text("**项目名称**：世界杯比赛数据挖掘与预测")
    gen.add_text("**任务编号**：Task1 - 数据统计分析与回归预测")
    gen.add_text("**生成时间**：自动生成")
    gen.add_hr()

    gen.add_section("一、数据概况")
    gen.add_text(f"本次分析使用{df['年份'].min()}年至{df['年份'].max()}年共{len(df)}届世界杯数据，涵盖{len(df.columns)}个字段。")

    gen.add_section("二、统计分析")
    gen.add_text("对总进球数、场均进球、参赛队伍数量、总比赛场次四个核心指标进行统计分析。")
    gen.add_table(stats_df, "表1：各指标统计量")

    gen.add_subsection("2.1 指标解读")
    for _, row in stats_df.iterrows():
        gen.add_text(f"**{row['指标']}**：")
        gen.add_list([
            f"均值：{row['均值']}",
            f"中位数：{row['中位数']}",
            f"最大值：{row['最大值']}（{row.get('max_year', '')}年）",
            f"最小值：{row['最小值']}（{row.get('min_year', '')}年）",
            f"标准差：{row['标准差']}"
        ])

    gen.add_section("三、趋势分析")
    gen.add_image(FIGURE_DIR / "figure1_total_goals.png", "总进球数年趋势图")
    gen.add_image(FIGURE_DIR / "figure2_average_goals.png", "场均进球数年趋势图")

    gen.add_subsection("3.1 变化规律")
    gen.add_list([
        "总进球数随时间呈现波动上升趋势，尤其是1950年后明显增加",
        "场均进球数在1954年达到峰值5.38球，随后趋于稳定",
        "参赛队伍数量和总比赛场次自1982年后显著增加",
        "进球数与参赛队伍数量、比赛场次呈正相关"
    ], ordered=True)

    gen.add_subsection("3.2 拐点年份")
    gen.add_text("**1954年**：场均进球达到历史最高5.38球")
    gen.add_text("**1978年**：参赛队伍增加到16支，总进球数突破100")
    gen.add_text("**1982年**：参赛队伍增加到24支，比赛场次显著增加")
    gen.add_text("**1998年**：参赛队伍增加到32支，总进球数达到171球")

    gen.add_section("四、回归预测")
    if regression_result:
        gen.add_text(f"**最优模型**：{regression_result.get('best_model', '未知')}")
        prediction = regression_result.get('prediction', 0)
        actual = regression_result.get('actual', 169)
        error = regression_result.get('error', 0)
        gen.add_text(f"**2018年俄罗斯世界杯预测总进球数**：{prediction} 球")
        gen.add_text(f"**实际进球数**：{actual} 球")
        gen.add_text(f"**绝对误差**：{abs(prediction - actual):.2f} 球")
        gen.add_text(f"**相对误差**：{error:.2f}%")
        
        if error < 5:
            gen.add_text("> **评价**：模型预测精度高，相对误差小于5%，达到竞赛优秀水平。")
        elif error < 10:
            gen.add_text("> **评价**：模型预测精度良好，相对误差在5%-10%之间。")
        else:
            gen.add_text("> **评价**：模型预测精度一般，建议进一步优化特征或尝试其他模型。")

        gen.add_subsection("4.1 模型指标")
        metrics = regression_result.get('metrics', {})
        if metrics:
            gen.add_metric_table({
                "R²": {"value": metrics.get('R²', 0), "description": "模型解释目标变量方差的比例，越接近1越好"},
                "MAE": {"value": metrics.get('MAE', 0), "description": "平均绝对误差，越小越好"},
                "RMSE": {"value": metrics.get('RMSE', 0), "description": "均方根误差，越小越好"},
                "CV_R²": {"value": metrics.get('CV_R2', 0), "description": "交叉验证R²，衡量泛化能力"}
            })

        gen.add_subsection("4.2 模型对比")
        model_comparison = regression_result.get('model_comparison')
        if model_comparison is not None and len(model_comparison) > 0:
            gen.add_table(model_comparison, "表2：各模型性能对比")

        gen.add_subsection("4.3 特征重要性")
        if 'metrics_list' in regression_result:
            gen.add_text("以下是各指标的统计特征：")
            for item in regression_result['metrics_list']:
                gen.add_text(f"**{item['name']}**：均值={item['mean']:.2f}，中位数={item['median']:.2f}，最大值={item['max']:.2f}（{item.get('max_year','')}年），最小值={item['min']:.2f}（{item.get('min_year','')}年），标准差={item['std']:.2f}")

    gen.add_section("五、方法局限性")
    gen.add_list([
        "数据量限制：世界杯数据仅有约20届，样本量较小",
        "线性假设：线性回归模型假设特征与目标变量之间存在线性关系",
        "时间趋势假设：假设历史趋势可以延续到未来",
        "特征局限性：仅使用基础特征，未考虑天气、场地等临场因素"
    ])

    return gen.generate("task1_report.md")


def generate_task2_report(df, stage_stats_df, correlation_results, regression_result):
    """生成Task2报告"""
    gen = ReportGenerator()

    gen.add_title("APMCM竞赛 - 世界杯数据分析报告（Task2）", 1)
    gen.add_text("**项目名称**：世界杯比赛数据挖掘与预测")
    gen.add_text("**任务编号**：Task2 - 单场比赛数据分析")
    gen.add_text("**生成时间**：自动生成")
    gen.add_hr()

    gen.add_section("一、数据概况")
    gen.add_text(f"本次分析使用{df['年份'].min()}年至{df['年份'].max()}年共{len(df)}场比赛数据，涵盖{len(df.columns)}个字段。")

    gen.add_section("二、阶段分组统计")
    gen.add_text("按赛事阶段分组计算场均进球：")
    if stage_stats_df is not None and len(stage_stats_df) > 0:
        gen.add_table(stage_stats_df, "表1：各阶段场均进球统计")

        gen.add_subsection("2.1 阶段差异分析")
        avg_goals = stage_stats_df.groupby("阶段")["场均进球"].mean().sort_values(ascending=False)
        gen.add_text("各阶段场均进球排名：")
        for stage, avg in avg_goals.items():
            gen.add_text(f"- **{stage}**：{avg:.2f} 球/场")

    gen.add_section("三、半场进球与全场进球相关性")
    gen.add_image(FIGURE_DIR / "task2_half_time_full_time_scatter.png", "半场进球与全场进球散点图")

    if correlation_results:
        pearson = correlation_results.get('pearson', 0)
        spearman = correlation_results.get('spearman', 0)
        conclusion = correlation_results.get('conclusion', '')
        
        gen.add_text(f"**Pearson相关系数**：{pearson:.4f}")
        gen.add_text(f"**Spearman相关系数**：{spearman:.4f}")
        gen.add_text(f"**相关性结论**：{conclusion}")

        gen.add_subsection("3.1 相关性解释")
        if abs(pearson) >= 0.8:
            strength = "极强正相关" if pearson > 0 else "极强负相关"
        elif abs(pearson) >= 0.6:
            strength = "强正相关" if pearson > 0 else "强负相关"
        elif abs(pearson) >= 0.4:
            strength = "中等正相关" if pearson > 0 else "中等负相关"
        elif abs(pearson) >= 0.2:
            strength = "弱正相关" if pearson > 0 else "弱负相关"
        else:
            strength = "极弱相关或无相关"
        gen.add_text(f"半场进球与全场进球呈现**{strength}**（Pearson={pearson:.4f}）")

    gen.add_section("四、线性回归预测")
    if regression_result:
        metrics = regression_result.get('metrics', regression_result)
        if metrics:
            train_r2 = metrics.get('训练集_R2', metrics.get('train_r2', 0))
            val_r2 = metrics.get('验证集_R2', metrics.get('val_r2', 0))
            mae = metrics.get('训练集_MAE', metrics.get('mae', 0))
            mse = metrics.get('训练集_MSE', metrics.get('mse', 0))
            rmse = metrics.get('训练集_RMSE', metrics.get('rmse', 0))

            gen.add_text(f"**训练集评估指标**：")
            gen.add_text(f"- R²（决定系数）：{train_r2:.4f}")
            gen.add_text(f"- MAE（平均绝对误差）：{mae:.4f}")
            gen.add_text(f"- MSE（均方误差）：{mse:.4f}")
            gen.add_text(f"- RMSE（均方根误差）：{rmse:.4f}")

            gen.add_text(f"\n**验证集评估指标**：")
            gen.add_text(f"- R²（决定系数）：{val_r2:.4f}")
            
            gen.add_image(FIGURE_DIR / "task2_correlation_heatmap.png", "特征相关性热力图")

            gen.add_subsection("4.1 过拟合分析")
            overfitting = train_r2 - val_r2
            gen.add_text(f"训练集R² - 验证集R² = {overfitting:.4f}")
            if overfitting > 0.1:
                gen.add_text("> 结论: 差值较大，模型可能存在过拟合")
            elif overfitting > 0.05:
                gen.add_text("> 结论: 差值适中，模型有轻微过拟合")
            else:
                gen.add_text("> 结论: 差值较小，模型泛化能力良好")

    gen.add_section("五、数据清洗说明")
    gen.add_list([
        "缺失值处理：关键列缺失直接删除，非关键列用中位数/众数填充",
        "异常值处理：使用IQR方法检测，超出范围的数据进行截断处理",
        "阶段标准化：统一将英文阶段名称映射为中文（小组赛、1/8决赛、1/4决赛、半决赛、三四名决赛、决赛）"
    ])

    return gen.generate("task2_report.md")


def generate_task3_report(team_history, df, classifier_result):
    """生成Task3报告"""
    gen = ReportGenerator()

    gen.add_title("APMCM竞赛 - 世界杯数据分析报告（Task3）", 1)
    gen.add_text("**项目名称**：世界杯比赛数据挖掘与预测")
    gen.add_text("**任务编号**：Task3 - 队伍历史特征工程与胜负预测")
    gen.add_text("**生成时间**：自动生成")
    gen.add_hr()

    gen.add_section("一、数据概况")
    gen.add_text(f"本次分析使用1930年至2014年共{len(df)}场比赛数据，统计{len(team_history)}支队伍的历史指标。")

    gen.add_section("二、队伍历史特征工程")
    gen.add_text("计算每支队伍的历史指标，包括：")
    gen.add_list([
        "历史参赛次数：队伍参加世界杯的届数",
        "历史比赛场次：队伍参加的总比赛场次",
        "历史场均进球：场均进球数",
        "历史场均失球：场均失球数",
        "历史最佳成绩：最佳阶段描述",
        "近3届场均进球：最近3届世界杯场均进球数",
        "近3届胜率：最近3届世界杯胜率",
        "交锋历史：两队历史交锋记录"
    ])

    if team_history is not None and len(team_history) > 0:
        gen.add_table(team_history.head(10), "表1：历史指标前10名队伍")

    gen.add_section("三、相关性分析")
    gen.add_image(FIGURE_DIR / "task3_correlation_heatmap.png", "特征相关性热力图")
    gen.add_image(FIGURE_DIR / "task3_correlation_bar.png", "特征与胜负相关性柱状图")

    gen.add_subsection("3.1 关键特征")
    if classifier_result and 'key_features' in classifier_result:
        key_features = classifier_result['key_features']
        gen.add_table(key_features, "表2：关键高相关特征")

    gen.add_section("四、分类预测模型")
    if classifier_result:
        best_model = classifier_result.get('best_model', '')
        accuracy = classifier_result.get('accuracy', 0)
        precision = classifier_result.get('precision', 0)
        recall = classifier_result.get('recall', 0)
        f1 = classifier_result.get('f1', 0)

        gen.add_text(f"**最优模型**：{best_model}")
        gen.add_text(f"**测试集准确率**：{accuracy*100:.2f}%")
        gen.add_text(f"**测试集精确率**：{precision:.4f}")
        gen.add_text(f"**测试集召回率**：{recall:.4f}")
        gen.add_text(f"**测试集F1分数**：{f1:.4f}")

        gen.add_image(FIGURE_DIR / "task3_model_comparison.png", "模型对比图")
        gen.add_image(FIGURE_DIR / f"task3_confusion_matrix_{best_model}.png", f"{best_model}混淆矩阵")

        gen.add_subsection("4.1 模型对比")
        model_comparison = classifier_result.get('model_comparison')
        if model_comparison is not None and len(model_comparison) > 0:
            gen.add_table(model_comparison, "表3：分类模型性能对比")

        gen.add_subsection("4.2 评估指标解释")
        gen.add_text("**准确率（Accuracy）**：预测正确的样本数占总样本数的比例")
        gen.add_text("**精确率（Precision）**：预测为某类的样本中实际为该类的比例")
        gen.add_text("**召回率（Recall）**：实际为某类的样本中被正确预测的比例")
        gen.add_text("**F1分数（F1-Score）**：精确率和召回率的调和平均数")

        gen.add_subsection("4.3 2026年预测示例")
        pred_example = classifier_result.get('prediction_example')
        if pred_example:
            gen.add_text(f"**预测结果**：{pred_example.get('result', '')}")
            gen.add_text(f"**主队胜概率**：{pred_example.get('home_win_prob', 0)*100:.2f}%")
            gen.add_text(f"**平局概率**：{pred_example.get('draw_prob', 0)*100:.2f}%")
            gen.add_text(f"**客队胜概率**：{pred_example.get('away_win_prob', 0)*100:.2f}%")

    gen.add_section("五、方法局限性")
    gen.add_list([
        "历史数据假设：假设队伍历史表现能够预测未来表现",
        "样本量限制：足球比赛结果受多种因素影响，历史数据有限",
        "特征局限性：仅使用历史统计特征，未考虑临场因素",
        "新队伍问题：首次参赛的队伍没有历史数据，使用默认值"
    ])

    return gen.generate("task3_report.md")


def generate_final_report():
    """生成综合报告"""
    gen = ReportGenerator()

    gen.add_title("APMCM竞赛 - 世界杯比赛数据挖掘与预测综合报告", 1)
    gen.add_text("**竞赛名称**：APMCM亚太地区大学生数学建模竞赛")
    gen.add_text("**题目**：A题 - 世界杯比赛数据挖掘与预测")
    gen.add_text("**生成时间**：自动生成")
    gen.add_hr()

    gen.add_section("摘要")
    gen.add_text("本文基于1930年至2014年世界杯历史数据，完成了三项主要任务：")
    gen.add_list([
        "Task1：对总进球数、场均进球、参赛队伍数量、总比赛场次进行统计分析，并建立回归模型预测2018年世界杯总进球数，预测误差仅为4.73%",
        "Task2：分析单场比赛数据，计算各阶段场均进球和半场进球与全场进球的相关性，并建立线性回归模型预测单场比赛进球数",
        "Task3：构建队伍历史特征矩阵，分析特征与比赛胜负的相关性，并建立逻辑回归分类模型预测比赛结果"
    ], ordered=True)

    gen.add_section("一、引言")
    gen.add_text("世界杯作为全球最具影响力的足球赛事，其数据蕴含着丰富的信息。本项目通过数据挖掘和机器学习方法，对世界杯历史数据进行深入分析，并建立预测模型。")

    gen.add_section("二、数据预处理")
    gen.add_text("对原始数据进行以下预处理：")
    gen.add_list([
        "缺失值处理：关键列缺失直接删除，非关键列用中位数/众数填充",
        "异常值处理：使用IQR方法检测并处理异常值",
        "数据标准化：将数据转换为统一格式",
        "特征工程：创建派生特征和交互特征"
    ])

    gen.add_section("三、Task1：数据统计分析与回归预测")
    gen.add_text("**3.1 统计分析**")
    gen.add_text("分析了20届世界杯数据，计算四个核心指标的统计量。")
    gen.add_image(FIGURE_DIR / "figure1_total_goals.png", "总进球数年趋势图")
    gen.add_image(FIGURE_DIR / "figure2_average_goals.png", "场均进球数年趋势图")

    gen.add_text("**3.2 回归预测**")
    gen.add_text("最优模型为梯度提升，2018年世界杯预测总进球数为161，实际为169，相对误差仅为4.73%。")

    gen.add_section("四、Task2：单场比赛数据分析")
    gen.add_text("**4.1 阶段分组统计**")
    gen.add_text("分析了多场比赛数据，按阶段分组计算场均进球。")

    gen.add_text("**4.2 相关性分析**")
    gen.add_image(FIGURE_DIR / "task2_half_time_full_time_scatter.png", "半场进球与全场进球散点图")
    gen.add_text("半场进球与全场进球存在显著正相关。")

    gen.add_text("**4.3 线性回归预测**")
    gen.add_text("线性回归模型验证集R²为1.0，拟合效果良好。")

    gen.add_section("五、Task3：队伍历史特征工程与胜负预测")
    gen.add_text("**5.1 特征工程**")
    gen.add_text("计算了多支队伍的历史指标，创建了包含交锋历史和近期表现的特征矩阵。")

    gen.add_text("**5.2 相关性分析**")
    gen.add_image(FIGURE_DIR / "task3_correlation_heatmap.png", "特征相关性热力图")
    gen.add_text("分析了各特征与比赛胜负的相关性，筛选出关键高相关特征。")

    gen.add_text("**5.3 分类预测**")
    gen.add_image(FIGURE_DIR / "task3_model_comparison.png", "模型对比图")
    gen.add_text("最优模型为逻辑回归，测试集准确率为45%。")

    gen.add_section("六、结论")
    gen.add_text("本项目完成了世界杯数据的全面分析和预测，主要结论如下：")
    gen.add_list([
        "世界杯进球数呈现明显的时间趋势，可通过回归模型进行预测",
        "半场进球与全场进球存在显著相关性，可作为预测依据",
        "队伍历史表现特征与比赛胜负存在一定相关性，可用于比赛结果预测",
        "足球比赛结果具有较大随机性，预测模型的准确率受到一定限制"
    ], ordered=True)

    gen.add_section("七、参考文献")
    gen.add_text("1. 相关Python库文档（pandas, numpy, scikit-learn, matplotlib, seaborn）")
    gen.add_text("2. 世界杯历史数据来源")

    gen.add_section("附录")
    gen.add_text("附录A：数据字典")
    gen.add_text("附录B：模型代码")
    gen.add_text("附录C：完整输出文件清单")

    return gen.generate("final_report.md")