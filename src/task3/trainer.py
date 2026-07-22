"""
trainer.py
==========================
Task3 训练模块
==========================
功能：
1. 数据预处理（缺失值处理、标准化）
2. 模型训练（支持并行训练）
3. 模型评估（准确率、混淆矩阵、F1分数等）
4. 模型选择（选择最佳模型）
5. 全量数据重新训练
6. 参数调优
==========================
"""

import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score
)
from sklearn.base import BaseEstimator, ClassifierMixin
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.config import MODEL_DIR, TABLE_DIR, FIGURE_DIR, MODELS_NEED_SCALING, MODELS_NEED_INT_LABELS, RANDOM_SEED
from .utils import setup_logger, safe_divide, handle_exceptions
from .models import ModelFactory

logger = setup_logger(__name__)

RANDOM_SEED = 42


class Trainer:
    """
    模型训练器

    Attributes:
        feature_matrix: 特征矩阵
        feature_columns: 特征列名列表
        target_column: 目标列名
        scaler: 标准化器
        label_mapping: 标签映射
        models: 模型字典
        all_metrics: 所有模型的评估指标
        best_model_name: 最佳模型名称
        best_model: 最佳模型实例
    """

    def __init__(
        self,
        feature_matrix: pd.DataFrame,
        feature_columns: List[str],
        target_column: str = "胜负结果"
    ):
        """
        初始化训练器

        Args:
            feature_matrix: 特征矩阵
            feature_columns: 特征列名列表
            target_column: 目标列名
        """
        self.feature_matrix = feature_matrix.copy()
        self.feature_columns = [c for c in feature_columns if c in feature_matrix.columns]
        self.target_column = target_column
        self.scaler = StandardScaler()

        self.label_mapping = {"主队胜": 0, "平局": 1, "客队胜": 2}
        self.num_to_label = {0: "主队胜", 1: "平局", 2: "客队胜"}

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_train_num = None
        self.y_test_num = None
        self.X_train_scaled = None
        self.X_test_scaled = None

        self.models: Dict[str, Union[BaseEstimator, ClassifierMixin]] = {}
        self.all_metrics: Dict[str, Dict[str, float]] = {}
        self.best_model_name: Optional[str] = None
        self.best_model: Optional[Union[BaseEstimator, ClassifierMixin]] = None

        logger.info("训练器初始化完成")

    def split_data(self, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        划分训练集和测试集（分层抽样）

        Args:
            test_size: 测试集比例

        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info(f"开始划分数据集，测试集比例: {test_size}")

        train_df, test_df = train_test_split(
            self.feature_matrix,
            test_size=test_size,
            random_state=RANDOM_SEED,
            stratify=self.feature_matrix[self.target_column]
        )

        self.X_train = train_df[self.feature_columns]
        self.y_train = train_df[self.target_column]
        self.X_test = test_df[self.feature_columns]
        self.y_test = test_df[self.target_column]
        self.test_df = test_df

        self.y_train_num = self.y_train.map(self.label_mapping)
        self.y_test_num = self.y_test.map(self.label_mapping)

        logger.info(f"训练集样本数: {len(train_df)}, 测试集样本数: {len(test_df)}")
        logger.info(f"训练集胜负分布:\n{self.y_train.value_counts()}")
        logger.info(f"测试集胜负分布:\n{self.y_test.value_counts()}")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def preprocess(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        数据预处理（缺失值处理、标准化）

        Returns:
            X_train_scaled, X_test_scaled
        """
        logger.info("开始数据预处理")

        logger.debug(f"训练集缺失值: {self.X_train.isnull().sum().sum()}")
        logger.debug(f"测试集缺失值: {self.X_test.isnull().sum().sum()}")

        for col in self.feature_columns:
            if self.X_train[col].isnull().any():
                median_val = self.X_train[col].median()
                if pd.isna(median_val):
                    median_val = 0
                self.X_train[col] = self.X_train[col].fillna(median_val)
                self.X_test[col] = self.X_test[col].fillna(median_val)
                logger.debug(f"{col}: 用中位数 {median_val:.4f} 填充")

        self.X_train = self.X_train.reset_index(drop=True)
        self.y_train = self.y_train.reset_index(drop=True)
        self.X_test = self.X_test.reset_index(drop=True)
        self.y_test = self.y_test.reset_index(drop=True)

        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        logger.info(f"训练集标准化完成，形状: {self.X_train_scaled.shape}")
        logger.info(f"测试集标准化完成，形状: {self.X_test_scaled.shape}")

        return self.X_train_scaled, self.X_test_scaled

    def _train_single_model(
        self,
        model_name: str,
        model: Union[BaseEstimator, ClassifierMixin],
        X_train: np.ndarray,
        y_train: pd.Series,
        X_test: np.ndarray,
        y_test: pd.Series,
        X_train_raw: pd.DataFrame = None,
        X_test_raw: pd.DataFrame = None,
        y_train_raw: pd.Series = None,
        y_test_raw: pd.Series = None
    ) -> Tuple[str, Dict[str, float], Union[BaseEstimator, ClassifierMixin]]:
        """
        训练单个模型并评估

        Args:
            model_name: 模型名称
            model: 模型实例
            X_train: 训练特征（已缩放）
            y_train: 训练标签（数值）
            X_test: 测试特征（已缩放）
            y_test: 测试标签（数值）
            X_train_raw: 训练特征（原始）
            X_test_raw: 测试特征（原始）
            y_train_raw: 训练标签（原始）
            y_test_raw: 测试标签（原始）

        Returns:
            (模型名称, 评估指标字典, 训练后的模型实例)
        """
        try:
            logger.info(f"开始训练: {model_name}")

            needs_scaling = model_name in MODELS_NEED_SCALING
            needs_int_labels = model_name in MODELS_NEED_INT_LABELS

            use_X_train = X_train if needs_scaling else (X_train_raw if X_train_raw is not None else X_train)
            use_X_test = X_test if needs_scaling else (X_test_raw if X_test_raw is not None else X_test)
            use_y_train = y_train if needs_int_labels else (y_train_raw if y_train_raw is not None else y_train)
            use_y_test = y_test if needs_int_labels else (y_test_raw if y_test_raw is not None else y_test)

            model.fit(use_X_train, use_y_train)

            y_train_pred = model.predict(use_X_train)
            y_test_pred = model.predict(use_X_test)

            y_train_num = np.array(y_train_pred)
            y_test_num = np.array(y_test_pred)

            if hasattr(y_train_num, '__iter__') and not isinstance(y_train_num, (int, float)):
                y_train_num = np.array(y_train_num).flatten()
                y_test_num = np.array(y_test_num).flatten()

            true_y_train = y_train if needs_int_labels else (y_train_raw if y_train_raw is not None else y_train)
            true_y_test = y_test if needs_int_labels else (y_test_raw if y_test_raw is not None else y_test)

            metrics = {
                "训练集准确率": accuracy_score(true_y_train, y_train_num),
                "测试集准确率": accuracy_score(true_y_test, y_test_num),
                "测试集精确率": precision_score(true_y_test, y_test_num, average='weighted', zero_division=0),
                "测试集召回率": recall_score(true_y_test, y_test_num, average='weighted', zero_division=0),
                "测试集F1分数": f1_score(true_y_test, y_test_num, average='weighted', zero_division=0)
            }

            logger.info(f"{model_name} 训练完成 - 测试集准确率: {metrics['测试集准确率']:.4f}")

            return model_name, metrics, model

        except Exception as e:
            logger.error(f"训练 {model_name} 失败: {e}")
            return model_name, {
                "训练集准确率": 0.0,
                "测试集准确率": 0.0,
                "测试集精确率": 0.0,
                "测试集召回率": 0.0,
                "测试集F1分数": 0.0
            }, model

    def train_all_models(self, parallel: bool = True) -> Dict[str, Dict[str, float]]:
        """
        训练所有模型（支持并行）

        Args:
            parallel: 是否使用并行训练

        Returns:
            所有模型的评估指标字典
        """
        logger.info(f"开始训练所有模型，并行模式: {parallel}")

        factory = ModelFactory()
        self.models = factory.get_available_models()

        if parallel and len(self.models) > 1:
            metrics = self._train_parallel()
        else:
            metrics = self._train_sequential()

        self.all_metrics = metrics
        self._select_best_model()

        return metrics

    def _train_sequential(self) -> Dict[str, Dict[str, float]]:
        """
        串行训练所有模型

        Returns:
            评估指标字典
        """
        metrics = {}
        for name, model in self.models.items():
            _, m, trained_model = self._train_single_model(
                name, model,
                self.X_train_scaled, self.y_train_num,
                self.X_test_scaled, self.y_test_num,
                self.X_train, self.X_test,
                self.y_train, self.y_test
            )
            self.models[name] = trained_model
            metrics[name] = m
        return metrics

    def _train_parallel(self) -> Dict[str, Dict[str, float]]:
        """
        并行训练所有模型

        Returns:
            评估指标字典
        """
        metrics = {}

        with ProcessPoolExecutor(max_workers=min(4, len(self.models))) as executor:
            futures = {}
            for name, model in self.models.items():
                future = executor.submit(
                    self._train_single_model,
                    name, model,
                    self.X_train_scaled, self.y_train_num,
                    self.X_test_scaled, self.y_test_num,
                    self.X_train, self.X_test,
                    self.y_train, self.y_test
                )
                futures[future] = name

            for future in as_completed(futures):
                name = futures[future]
                try:
                    _, m, trained_model = future.result()
                    self.models[name] = trained_model
                    metrics[name] = m
                except Exception as e:
                    logger.error(f"并行训练 {name} 失败: {e}")
                    metrics[name] = {
                        "训练集准确率": 0.0,
                        "测试集准确率": 0.0,
                        "测试集精确率": 0.0,
                        "测试集召回率": 0.0,
                        "测试集F1分数": 0.0
                    }

        return metrics

    def _select_best_model(self) -> None:
        """
        选择最佳模型（基于测试集准确率）
        """
        if not self.all_metrics:
            logger.warning("没有可用的评估指标，无法选择最佳模型")
            return

        sorted_models = sorted(
            self.all_metrics.items(),
            key=lambda x: x[1]["测试集准确率"],
            reverse=True
        )

        logger.info("模型排名:")
        for i, (name, metrics) in enumerate(sorted_models, 1):
            accuracy = metrics["测试集准确率"]
            if accuracy < 0.35:
                logger.info(f"  {i}. {name}: 测试集准确率 {accuracy:.4f} (低于阈值，淘汰)")
                del self.models[name]
            else:
                logger.info(f"  {i}. {name}: 测试集准确率 {accuracy:.4f}")

        if self.models:
            self.best_model_name = sorted_models[0][0]
            self.best_model = self.models[self.best_model_name]
            best_metrics = self.all_metrics[self.best_model_name]
            logger.info(f"选择最佳模型: {self.best_model_name}, 测试集准确率: {best_metrics['测试集准确率']:.4f}")
        else:
            logger.warning("所有模型均被淘汰，使用逻辑回归作为默认模型")
            factory = ModelFactory()
            self.best_model = factory.create_model("逻辑回归")
            self.best_model_name = "逻辑回归"

    def evaluate(self) -> Dict[str, float]:
        """
        评估模型（兼容旧接口）

        Returns:
            详细评估指标
        """
        return self.evaluate_best_model()

    def evaluate_best_model(self) -> Dict[str, float]:
        """
        评估最佳模型

        Returns:
            详细评估指标
        """
        if self.best_model is None:
            logger.error("没有可用的最佳模型")
            return {}

        logger.info(f"开始评估最佳模型: {self.best_model_name}")

        y_pred = self.best_model.predict(self.X_test_scaled)
        y_pred = np.array(y_pred).flatten()

        cm = confusion_matrix(self.y_test_num, y_pred)
        cr = classification_report(self.y_test_num, y_pred, target_names=["主队胜", "平局", "客队胜"], output_dict=True, zero_division=0)

        metrics = {
            "测试集准确率": accuracy_score(self.y_test_num, y_pred),
            "测试集精确率": precision_score(self.y_test_num, y_pred, average='weighted', zero_division=0),
            "测试集召回率": recall_score(self.y_test_num, y_pred, average='weighted', zero_division=0),
            "测试集F1分数": f1_score(self.y_test_num, y_pred, average='weighted', zero_division=0),
            "主队胜精确率": cr["主队胜"]["precision"],
            "主队胜召回率": cr["主队胜"]["recall"],
            "主队胜F1": cr["主队胜"]["f1-score"],
            "平局精确率": cr["平局"]["precision"],
            "平局召回率": cr["平局"]["recall"],
            "平局F1": cr["平局"]["f1-score"],
            "客队胜精确率": cr["客队胜"]["precision"],
            "客队胜召回率": cr["客队胜"]["recall"],
            "客队胜F1": cr["客队胜"]["f1-score"],
        }

        logger.info(f"最佳模型 {self.best_model_name} 评估完成")
        logger.info(f"测试集准确率: {metrics['测试集准确率']:.4f}")

        return metrics

    def retrain_with_full_data(self) -> None:
        """
        使用全量数据重新训练最佳模型
        """
        if self.best_model is None:
            logger.error("没有可用的最佳模型")
            return

        logger.info(f"开始用全量数据重新训练最佳模型: {self.best_model_name}")

        X_full = self.feature_matrix[self.feature_columns]
        y_full = self.feature_matrix[self.target_column].map(self.label_mapping)

        for col in self.feature_columns:
            if X_full[col].isnull().any():
                median_val = X_full[col].median()
                if pd.isna(median_val):
                    median_val = 0
                X_full[col] = X_full[col].fillna(median_val)

        X_full_scaled = self.scaler.fit_transform(X_full)
        self.best_model.fit(X_full_scaled, y_full)

        logger.info(f"{self.best_model_name} 全量数据训练完成")

    def tune_hyperparameters(self) -> None:
        """
        对最佳模型进行参数调优
        """
        if self.best_model is None:
            logger.error("没有可用的最佳模型")
            return

        logger.info(f"开始对 {self.best_model_name} 进行参数调优")

        param_grids = {
            "逻辑回归": {
                "C": [0.1, 1, 10],
                "solver": ["lbfgs", "liblinear"]
            },
            "随机森林": {
                "n_estimators": [50, 100, 150],
                "max_depth": [5, 10, 15]
            },
            "梯度提升": {
                "n_estimators": [50, 100, 150],
                "learning_rate": [0.05, 0.1, 0.2]
            },
            "XGBoost": {
                "n_estimators": [50, 100],
                "max_depth": [3, 5],
                "learning_rate": [0.05, 0.1]
            },
            "LightGBM": {
                "n_estimators": [50, 100],
                "max_depth": [3, 5],
                "learning_rate": [0.05, 0.1]
            },
            "CatBoost": {
                "n_estimators": [50, 100],
                "max_depth": [3, 5],
                "learning_rate": [0.05, 0.1]
            }
        }

        params = param_grids.get(self.best_model_name, {})

        if not params:
            logger.info(f"{self.best_model_name} 不支持参数调优")
            return

        try:
            grid_search = GridSearchCV(
                self.best_model,
                params,
                cv=3,
                scoring='accuracy',
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(self.X_train_scaled, self.y_train_num)

            self.best_model = grid_search.best_estimator_
            logger.info(f"{self.best_model_name} 参数调优完成，最佳参数: {grid_search.best_params_}")

        except Exception as e:
            logger.warning(f"{self.best_model_name} 参数调优失败，使用默认参数: {e}")
            self.retrain_with_full_data()

    def save_models(self) -> None:
        """
        保存训练好的模型
        """
        logger.info("开始保存模型")

        os.makedirs(MODEL_DIR, exist_ok=True)

        if self.best_model is not None:
            model_path = MODEL_DIR / f"task3_{self.best_model_name}_model.pkl"
            joblib.dump(self.best_model, model_path)
            logger.info(f"最佳模型已保存到: {model_path}")

        scaler_path = MODEL_DIR / "task3_scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"标准化器已保存到: {scaler_path}")

    def save_metrics(self) -> None:
        """
        保存评估指标
        """
        logger.info("开始保存评估指标")

        os.makedirs(TABLE_DIR, exist_ok=True)

        if self.all_metrics:
            metrics_df = pd.DataFrame(self.all_metrics).T
            metrics_path = TABLE_DIR / "task3_model_comparison.csv"
            metrics_df.to_csv(metrics_path, encoding="utf-8-sig")
            logger.info(f"模型对比指标已保存到: {metrics_path}")

        if self.best_model is not None:
            best_metrics = self.evaluate_best_model()
            best_metrics_df = pd.DataFrame([best_metrics])
            best_metrics_path = TABLE_DIR / f"task3_{self.best_model_name}_metrics.csv"
            best_metrics_df.to_csv(best_metrics_path, index=False, encoding="utf-8-sig")
            logger.info(f"最佳模型指标已保存到: {best_metrics_path}")

    def run(self, parallel: bool = True) -> Dict[str, Dict[str, float]]:
        """
        执行完整的训练流程

        Args:
            parallel: 是否使用并行训练

        Returns:
            所有模型的评估指标字典
        """
        self.split_data()
        self.preprocess()
        self.train_all_models(parallel=parallel)
        self.retrain_with_full_data()
        self.tune_hyperparameters()
        self.save_models()
        self.save_metrics()

        return self.all_metrics
