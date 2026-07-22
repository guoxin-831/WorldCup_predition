"""
models.py
==========================
Task3 模型定义模块
==========================
功能：
1. 定义所有分类模型的配置参数
2. 提供模型工厂模式创建模型实例
3. 支持模型配置的动态调整
4. 处理可选依赖的优雅降级
==========================
"""

from typing import Dict, List, Optional, Union
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

RANDOM_SEED = 42


class ModelConfig:
    """
    模型配置类
    """

    def __init__(self):
        self.model_params = {
            "逻辑回归": {
                "solver": "lbfgs",
                "max_iter": 1000,
                "class_weight": "balanced",
                "random_state": RANDOM_SEED
            },
            "随机森林": {
                "n_estimators": 150,
                "max_depth": 10,
                "min_samples_split": 4,
                "min_samples_leaf": 1,
                "class_weight": "balanced",
                "random_state": RANDOM_SEED
            },
            "梯度提升": {
                "n_estimators": 100,
                "max_depth": 3,
                "learning_rate": 0.1,
                "min_samples_split": 10,
                "min_samples_leaf": 5,
                "random_state": RANDOM_SEED
            },
            "XGBoost": {
                "n_estimators": 100,
                "max_depth": 3,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 1.0,
                "scale_pos_weight": 1,
                "random_state": RANDOM_SEED,
                "verbosity": 0
            },
            "LightGBM": {
                "n_estimators": 100,
                "max_depth": 3,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 1.0,
                "class_weight": "balanced",
                "random_state": RANDOM_SEED,
                "verbose": -1
            },
            "K近邻": {
                "n_neighbors": 7,
                "weights": "distance"
            },
            "支持向量机": {
                "kernel": "rbf",
                "class_weight": "balanced",
                "random_state": RANDOM_SEED,
                "probability": True
            },
            "CatBoost": {
                "n_estimators": 100,
                "max_depth": 3,
                "learning_rate": 0.1,
                "class_weights": [1, 1, 1],
                "random_state": RANDOM_SEED,
                "verbose": 0
            }
        }

    def get_params(self, model_name: str) -> Dict[str, any]:
        """
        获取模型参数

        Args:
            model_name: 模型名称

        Returns:
            模型参数字典
        """
        return self.model_params.get(model_name, {})

    def update_params(self, model_name: str, params: Dict[str, any]) -> None:
        """
        更新模型参数

        Args:
            model_name: 模型名称
            params: 新的参数字典
        """
        if model_name in self.model_params:
            self.model_params[model_name].update(params)

    def get_all_model_names(self) -> List[str]:
        """
        获取所有模型名称列表

        Returns:
            模型名称列表
        """
        names = ["逻辑回归", "随机森林", "梯度提升", "K近邻", "支持向量机"]
        if HAS_XGBOOST:
            names.append("XGBoost")
        if HAS_LIGHTGBM:
            names.append("LightGBM")
        if HAS_CATBOOST:
            names.append("CatBoost")
        return names


class ModelFactory:
    """
    模型工厂类
    """

    def __init__(self):
        self.config = ModelConfig()

    def create_model(self, model_name: str) -> Optional[Union[BaseEstimator, ClassifierMixin]]:
        """
        创建模型实例

        Args:
            model_name: 模型名称

        Returns:
            模型实例，如果创建失败返回None
        """
        try:
            params = self.config.get_params(model_name)

            if model_name == "逻辑回归":
                return LogisticRegression(**params)

            elif model_name == "随机森林":
                return RandomForestClassifier(**params)

            elif model_name == "梯度提升":
                return GradientBoostingClassifier(**params)

            elif model_name == "XGBoost":
                if not HAS_XGBOOST:
                    return None
                return xgb.XGBClassifier(**params)

            elif model_name == "LightGBM":
                if not HAS_LIGHTGBM:
                    return None
                return lgb.LGBMClassifier(**params)

            elif model_name == "CatBoost":
                if not HAS_CATBOOST:
                    return None
                return CatBoostClassifier(**params)

            elif model_name == "K近邻":
                return KNeighborsClassifier(**params)

            elif model_name == "支持向量机":
                return SVC(**params)

            else:
                return None

        except Exception as e:
            from .utils import setup_logger
            logger = setup_logger()
            logger.error(f"创建模型 {model_name} 失败: {e}")
            return None

    def create_voting_classifier(self, base_models: List[tuple]) -> Optional[VotingClassifier]:
        """
        创建投票集成分类器

        Args:
            base_models: 基础模型列表，格式为 [(name, model), ...]

        Returns:
            VotingClassifier实例
        """
        try:
            return VotingClassifier(
                estimators=base_models,
                voting="soft",
                weights=[1] * len(base_models)
            )
        except Exception as e:
            from .utils import setup_logger
            logger = setup_logger()
            logger.error(f"创建投票集成分类器失败: {e}")
            return None

    def create_stacking_classifier(self, base_models: List[tuple]) -> Optional[StackingClassifier]:
        """
        创建堆叠集成分类器

        Args:
            base_models: 基础模型列表，格式为 [(name, model), ...]

        Returns:
            StackingClassifier实例
        """
        try:
            final_estimator = LogisticRegression(
                solver="lbfgs",
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_SEED
            )
            return StackingClassifier(
                estimators=base_models,
                final_estimator=final_estimator,
                cv=5,
                stack_method="predict_proba"
            )
        except Exception as e:
            from .utils import setup_logger
            logger = setup_logger()
            logger.error(f"创建堆叠集成分类器失败: {e}")
            return None

    def get_available_models(self) -> Dict[str, Union[BaseEstimator, ClassifierMixin]]:
        """
        获取所有可用模型的字典

        Returns:
            模型名称到模型实例的字典
        """
        models = {}
        for name in self.config.get_all_model_names():
            model = self.create_model(name)
            if model is not None:
                models[name] = model
        return models
