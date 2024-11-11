"""
Linear regression model
"""

from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor
from models.ml_models.base_model import MLModel, DataType, TargetType


class LinRegModel(MLModel):
    """
    Work with LinearRegression estimator
    """

    model_class = LinearRegression

    def fit(self, X: DataType, y: TargetType):
        self.model.fit(X, y)

    def predict(self, X: DataType) -> TargetType:
        return self.model.predict(X)

    @classmethod
    def get_param_names(cls) -> list[str]:
        return cls.model_class._get_param_names()  # pylint: disable=protected-access
    

class CatBoostRegModel(MLModel):
    """
    Work with LinearRegression estimator
    """

    model_class = CatBoostRegressor

    def fit(self, X: DataType, y: TargetType):
        self.model.fit(X, y)

    def predict(self, X: DataType) -> TargetType:
        return self.model.predict(X)

    @classmethod
    def get_param_names(cls) -> list[str]:
        return ['learning_rate', 'iterations', 'depth', 'l2_leaf_reg',
                'border_count', 'random_strength', 'bagging_temperature',
                'oddpriority_threshold', 'feature_border_type', 'task_type']
