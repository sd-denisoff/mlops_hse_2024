from .base_model import MLModel
from sklearn.linear_model import LinearRegression
import pandas as pd
from typing import Union
import numpy as np

class LinRegModel(MLModel):
    def __init__(self, hyperparams={}):
        self.model_class = LinearRegression

        missing_params = (
            set(hyperparams.keys())
            .difference(set(self.model_class._get_param_names()))
        )
        assert len(missing_params) == 0, \
        f"hyperparams {missing_params} are not available"

        self.hyperparams = hyperparams
        

    def fit(self, X: Union[pd.DataFrame, np.array], y: Union[pd.Series, np.array]):
        self.model = self.model_class(**self.hyperparams)
        self.model.fit(X, y)

    @staticmethod
    def _get_param_names():
        return LinearRegression._get_param_names()
