"""
Abstract model class
"""

import abc
from typing import NewType

import numpy as np
import pandas as pd

DataType = NewType("DataType", pd.DataFrame | np.array)
TargetType = NewType("TargetType", pd.Series | np.array)


class MLModel(abc.ABC):
    """
    Abstract class for defining ML models
    """

    model_class = None
    hyperparams: dict = {}

    def __init__(self, hyperparams: dict = None):
        hyperparams = hyperparams or {}

        missing_params = (
            set(hyperparams.keys()) - set(self.__class__._get_param_names())
        )
        assert len(missing_params) == 0, f"hyperparams {missing_params} are not passed"

        self.hyperparams = hyperparams
        self.model = self.model_class(**self.hyperparams)

    @abc.abstractmethod
    def fit(self, X: DataType, y: TargetType):
        """
        Fit the model
        :param X: train objects
        :param y: targets
        :return: None
        """
        pass

    @abc.abstractmethod
    def predict(self, X: DataType) -> TargetType:
        """
        Get predictions for X
        :param X: test objects
        :return: predictions
        """
        pass

    @classmethod
    @abc.abstractmethod
    def get_param_names(cls) -> list[str]:
        """
        Get whole list of hyperparameters
        :return: parameters list
        """
        pass
