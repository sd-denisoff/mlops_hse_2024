"""
Abstract model class
"""

import abc
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Literal, Union
import joblib

@dataclass
class MLModel(abc.ABC):
    """
    Abstract class for defining ML models
    """

    hyperparams: dict
    @abc.abstractclassmethod
    def fit(
        self,
        X: Union[pd.DataFrame, np.array],
        y: Union[pd.Series, np.array]
    ): pass

    @classmethod
    @abc.abstractmethod
    def _get_param_names(cls): pass

    @abc.abstractclassmethod
    def predict(self, X: Union[pd.DataFrame, np.array]): pass
