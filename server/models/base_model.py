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

    @abc.abstractclassmethod
    def predict(self, X: Union[pd.DataFrame, np.array]): pass

    # @abc.abstractclassmethod
    # def save(self, storage: Literal["local", "minio"]) -> tuple[bool, str]:
    #     """
    #     Save model to storage

    #     Parameters
    #     ------------
    #         storage: Literal["local", "minio"]
    #             Final destination for model saving

    #     Returns
    #     -----------
    #         is_saved: bool
    #             Flag for successful saving
    #         description: str
    #             Saving status details
    #     """
    #     pass
    
    
    def save_model(self, path):
        """
        Сохранение модели в файл.

        :param path: Путь к файлу, где будет сохранена модель.
        """
        joblib.dump(self.model, path)
