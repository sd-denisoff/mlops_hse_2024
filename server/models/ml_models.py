from .base_model import MLModel
from sklearn.linear_model import LinearRegression
import pandas as pd

class LinRegModel(MLModel):
    def __init__(self, hyperparams={}):
        self.hyperparams = hyperparams

    def fit(self, X: pd.DataFrame, y: pd.Series): pass

    def predict(self, X: pd.DataFrame, y: pd.Series): pass

    def save(self, X: pd.DataFrame, y: pd.Series): pass
