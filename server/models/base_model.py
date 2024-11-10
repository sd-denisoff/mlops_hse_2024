"""
Abstract model class
"""

import abc
from dataclasses import dataclass
from typing import Literal


@dataclass
class MLModel(abc.ABC):
    """
    Abstract class for defining ML models
    """

    hyperparams: dict

    def save(self, storage: Literal["local", "minio"]) -> tuple[bool, str]:
        """
        Save model to storage

        Parameters
        ------------
            storage: Literal["local", "minio"]
                Final destination for model saving

        Returns
        -----------
            is_saved: bool
                Flag for successful saving
            description: str
                Saving status details
        """
        pass
