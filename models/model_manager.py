"""
Manager for models
"""

import hashlib
from pathlib import Path

import joblib
import pandas as pd

from models.ml_models.base_model import MLModel, DataType, TargetType
from models.ml_models.ml_models import LinRegModel, CatBoostRegModel


class ModelManager:
    """
    Managing models: training, saving and listing
    """

    model_classes = [LinRegModel, CatBoostRegModel]

    def __init__(self, storage_dir: str, hash_len: int = 16):
        """
        Инициализация ModelManager с директорией для хранения моделей.
        :param storage_dir: Путь к директории, где будут храниться модели.
        :param hash_len: Длина хэша для формирования имени модели
        """
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._hash_len = hash_len
        self._available_models = {model.__name__: model for model in self.model_classes}

    def create_trainer(
        self,
        model_type: str,
        model_params: dict = None,
    ) -> MLModel:
        """
        Создаёт тренера модели на основе указанного типа.
        :param model_type: Тип модели.
        :param model_params: Параметры для модели.
        :return: Экземпляр тренера.
        """
        if model_type not in self._available_models:
            raise ValueError(f"Unsupported model type '{model_type}'")
        return self._available_models[model_type](hyperparams=model_params or {})

    def _hash_string(self, s: str) -> str:
        """
        Генерирует хэш фиксированной длины для строки
        :param s: Идентификатор модели
        :return: Хэш длины self.hash_len
        """
        return hashlib.sha256(s.encode()).hexdigest()[: self._hash_len]

    def _generate_model_name(
        self,
        model_type: str,
        model_params: dict,
        train_data: pd.DataFrame,
    ) -> str:
        """
        Генерирует хэш фиксированной длины для строки
        :param model_type: Тип модели.
        :param model_params: Параметры для модели.
        :param train_data: Данные для обучения.
        :return: Хэш длины self.hash_len
        """
        # Генерация хэша для параметров
        params_string = "".join([str(param) for param in sorted(model_params.items())])
        params_hash = self._hash_string(params_string)

        # Генерация хэша для DataFrame
        data_string = train_data.to_csv(index=False)
        data_hash = self._hash_string(data_string)

        # Генерация уникального ID
        unique_id = f"{model_type}_{params_hash}_{data_hash}"

        return unique_id

    def train_and_save_model(
        self,
        model_type: str,
        X_train: DataType,
        y_train: TargetType,
        model_params: dict = None,
    ) -> str:
        """
        Создаёт, обучает и сохраняет модель.
        :param model_type: Тип модели.
        :param X_train: Данные для обучения модели.
        :param y_train: Целевые значения для обучения модели.
        :param model_params: Параметры для модели.
        :return: ID модели
        """
        trainer = self.create_trainer(model_type, model_params)
        trainer.fit(X_train, y_train)

        merged_data = X_train.copy()
        merged_data["target"] = list(y_train)

        model_name = self._generate_model_name(model_type, model_params, merged_data)
        self.save_model(trainer, model_name)

        return model_name

    def save_model(self, model: MLModel, model_name: str):
        """
        Сохраняет модель в указанной директории.
        :param model: Обученная модель для сохранения.
        :param model_name: Имя файла модели.
        """
        model_path = self._storage_dir / f"{model_name}.joblib"
        joblib.dump(model, model_path)

    def load_model(self, model_name: str) -> MLModel:
        """
        Загружает модель из файла.
        :param model_name: Имя файла модели для загрузки.
        :return: Загруженная модель.
        """
        model_path = self._storage_dir / f"{model_name}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model for loading {model_name} not found.")
        return joblib.load(model_path)

    def delete_model(self, model_name: str):
        """
        Удаляет модель из хранилища.
        :param model_name: Имя файла модели для удаления.
        """
        model_path = self._storage_dir / f"{model_name}.joblib"
        if model_path.exists():
            model_path.unlink()
        else:
            raise FileNotFoundError(f"Model for deleting {model_name} not found.")

    def list_models(self) -> list[str]:
        """
        Возвращает список всех моделей, сохраненных в директории.
        :return: Список имен файлов моделей.
        """
        return [Path(file).stem for file in self._storage_dir.glob("*.joblib")]

    def predict(self, model_name: str, X: DataType) -> TargetType:
        """
        Делает предсказание для обученной модели
        :param model_name: Имя файла модели для загрузки.
        :param X: Данные для предсказания модели
        :return: Предсказания
        """
        model = self.load_model(model_name)
        return model.predict(X)


MODEL_MANAGER = ModelManager("./models_storage")
