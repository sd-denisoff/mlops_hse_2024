"""  
Manager for models  
"""

# pylint: disable=too-many-arguments, too-many-positional-arguments, redefined-outer-name, logging-fstring-interpolation

import hashlib
import logging
import os
import subprocess
import uuid
from pathlib import Path

import boto3
import joblib
import mlflow
import pandas as pd
from botocore.exceptions import NoCredentialsError
from mlflow.models import infer_signature
from sklearn.metrics import mean_squared_error as mse

from models.ml_models.base_model import MLModel, DataType, TargetType
from models.ml_models.ml_models import LinRegModel, CatBoostRegModel

LOGGER = logging.getLogger(__name__)


class ModelManager:
    """
    Managing models: training, saving and listing
    """

    model_classes = [LinRegModel, CatBoostRegModel]

    def __init__(
        self,
        storage_dir: str,
        mlflow_tracking_uri: str,
        minio_endpoint: str,
        access_key_id: str,
        secret_access_key: str,
        minio_bucket: str,
        data_dir: str,
        hash_len: int = 16,
    ):
        """
        Инициализация ModelManager с директорией для хранения моделей и MLflow.
        :param storage_dir: Путь к директории, где будут храниться модели.
        :param mlflow_tracking_uri: URI для MLflow tracking server.
        :param hash_len: Длина хэша для формирования имени модели
        """
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._hash_len = hash_len
        self._available_models = {model.__name__: model for model in self.model_classes}
        self._mlflow_tracking_uri = mlflow_tracking_uri
        mlflow.set_tracking_uri(self._mlflow_tracking_uri)
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=minio_endpoint,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
        )
        self.minio_bucket = minio_bucket

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
        with mlflow.start_run():
            trainer = self.create_trainer(model_type, model_params)
            trainer.fit(X_train, y_train)

            LOGGER.info(f"Model {model_type} trained")

            merged_data = X_train.copy()
            merged_data["target"] = list(y_train)

            model_name = self._generate_model_name(
                model_type, model_params, merged_data
            )
            self.save_model(trainer, model_name)
            target_pred = trainer.predict(X_train)
            # Логирование параметров, метрик и модели в MLflow
            mlflow.log_params(model_params)
            mlflow.log_metrics({"train_loss": mse(y_train, target_pred)})
            signature = infer_signature(X_train, target_pred)
            mlflow.sklearn.log_model(trainer, model_name, signature=signature)

            LOGGER.info(f"Model {model_type} saved with name: {model_name}")

        df_name = self.save_data_to_s3(merged_data)
        LOGGER.info(f"Data for {model_name} saved saved as {df_name} to s3")

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
        LOGGER.info(f"Loading model {model_name}")
        return joblib.load(model_path)

    def delete_model(self, model_name: str):
        """
        Удаляет модель из хранилища.
        :param model_name: Имя файла модели для удаления.
        """
        model_path = self._storage_dir / f"{model_name}.joblib"
        if model_path.exists():
            LOGGER.info(f"Deleting model {model_name}")
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
        LOGGER.info(f"Getting predictions for model {model_name}")
        return model.predict(X)

    def save_data_to_s3(self, dataframe: pd.DataFrame):
        """
        Сохраняет полученный pd DataFrame в Minio
        """
        df_name = str(uuid.uuid4()) + ".csv"
        df_path = os.path.join(self._data_dir, df_name)
        dataframe.to_csv(df_path, index=False)

        try:
            self.s3_client.upload_file(df_path, self.minio_bucket, df_name)
        except FileNotFoundError:
            print(f"The file {df_path} was not found")
        except NoCredentialsError:
            print("Credentials not available")

        try:
            subprocess.run(["dvc", "add", df_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error adding file to DVC: {e}")

        try:
            subprocess.run(["dvc", "push"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error pushing files to DVC: {e}")

        return df_name


minio_endpoint = os.environ.get("MINIO_ENDPOINT", "http://minio:9000")
access_key_id = os.environ.get("MINIO_ROOT_USER", "user")
secret_access_key = os.environ.get("MINIO_ROOT_PASSWORD", "password")
minio_bucket = os.environ.get("MINIO_BUCKET", "trainer-bucket")

MODEL_MANAGER = ModelManager(
    "./models_storage",
    "http://mlflow:5001",
    minio_endpoint,
    access_key_id,
    secret_access_key,
    minio_bucket,
    "./data_tmp",
)
