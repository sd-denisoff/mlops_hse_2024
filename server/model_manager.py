import os
import joblib
from pathlib import Path

# логика по подсчету хешей 

class ModelManager:
    def __init__(self, storage_dir):
        """
        Инициализация ModelManager с директорией для хранения моделей.
        :param storage_dir: Путь к директории, где будут храниться модели.
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def create_trainer(self, model_type, model_params=None):
        """
        Создаёт тренера модели на основе указанного типа.
        :param model_type: Тип модели ("boosting" или "linear").
        :param model_params: Параметры для модели.
        :return: Экземпляр тренера.
        """
        if model_type == 'linear':
            return LinRegTrainer(model_params)
        elif model_type == 'boosting':
            return CatBoostTrainer(model_params)
        else:
            raise ValueError("Unsupported model type")

    def train_and_save_model(self, model_type, model_name, X_train, y_train, model_params=None):
        """
        Создаёт, обучает и сохраняет модель.
        :param model_type: Тип модели.
        :param model_name: Имя под которым модель будет сохранена.
        :param X_train: Данные для обучения модели.
        :param y_train: Целевые значения для обучения модели.
        :param model_params: Параметры для модели.
        """
        trainer = self.create_trainer(model_type, model_params)
        trainer.train(X_train, y_train)
        self.save_model(trainer.model, model_name)

    def save_model(self, model, model_name):
        """
        Сохраняет модель в указанной директории.
        :param model: Обученная модель для сохранения.
        :param model_name: Имя файла модели.
        """
        model_path = self.storage_dir / f"{model_name}.joblib"
        joblib.dump(model, model_path)

    def load_model(self, model_name):
        """
        Загружает модель из файла.
        :param model_name: Имя файла модели для загрузки.
        :return: Загруженная модель.
        """
        model_path = self.storage_dir / f"{model_name}.joblib"
        if model_path.exists():
            return joblib.load(model_path)
        else:
            raise FileNotFoundError(f"Модель {model_name} не найдена в {self.storage_dir}.")

    def delete_model(self, model_name):
        """
        Удаляет модель из хранилища.
        :param model_name: Имя файла модели для удаления.
        """
        model_path = self.storage_dir / f"{model_name}.joblib"
        if model_path.exists():
            model_path.unlink()
        else:
            raise FileNotFoundError(f"Модель {model_name} для удаления не найдена.")

    def list_models(self):
        """
        Возвращает список всех моделей, сохраненных в директории.
        :return: Список имен файлов моделей.
        """
        return [file.name for file in self.storage_dir.glob("*.joblib")]
