import os
import joblib
from pathlib import Path
from .ml_models.ml_models import LinRegModel
import hashlib

# логика по подсчету хешей 

class ModelManager:
    model_classes = {
        "linreg": LinRegModel,
    }
    def __init__(self, storage_dir, hash_len=16):
        """
        Инициализация ModelManager с директорией для хранения моделей.
        :param storage_dir: Путь к директории, где будут храниться модели.
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.hash_len = hash_len

    def create_trainer(self, model_type, model_params=None):
        """
        Создаёт тренера модели на основе указанного типа.
        :param model_type: Тип модели ("boosting" или "linear").
        :param model_params: Параметры для модели.
        :return: Экземпляр тренера.
        """
        if model_type in self.model_classes:
            return self.model_classes[model_type](hyperparams=model_params)
        else:
            raise ValueError(f"Unsupported model type '{model_type}'")
        
    def hash_string(self, s):
        """
        Генерирует хэш фиксированной длины для строки
        :param s: (str)
        :return: Хэш длины self.hash_len
        """
        return hashlib.sha256(s.encode()).hexdigest()[:self.hash_len]
    
    def generate_model_name(self, model_class, params_dict, data):
        
        #TO DO: assert data == pd.DataFrame (?)

        # Генерация хэша для параметров
        params_string = ''.join([str(param) for param in sorted(params_dict.items())])
        params_hash = self.hash_string(params_string)

        # Генерация хэша для DataFrame
        data_string = data.to_csv(index=False)  # Преобразуем DataFrame в строку CSV
        data_hash = self.hash_string(data_string)

        # Генерация уникального ID
        unique_id = f"{model_class}_{params_hash}_{data_hash}"

        return unique_id

    def train_and_save_model(self, model_type, X_train, y_train, model_params={}):
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

        # TO DO: склеить X_train и y_train для подачи в generate_model_name
        model_name = self.generate_model_name(model_type, model_params, X_train)

        self.save_model(trainer, model_name)

        return model_name

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
    
    def predict(self, model_name, X):
        """
        Делает предсказание для обученной модели
        :param model_name: Имя файла модели для загрузки.
        :param X: Данные для предсказания модели
        :return: Загруженная модель.
        """

        model = self.load_model(model_name)
        return model.predict(X)


