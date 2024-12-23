# MLOps FTDA HSE

> Домашние задания по курсу "Запуск ML моделей в промышленной среде" на ФТиАД ВШЭ

### Состав команды

- Денисов Степан
- Татаринов Артем
- Карлов Владимир

### Описание проекта

Репозиторий содержит сервис, реализующий end-to-end решение для запуска, обучения и использования ML-моделей.

Стек технологий:

<div>
  <img src="https://skillicons.dev/icons?i=py" height="40" alt="python" />
  <img src="https://skillicons.dev/icons?i=fastapi" height="40" alt="fastapi" />
  <img src="https://raw.githubusercontent.com/grpc/grpc.io/refs/heads/main/static/img/grpc.svg" height="40" alt="grpc" />
  <img src="https://cdn.simpleicons.org/streamlit/FF4B4B" height="40" alt="streamlit" />
  <img src="https://skillicons.dev/icons?i=docker" height="40" alt="docker" />
  <img src="https://cdn.simpleicons.org/poetry/60A5FA" height="40" alt="poetry" />
</div>

Ветки:

- `master` – стабильная версия сервиса с оттестированным и полным функционалом
- `dev-hw1` – версия сервиса на момент разработки ДЗ-1
- `dev-hw2` – версия сервиса на момент разработки ДЗ-2
- `dev-hw3` – версия сервиса на момент разработки ДЗ-3

### Доступный функционал

Сервис осуществляет полный цикл работы с ML моделями. Реализован следующий функционал:

- Получение списка классов моделей, доступных для обучения
- Обучение с заданными гиперпараметрами и сохранение модели
- Получение списка обученных моделей
- Получение предсказаний конкретной модели
- Переобучение модели
- Удаление модели
- Повторное обучение модели (подразумевается ее удаление и новый запуск процедуры обучения, в названии моделей
  присутствует хэш от гиперпараметров и данных)
- Проверка статуса работы сервиса

Авторизация на данный момент реализована только на фронтенде.

### Инструкция по запуску

0. Установите python >=3.10, <=3.12 и poetry для управления зависимостями

1. Склонируйте репозиторий и перейдите в директорию с проектом

```bash
git clone https://github.com/sd-denisoff/mlops_hse_2024.git
cd mlops_hse_2024
```

2. Установите зависимости

```bash
poetry install
```

3. Запустите сервер

FastAPI:
```bash
poetry run uvicorn server.rest.app:app --port 8080 --reload
```

gRPC:
```bash
poetry run python3 server/grpc/server.py
```

Адрес Swagger в случае запуска сервера на FastAPI: http://localhost:8000/docs

4. Запустите графический интерфейс

```bash
poetry run streamlit run gui/🏚_Главная.py
```

Адрес дашборда: http://localhost:8501

### Запуск всего через Docker

1. Добавьте файл `.env` с переменными окружения

```
MINIO_ROOT_USER=user
MINIO_ROOT_PASSWORD=password
MINIO_PORT=9000
MINIO_BUCKET=trainer-bucket

MLFLOW_PORT=5001
MLFLOW_DB_PATH=sqlite:///ml_flow.db
MLFLOW_ARTIFACTS_PATH=/mlflow/artifacts
```

2. Запустите сервис

```bash
docker build -f Dockerfile -t trainer-base:latest .
docker-compose up --build
```

3. Запустите MLflow

```bash
docker compose -f mlflow-docker-compose.yml up --build
```

Адрес интерфейса: http://localhost:5001


### Тестирование

Запуск unit-тестов:
```bash
pytest unit_tests
```

Запуск интеграционных тестов (после запуска сервиса с MinIO по [инструкции](#запуск-всего-через-docker)):
```bash
pytest integration_tests
```

### Проверка качества кода

```bash
poetry run pylint .
poetry run black --check --extend-exclude='/server/grpc/*' . 
```
