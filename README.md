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

> TBA

### Инструкция по запуску

0. Склонируйте репозиторий и перейдите в директорию с проектом

```bash
git clone https://github.com/sd-denisoff/mlops_hse_2024.git
cd mlops_hse_2024
```

1. Установите зависимости

```bash
poetry install
```

2. Запустите сервер

FastAPI:

```bash
poetry run uvicorn server.rest.app:app --port 8080 --reload
```

gRPC:

> TBA

Адрес Swagger в случае запуска сервера на FastAPI: http://localhost:8000/docs

3. Запустите графический интерфейс

```bash
poetry run streamlit run dashboard.py
```

Адрес дашборда: http://localhost:8501

### Использование сервиса

> TBA

### Проверка качества кода

```bash
pylint .
black --extend-exclude='/server/grpc/proto/*' .
ruff check --exclude='*.ipynb' .
```
