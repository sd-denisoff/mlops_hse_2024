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

### Инструкция по запуску

#### FastApi

1. Запустите сервер FastApi

```
poetry run python3 server/rest/run.py
``` 

или 

```
poetry run uvicorn server.rest.app:app --port <your_port>
```

2. Запустите streamlit

> TBA 

### Проверка кода:

```bash
cd <project_root_directory>
pylint .
black --extend-exclude='/server/grpc/proto/*' .
 ruff check --exclude='*.ipynb' .
```

### Использование сервиса

> TBA

### Доступный функционал

> TBA
