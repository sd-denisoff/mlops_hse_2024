# Базовый докер, который будем использовать для всех сервисов, чтобы избежать дублирования кода

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV POETRY_VERSION=1.4.0
ENV PATH="/root/.local/bin:$PATH"

# Устанавливаем системные зависимости и Poetry
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 - --version $POETRY_VERSION

# Копируем файлы
COPY . .

## для @astatarinov - проблемы с установкой через curl на рабочем маке
# RUN pip install poetry

# Настраиваем Poetry и устанавливаем зависимости
RUN poetry config virtualenvs.create false && poetry install --no-interaction

# CMD ["sh"]