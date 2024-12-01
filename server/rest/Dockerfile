# syntax=docker/dockerfile:1

FROM trainer-base:latest

# Копируем код приложения
COPY . .

# Открываем порт для FastAPI
EXPOSE 8080

# Команда для запуска приложения
CMD ["python", "run.py"]
