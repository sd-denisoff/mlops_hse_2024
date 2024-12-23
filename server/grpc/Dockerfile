# syntax=docker/dockerfile:1

FROM trainer-base:latest

# Копируем код gRPC сервиса
COPY . .

# Открываем порт для gRPC
EXPOSE 50051

# Команда для запуска сервера
CMD ["python", "server.py"]
