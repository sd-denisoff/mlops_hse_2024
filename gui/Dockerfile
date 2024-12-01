# syntax=docker/dockerfile:1

FROM trainer-base:latest

# Копируем код Streamlit приложения
COPY . .

# Открываем порт для Streamlit
EXPOSE 8501

# Команда для запуска приложения
CMD ["streamlit", "run", "🏚_Главная.py", "--server.port=8501", "--server.address=0.0.0.0"]
