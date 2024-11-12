"""
Predictions page
"""

import os
from time import sleep

import requests
import streamlit as st
from sklearn.metrics import mean_squared_error

from gui.utils import init_page, API_URL, read_dataset

init_page(title="Предсказания", desc="Получение предсказаний выбранной модели")

trained_models = requests.get(f"{API_URL}/trained_models").json()["trained_models"]

if not trained_models:
    st.info("Нет обученных моделей", icon="ℹ️")
    st.stop()

col1, col2 = st.columns(2)

model_id = col1.selectbox(
    label="Обученная модель",
    options=trained_models,
    help="Выберите обученную модель",
)

dataset_name = col2.selectbox(
    label="Данные для предсказания",
    options=list(os.listdir("datasets")),
    help="Выберите датасет для получения предсказаний",
)

X_test, y_test = read_dataset(dataset_name=dataset_name, data_type="test")

if st.button(
    label="Получить предсказания",
    help="Запуск процесса получения предсказаний",
    type="primary",
    use_container_width=True,
):
    with st.spinner("Получение предсказаний"):
        sleep(2)
        response = requests.post(
            f"{API_URL}/predict",
            json={
                "model_id": model_id,
                "features": X_test,
            },
        )

    if response.ok:
        predictions = response.json().get("predictions")
        st.info(
            f"Предсказания: {list(map(lambda x: round(x, 2), predictions))}", icon="ℹ️"
        )
        st.info(f"MSE: {mean_squared_error(y_test, predictions):.2f}", icon="📈")
    else:
        st.error(
            f"Ошибка получения предсказаний: {response.json().get("detail", "unknown error")}",
            icon="🚨",
        )
