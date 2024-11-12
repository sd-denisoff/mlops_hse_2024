"""
Managing page
"""

from time import sleep

import requests
import streamlit as st

from gui.utils import init_page, API_URL

init_page(title="Управление", desc="Управление списком моделей")

trained_models = requests.get(f"{API_URL}/trained_models").json()["trained_models"]

if not trained_models:
    st.info("Нет обученных моделей", icon="ℹ️")
    st.stop()

col1, col2 = st.columns(2)

show_models_btn = col1.button(
    label="Отобразить список моделей",
    help="Получение всех обученных моделей",
    use_container_width=True,
)

if show_models_btn:
    with st.spinner("Загрузка списка моделей"):
        sleep(2)
        col2.write(trained_models)

st.divider()

col1, col2 = st.columns(2)

model_id = col1.selectbox(
    label="Модель для удаления",
    options=trained_models,
    help="Выберите обученную модель для удаления",
)

delete_model_btn = col1.button(
    label="Удалить модель",
    help="Получение всех обученных моделей",
    use_container_width=True,
)

if delete_model_btn:
    with st.spinner("Удаление модели"):
        sleep(2)
        response = requests.delete(f"{API_URL}/models/{model_id}")
        if response.ok:
            col2.success(f"Модель {model_id} удалена")
        else:
            col2.error(
                f"Ошибка удаления модели: {response.json().get("detail", "unknown error")}",
                icon="🚨",
            )
