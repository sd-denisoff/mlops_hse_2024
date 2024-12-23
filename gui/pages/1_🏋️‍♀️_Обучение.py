"""
Training page
"""

import os
from time import sleep

import requests
import streamlit as st

from gui.utils import init_page, API_URL, read_dataset

init_page(title="Обучение модели", desc="Обучение модели с заданными гиперпараметрами")

available_models = requests.get(f"{API_URL}/models").json()

col1, col2 = st.columns(2)

model_to_train = col1.selectbox(
    label="Модель для обучения",
    options=list(available_models.keys()),
    help="Выберите желаемую архитектуру модели",
)
hyperparameters = available_models[model_to_train]

col2.write("Список доступных гиперпараметров:")
col2.write(hyperparameters)

with st.expander("Настройка гиперпараметров", expanded=True):
    defined_hyperparameters = {
        param: st.text_input(label=param, help=f"Значение гиперпараметра {param}")
        for param in hyperparameters
    }

cleaned_hyperparameters = {
    param: value for param, value in defined_hyperparameters.items() if value
}

dataset_name = st.selectbox(
    label="Данные для обучения",
    options=list(os.listdir("datasets")),
    help="Выберите датасет для обучения модели",
)

X_train, y_train = read_dataset(dataset_name=dataset_name, data_type="train")

if st.button(
    label="Обучить",
    help="Запуск процесса обучения и сохранения модели",
    type="primary",
    use_container_width=True,
):
    with st.spinner("Обучение модели"):
        sleep(5)
        response = requests.post(
            f"{API_URL}/train",
            json={
                "model_spec": {
                    "type": model_to_train,
                    "parameters": cleaned_hyperparameters,
                },
                "features": X_train,
                "targets": y_train,
            },
        )

    if response.ok:
        st.success(
            "ID обученной модели: {}".format(response.json().get("model_id")), icon="✅"
        )
    else:
        st.error(
            "Ошибка обучения модели: {}".format(
                response.json().get("detail", "unknown error")
            ),
            icon="🚨",
        )
