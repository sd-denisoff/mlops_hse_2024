"""
GUI utils
"""

import uuid

import pandas as pd
import requests
import streamlit as st
from PIL import Image

from gui.static import GUI_STATIC_PATH

API_URL = "http://rest_app:8080"

DATASET_PATH = "datasets"

USERNAME = "admin"
PASSWORD = "admin"


def init_page(
    title: str,
    desc: str,
):
    """
    Initialize new streamlit page with config and base design
    """
    st.set_page_config(
        layout="centered",
        page_title=f"ML Service • {title}",
        page_icon="📈",
    )

    logo = Image.open(GUI_STATIC_PATH / "logo.png")
    st.sidebar.image(logo)
    st.sidebar.markdown(f"# {title}")

    server_healthy = is_server_online()
    if server_healthy:
        st.sidebar.success("Сервер работает", icon="✅")
    else:
        st.sidebar.error("Сервер недоступен", icon="🚨")

    st.markdown(f"# {title}")
    st.write(desc)
    st.divider()

    if not check_auth():
        process_auth()


def check_auth() -> bool:
    """
    Check if user is already authenticated
    """
    return st.session_state.get("jwt_token") is not None


def process_auth():
    """
    Process authentication
    """
    st.info("Пользователь не авторизован", icon="ℹ️")
    login = st.text_input("Логин", placeholder=USERNAME)
    password = st.text_input("Пароль", type="password", placeholder=PASSWORD)
    if login != USERNAME or password != PASSWORD:
        if login and password:
            st.error("Неверные данные", icon="🚨")
        st.stop()
    st.session_state["jwt_token"] = str(uuid.uuid4())
    st.rerun()


def is_server_online() -> bool:
    """
    Server health check
    """
    try:
        response = requests.get(f"{API_URL}/status")
        return response.json().get("status") == "online"
    except requests.exceptions.ConnectionError:
        return False


def read_dataset(
    dataset_name: str,
    data_type: str,
) -> tuple[list[dict[str, float]], list[float]]:
    """
    Read prepared dataset
    """
    data = pd.read_csv(f"{DATASET_PATH}/{dataset_name}/{data_type}_data.csv")
    y = data["target"].to_list()
    X = data.drop(columns="target").to_dict(orient="records")
    return X, y
