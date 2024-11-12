"""
GUI for ML service
"""

# pylint: disable=invalid-name, non-ascii-file-name, unspecified-encoding

import streamlit as st

from gui.static import GUI_STATIC_PATH
from gui.utils import init_page

init_page(title="Главная", desc="ML Service GUI")

with open(GUI_STATIC_PATH / "README.md", encoding="utf8") as f:
    readme = f.read()
    st.markdown(readme, unsafe_allow_html=True)
