import streamlit as st
import importlib

PAGES = {
    "Анализ и модель": "analysis_and_model",
    "Презентация": "presentation"
}

st.sidebar.title("Навигация")
selection = st.sidebar.radio("Выберите страницу:", list(PAGES.keys()))
page = importlib.import_module(PAGES[selection])
page.run()