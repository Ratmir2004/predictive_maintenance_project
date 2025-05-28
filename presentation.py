import streamlit as st

def run():
    st.title("🎯 Презентация проекта")
    slide = st.slider("Слайд", 1, 6, 1)

    if slide == 1:
        st.header("📌 Цель проекта")
        st.markdown("- Разработка модели для предиктивного обслуживания")
        st.markdown("- Визуальный интерфейс на Streamlit")

    elif slide == 2:
        st.header("📊 Исходные данные")
        st.markdown("- Данные: температура, износ, обороты и т.д.")
        st.markdown("- Целевая переменная: Machine failure")

    elif slide == 3:
        st.header("🔍 Обработка и подготовка")
        st.markdown("- Удалены лишние признаки")
        st.markdown("- Кодировка категорий")
        st.markdown("- Масштабирование признаков")

    elif slide == 4:
        st.header("🧠 Обучение модели")
        st.markdown("- RandomForestClassifier")
        st.markdown("- train_test_split")
        st.markdown("- Accuracy и ROC AUC")

    elif slide == 5:
        st.header("✅ Результаты")
        st.markdown("- Accuracy ≈ 0.95")
        st.markdown("- ROC AUC ≈ 0.98")
        st.markdown("- Успешное распознавание отказов")

    elif slide == 6:
        st.header("📌 Выводы и перспективы")
        st.markdown("- Приложение готово к демонстрации")
        st.markdown("- Можно расширить функционал: XGBoost, Telegram-бот, API")