import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import os

def run():
    st.title("Анализ и обучение модели")

    default_path = os.path.join("data", "predictive_maintenance.csv")
    file = st.file_uploader("Загрузите CSV-файл", type="csv")

    if file:
        df = pd.read_csv(file)
    else:
        st.info("Загружается файл по умолчанию.")
        df = pd.read_csv(default_path)

    st.write("Данные:")
    st.dataframe(df.head())

    if "Machine failure" not in df.columns:
        st.error("В таблице должен быть столбец 'Machine failure'")
        return

    drop_cols = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    df = df.drop(columns=drop_cols, errors='ignore')
    if 'Type' in df.columns:
        df['Type'] = LabelEncoder().fit_transform(df['Type'])

    X = df.drop("Machine failure", axis=1)
    y = df["Machine failure"]

    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if len(model.classes_) > 1 else [0]*len(y_test)

    st.subheader("Метрики модели:")
    st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.3f}")
    try:
        st.write(f"**ROC AUC:** {roc_auc_score(y_test, y_proba):.3f}")
    except:
        st.write("**ROC AUC:** Недоступен (один класс в выборке)")

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)