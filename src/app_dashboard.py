import streamlit as st
import pandas as pd
import plotly.express as px


if __name__ == '__main__':
    st.title("ML Pipeline Dashboard")

    # Carregando métricas exemplo (substitua pelo tracking real do MLflow)
    df = pd.DataFrame({"accuracy": [0.88, 0.91], "f1_score": [0.84, 0.89]})
    fig = px.scatter(df, x="accuracy", y="f1_score", title="Performance dos Modelos")
    st.plotly_chart(fig)
