# dashboard.py
import streamlit as st
import json
import pandas as pd
import altair as alt
import time

STATUS_JSON = "status.json"
REFRESH_INTERVAL = 2  # segundos

st.set_page_config(page_title="Estacionamento Inteligente", layout="centered")

st.title("🚗 Estacionamento Inteligente com IA")
st.subheader("📡 Status em tempo real")

# Criar container vazio que será atualizado
placeholder = st.empty()

while True:
    with placeholder.container():
        try:
            with open(STATUS_JSON, "r", encoding="utf-8") as f:
                vagas = json.load(f)
        except FileNotFoundError:
            st.warning("⏳ Aguardando o sistema detectar vagas...")
            time.sleep(REFRESH_INTERVAL)
            continue

        # Mostrar lista
        for vaga in vagas:
            cor = "🟢" if vaga["status"] == "Livre" else "🔴"
            st.write(f"{cor} Vaga {vaga['vaga']} - {vaga['status']}")

        # Gráfico
        if vagas:
            df = pd.DataFrame(vagas)
            chart = alt.Chart(df).mark_bar().encode(
                x="status",
                y="count()",
                color="status"
            )
            st.altair_chart(chart, use_container_width=True)

    # Espera antes de atualizar novamente
    time.sleep(REFRESH_INTERVAL)
