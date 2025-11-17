import streamlit as st
import json
import pandas as pd
import time

STATUS_JSON = "status.json"

st.set_page_config(page_title="Estacionamento Real com IA", layout="centered")
st.title("🚗 Estacionamento Inteligente (Vídeo Real)")
st.subheader("Atualização em tempo real")

while True:
    try:
        with open(STATUS_JSON, "r", encoding="utf-8") as f:
            vagas = json.load(f)
    except FileNotFoundError:
        st.warning("Aguardando dados...")
        time.sleep(1)
        continue

    df = pd.DataFrame(vagas)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total", len(df))
    with col2:
        st.metric("Ocupadas", len(df[df["status"] == "Ocupada"]))

    st.bar_chart(df["status"].value_counts())
    time.sleep(2)
    st.rerun()
