import streamlit as st
import cv2
import json

# Carregar dados das vagas (mesmo arquivo do main.py)
with open("slots.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    vagas = data["slots"]

# Simulação de ocupação (por enquanto, só demonstrativo)
# Depois iremos atualizar isso em tempo real
status_vagas = {i: "Livre" for i in range(len(vagas))}

# Configuração do layout
st.set_page_config(page_title="Estacionamento Inteligente", layout="centered")

st.title("🚗 Estacionamento Inteligente com IA")
st.subheader("Mapa das Vagas")

# Mostrar status de cada vaga
for i, vaga in enumerate(vagas):
    st.write(f"Vaga {i+1}: **{status_vagas[i]}**")

