import streamlit as st
import cv2
import pickle

# Carregar dados das vagas
with open("vagas.pkl", "rb") as f:
    vagas = pickle.load(f)

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

