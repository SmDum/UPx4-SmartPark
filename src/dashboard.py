# dashboard.py

import streamlit as st  # Importa a biblioteca Streamlit para criar dashboards web interativos
import json             # Importa o módulo json para ler arquivos JSON
import pandas as pd     # Importa o pandas para manipulação de dados em DataFrames
import altair as alt    # Importa o Altair para criação de gráficos
import time             # Importa o módulo time para controlar intervalos de atualização

STATUS_JSON = "status.json"      # Define o nome do arquivo JSON com o status das vagas
REFRESH_INTERVAL = 2             # Define o intervalo de atualização em segundos

st.set_page_config(page_title="Estacionamento Inteligente", layout="centered")  # Configura o título e layout da página

st.title("🚗 Estacionamento Inteligente com IA")      # Define o título principal do dashboard
st.subheader("📡 Status em tempo real")               # Define o subtítulo do dashboard

# Criar container vazio que será atualizado
placeholder = st.empty()     # Cria um espaço vazio na página para atualizar o conteúdo dinamicamente

while True:  # Loop infinito para atualizar o dashboard em tempo real
    with placeholder.container():   # Usa o container do placeholder para atualizar o conteúdo
        try:
            with open(STATUS_JSON, "r", encoding="utf-8") as f:  # Tenta abrir o arquivo JSON com o status das vagas
                vagas = json.load(f)                             # Carrega os dados do arquivo JSON em uma lista de dicionários
        except FileNotFoundError:                                # Caso o arquivo não exista ainda
            st.warning("⏳ Aguardando o sistema detectar vagas...")  # Mostra um aviso na tela
            time.sleep(REFRESH_INTERVAL)                            # Aguarda o intervalo antes de tentar novamente
            continue                                               # Volta para o início do loop

        # Mostrar lista
        for vaga in vagas:                                         # Para cada vaga no arquivo JSON
            cor = "🟢" if vaga["status"] == "Livre" else "🔴"      # Define o emoji de cor conforme o status da vaga
            st.write(f"{cor} Vaga {vaga['vaga']} - {vaga['status']}")  # Exibe o status da vaga na tela

        # Gráfico
        if vagas:                                                  # Se houver vagas no arquivo
            df = pd.DataFrame(vagas)                               # Converte a lista de vagas em um DataFrame do pandas
            chart = alt.Chart(df).mark_bar().encode(               # Cria um gráfico de barras com Altair
                x="status",                                        # Eixo X: status da vaga (Livre/Ocupada)
                y="count()",                                       # Eixo Y: quantidade de vagas por status
                color="status"                                     # Cor das barras conforme o status
            )
            st.altair_chart(chart, use_container_width=True)       # Exibe o gráfico no dashboard

    # Espera antes de atualizar novamente
    time.sleep(REFRESH_INTERVAL)                                   # Aguarda o intervalo antes de atualizar o dashboard novamente