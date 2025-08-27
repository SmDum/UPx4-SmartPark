# main.py
import cv2  # Biblioteca para trabalhar com imagens e vídeos
import json  # Biblioteca para ler e salvar arquivos em formato JSON
import numpy as np  # Biblioteca para cálculos e matrizes
from ultralytics import YOLO  # Biblioteca para detectar carros usando inteligência artificial

PARKING_SPOTS_FILE = "slots.json"   # Arquivo com as vagas desenhadas
PARKING_STATUS_FILE = "status.json" # Arquivo onde será salvo o status das vagas (livre/ocupada)
CAMERA_SOURCE = 0                   # 0 para webcam, ou coloque o caminho de um vídeo
MIN_CONFIDENCE = 0.60               # Só considera o carro se a IA estiver pelo menos 60% confiante
CAR_CLASSES = {"car", "truck"}      # Só conta carros e caminhões

# Função para carregar as vagas do arquivo
def carregar_vagas(caminho):
    with open(caminho, "r", encoding="utf-8") as arquivo:
        dados = json.load(arquivo)
    return dados["slots"]

# Função para saber se um ponto está dentro de uma vaga desenhada
def ponto_dentro_vaga(ponto, pontos_vaga):
    x, y = ponto
    dentro = False
    n = len(pontos_vaga)
    for i in range(n):
        x1, y1 = pontos_vaga[i]
        x2, y2 = pontos_vaga[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1):
            dentro = not dentro
    return dentro

# Função para saber quanto a caixa do carro cobre a vaga
def quanto_cobre_vaga(caixa, pontos_vaga, tamanho_imagem):
    x1, y1, x2, y2 = [int(v) for v in caixa]
    largura, altura = tamanho_imagem
    x1 = max(0, min(largura - 1, x1)); x2 = max(0, min(largura - 1, x2))
    y1 = max(0, min(altura - 1, y1)); y2 = max(0, min(altura - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return 0.0

    mascara_vaga = np.zeros((altura, largura), dtype=np.uint8)
    cv2.fillPoly(mascara_vaga, [np.array(pontos_vaga, dtype=np.int32)], 1)

    mascara_caixa = np.zeros_like(mascara_vaga)
    cv2.rectangle(mascara_caixa, (x1, y1), (x2, y2), 1, thickness=-1)

    intersecao = np.logical_and(mascara_vaga, mascara_caixa).sum()
    area_vaga = mascara_vaga.sum() + 1e-9
    return float(intersecao) / float(area_vaga)

# Função para desenhar as vagas e mostrar se estão livres ou ocupadas
def desenhar_vagas(imagem, vagas, vagas_ocupadas):
    for vaga in vagas:
        id_vaga = vaga["id"]
        pontos = np.array(vaga["points"], dtype=np.int32)
        cor = (0, 0, 255) if id_vaga in vagas_ocupadas else (0, 200, 70)  # Vermelho se ocupada, verde se livre
        cv2.polylines(imagem, [pontos], True, cor, 2)
        cx, cy = pontos.mean(axis=0).astype(int)
        texto = f"Vaga {id_vaga} - {'OCUPADA' if id_vaga in vagas_ocupadas else 'LIVRE'}"
        cv2.putText(imagem, texto, (cx - 40, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2, cv2.LINE_AA)
    return imagem

# Função principal do programa
def principal():
    modelo = YOLO("yolov8n.pt")  # Carrega o modelo de IA para detectar carros
    vagas = carregar_vagas(PARKING_SPOTS_FILE)  # Carrega as vagas desenhadas

    # Abre a câmera ou vídeo
    fonte = CAMERA_SOURCE if isinstance(CAMERA_SOURCE, int) or str(CAMERA_SOURCE).isdigit() else CAMERA_SOURCE
    if isinstance(fonte, str) and fonte.isdigit():
        fonte = int(fonte)
    camera = cv2.VideoCapture(fonte)
    if not camera.isOpened():
        raise RuntimeError(f"Não foi possível abrir a câmera ou vídeo: {CAMERA_SOURCE}")

    cv2.namedWindow("Estacionamento Inteligente", cv2.WINDOW_NORMAL)  # Cria a janela para mostrar o vídeo

    while True:
        ok, imagem = camera.read()  # Lê uma imagem da câmera
        if not ok:
            break
        altura, largura = imagem.shape[:2]

        resultados = modelo(imagem, stream=True, imgsz=640, conf=MIN_CONFIDENCE)  # Detecta carros na imagem

        carros_detectados = []
        for resultado in resultados:
            nomes = resultado.names
            for caixa in resultado.boxes:
                confianca = float(caixa.conf[0])
                classe = int(caixa.cls[0])
                nome = nomes.get(classe, str(classe))
                if nome in CAR_CLASSES and confianca >= MIN_CONFIDENCE:
                    x1, y1, x2, y2 = map(int, caixa.xyxy[0])
                    carros_detectados.append((x1, y1, x2, y2, nome, confianca))

        vagas_ocupadas = set()
        status_vagas = []
        for vaga in vagas:
            id_vaga = vaga["id"]
            pontos = vaga["points"]
            ocupada = False
            for (x1, y1, x2, y2, nome, confianca) in carros_detectados:
                centro_x = (x1 + x2) // 2
                centro_y = (y1 + y2) // 2
                centro_dentro = ponto_dentro_vaga((centro_x, centro_y), pontos)
                sobreposicao = quanto_cobre_vaga((x1, y1, x2, y2), pontos, (largura, altura))
                if centro_dentro or sobreposicao > 0.15:
                    ocupada = True
                    break
            if ocupada:
                vagas_ocupadas.add(id_vaga)

            status_vagas.append({
                "vaga": id_vaga,
                "status": "Ocupada" if ocupada else "Livre"
            })

        # Salva o status das vagas em um arquivo
        with open(PARKING_STATUS_FILE, "w", encoding="utf-8") as arquivo:
            json.dump(status_vagas, arquivo, ensure_ascii=False, indent=2)

        imagem = desenhar_vagas(imagem, vagas, vagas_ocupadas)  # Desenha as vagas na imagem
        for (x1, y1, x2, y2, nome, confianca) in carros_detectados:
            cv2.rectangle(imagem, (x1, y1), (x2, y2), (220, 220, 220), 1)
            cv2.putText(imagem, f"{nome} {confianca:.2f}", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

        total_vagas = len(vagas)
        ocupadas = len(vagas_ocupadas)
        livres = total_vagas - ocupadas
        cv2.rectangle(imagem, (10, 10), (280, 80), (30, 30, 30), -1)
        cv2.putText(imagem, f"Total: {total_vagas}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(imagem, f"Ocupadas: {ocupadas}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 200, 50) if ocupadas==0 else (50, 50, 200), 2)
        cv2.putText(imagem, f"Livres: {livres}", (150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 200, 50), 2)

        cv2.imshow("Estacionamento Inteligente", imagem)  # Mostra a imagem na tela
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):  # Sai se apertar 'q' ou ESC
            break

    camera.release()  # Fecha a câmera
    cv2.destroyAllWindows()  # Fecha a janela

if __name__ == "__main__":
    principal()  # Roda