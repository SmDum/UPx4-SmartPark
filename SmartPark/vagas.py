import cv2                  # Importa a biblioteca OpenCV para processamento de imagens e vídeo
import supervision as sv    # Importa a biblioteca supervision (não está sendo usada neste código)
from ultralytics import YOLO # Importa a classe YOLO da biblioteca ultralytics para detecção de objetos
import numpy as np          # Importa a biblioteca NumPy para manipulação de arrays numéricos

model = YOLO("yolov8n.pt")  # Carrega o modelo YOLO pré-treinado (versão nano) para detecção de objetos

# Define as vagas de estacionamento como polígonos (lista de pontos)
parking_spots = {
   "slots": [
    {
      "id": 1,
      "points": [
        [587, 430], [586, 339], [474, 340], [470, 432], [582, 430]
      ]
    },
    {
      "id": 2,
      "points": [
        [586, 320], [588, 223], [471, 230], [470, 323], [587, 323]
      ]
    },
    {
      "id": 3,
      "points": [
        [591, 208], [590, 125], [451, 132], [464, 212], [592, 211]
      ]
    }
  ]
}

cap = cv2.VideoCapture(0)  # Inicializa a captura de vídeo (0 = webcam padrão)

while True:  # Loop principal para processar cada frame do vídeo
    ret, frame = cap.read()  # Lê um frame da câmera; ret indica sucesso, frame é a imagem capturada
    if not ret:              # Se não conseguiu ler o frame (fim do vídeo ou erro), sai do loop
        break

    results = model(frame)   # Executa a detecção de objetos no frame usando YOLO
    detections = results[0].boxes.xyxy.cpu().numpy()  # Obtém as caixas delimitadoras das detecções
    classes = results[0].boxes.cls.cpu().numpy()      # Obtém as classes dos objetos detectados

    # Para cada vaga definida
    for i, vaga in enumerate(parking_spots["slots"]):
        pts = np.array(vaga["points"], np.int32)      # Converte os pontos da vaga para array NumPy
        pts = pts.reshape((-1, 1, 2))                 # Ajusta o formato para uso com funções do OpenCV
        vaga_ocupada = False                          # Inicializa o status da vaga como livre

        # Para cada detecção de objeto
        for box, cls in zip(detections, classes):
            if int(cls) in [2, 3, 5, 7]:              # Filtra apenas carros, motos, ônibus e caminhões
                x1, y1, x2, y2 = box                  # Coordenadas da caixa delimitadora
                cx = int((x1 + x2) / 2)               # Calcula o centro X da caixa
                cy = int((y1 + y2) / 2)               # Calcula o centro Y da caixa
                # Verifica se o centro do objeto está dentro do polígono da vaga
                if cv2.pointPolygonTest(pts, (cx, cy), False) >= 0:
                    vaga_ocupada = True               # Marca a vaga como ocupada

        color = (0, 0, 255) if vaga_ocupada else (0, 255, 0)  # Vermelho se ocupada, verde se livre
        status = "OCUPADA" if vaga_ocupada else "LIVRE"       # Texto de status

        # Desenha o polígono da vaga na imagem
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
        # Escreve o status da vaga na imagem
        cv2.putText(frame, f"Vaga {vaga['id']}: {status}", (pts[0][0][0], pts[0][0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Parking AI", frame)  # Exibe o frame processado em uma janela

    # Espera 1ms por uma tecla e verifica se foi pressionado 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

cap.release()           # Libera o recurso da câmera
cv2.destroyAllWindows   # Fecha todas as janelas abertas do OpenCV