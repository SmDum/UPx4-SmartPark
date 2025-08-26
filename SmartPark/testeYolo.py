import cv2  # Importa a biblioteca OpenCV para processamento de imagens e vídeo
from ultralytics import YOLO  # Importa a classe YOLO da biblioteca ultralytics para detecção de objetos

# Carrega o modelo YOLO pré-treinado (versão pequena para ser rápido)
model = YOLO("yolov8n.pt")  # Instancia o modelo YOLO com pesos pré-treinados (versão nano)

video_path = 0  # Define a fonte do vídeo: 0 para webcam, ou pode ser um caminho para arquivo de vídeo
cap = cv2.VideoCapture(video_path)  # Inicializa a captura de vídeo

while True:  # Loop principal para processar cada frame do vídeo
    ret, frame = cap.read()  # Lê um frame do vídeo; ret indica sucesso, frame é a imagem capturada
    if not ret:  # Se não conseguiu ler o frame (fim do vídeo ou erro), sai do loop
        break

    # Faz a detecção no frame
    results = model(frame, stream=True)  # Executa a detecção de objetos no frame, retornando resultados em stream

    # Desenha as detecções
    for r in results:  # Itera sobre os resultados de detecção
        for box in r.boxes:  # Itera sobre cada caixa delimitadora detectada
            conf = float(box.conf[0])  # Obtém a confiança da detecção (probabilidade)
            cls = int(box.cls[0])  # Obtém o índice da classe detectada
            label = model.names[cls]  # Obtém o nome da classe a partir do índice

            # Filtra apenas carros e caminhões acima de 60% de confiança
            if label in ["car", "truck"] and conf > 0.6:  # Verifica se é carro ou caminhão e confiança > 60%
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Obtém as coordenadas da caixa delimitadora e converte para int
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Desenha um retângulo verde ao redor do objeto
                cv2.putText(frame, f"{label} {conf:.2f}",  # Escreve o nome e confiança acima da caixa
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)

    cv2.imshow("Deteccao de Carros", frame)  # Exibe o frame processado em uma janela

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Espera por 1ms e verifica se a tecla 'q' foi pressionada para sair
        break

cap.release()  # Libera o recurso da câmera ou arquivo de vídeo
cv2.destroyAllWindows()  # Fecha todas as janelas abertas