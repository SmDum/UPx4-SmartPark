import cv2
from ultralytics import YOLO

# Carrega o modelo YOLO pré-treinado (versão pequena para ser rápido)
model = YOLO("yolov8n.pt")  

video_path = 0  # webcam ou "data/teste.mp4"
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Faz a detecção no frame
    results = model(frame, stream=True)

    # Desenha as detecções
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            # Filtra apenas carros e caminhões acima de 60% de confiança
            if label in ["car", "truck"] and conf > 0.6:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)

    cv2.imshow("Deteccao de Carros", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
