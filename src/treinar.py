from ultralytics import YOLO

# Carregar modelo base YOLO pré-treinado
model = YOLO("yolov8n.pt")

# Treinar no dataset exportado do Roboflow
model.train(
    data="../dataset/data.yaml",  # caminho para o dataset
    epochs=50,                 # número de épocas
    imgsz=640                  # tamanho das imagens
)
