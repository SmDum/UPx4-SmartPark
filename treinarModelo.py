from ultralytics import YOLO

# Carregar modelo base YOLO pré-treinado
model = YOLO("yolov8n.pt")

# Treinar no dataset exportado do Roboflow
model.train(
    data="../dataset/data.yaml",  # caminho para o dataset
    epochs=300,                 # número de épocas
    imgsz=640,                  # tamanho das imagens
    batch = 8,                   # tamanho do batch (ajuste conforme sua GPU permita)
    patience = 50,              # paciência para early stopping
    augment = True,            # aplicar aumentos de dados (data augmentation)
)
