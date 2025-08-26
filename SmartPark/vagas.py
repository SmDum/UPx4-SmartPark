import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8n.pt")

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

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()

    for i, vaga in enumerate(parking_spots["slots"]):
        pts = np.array(vaga["points"], np.int32)
        pts = pts.reshape((-1, 1, 2))
        vaga_ocupada = False

        for box, cls in zip(detections, classes):
            if int(cls) in [2, 3, 5, 7]:
                x1, y1, x2, y2 = box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                # Verifica se o centro está dentro do polígono
                if cv2.pointPolygonTest(pts, (cx, cy), False) >= 0:
                    vaga_ocupada = True

        color = (0, 0, 255) if vaga_ocupada else (0, 255, 0)
        status = "OCUPADA" if vaga_ocupada else "LIVRE"

        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
        cv2.putText(frame, f"Vaga {vaga['id']}: {status}", (pts[0][0][0], pts[0][0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Parking AI", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Espera por 1ms e verifica se a tecla 'q' foi pressionada para sair
      break

cap.release()
cv2.destroyAllWindows()