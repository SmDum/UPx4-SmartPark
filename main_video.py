import cv2
import json
import numpy as np
from ultralytics import YOLO
import time
from src.redis.models.connection.redis_conn import RedisConnectionHandle
from src.redis.models.redis.redis_repo import RedisRepository

SLOTS_JSON = "Scripts_VidaReal_Video/slots.json"
VIDEO_PATH = "videos/estacionamento1.mp4"
CONF_MIN = 0.5
VALID_CLASSES = {"car", "truck", "bus"}

def load_slots(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["slots"]

def point_in_poly(pt, poly_pts):
    x, y = pt
    inside = False
    n = len(poly_pts)
    for i in range(n):
        x1, y1 = poly_pts[i]
        x2, y2 = poly_pts[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1):
            inside = not inside
    return inside

def rect_poly_overlap_ratio(rect_xyxy, poly_pts, frame_wh):
    x1, y1, x2, y2 = [int(v) for v in rect_xyxy]
    w, h = frame_wh
    mask_poly = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask_poly, [np.array(poly_pts, dtype=np.int32)], 1)
    mask_rect = np.zeros_like(mask_poly)
    cv2.rectangle(mask_rect, (x1, y1), (x2, y2), 1, -1)
    inter = np.logical_and(mask_poly, mask_rect).sum()
    poly_area = mask_poly.sum() + 1e-9
    return inter / poly_area

def draw_slots(frame, slots, occupied_ids):
    for s in slots:
        sid = s["id"]
        pts = np.array(s["points"], dtype=np.int32)
        color = (0, 0, 255) if sid in occupied_ids else (0, 200, 70)
        cv2.polylines(frame, [pts], True, color, 2)
        cx, cy = pts.mean(axis=0).astype(int)
        label = f"Vaga {sid}: {'Ocupada' if sid in occupied_ids else 'Livre'}"
        cv2.putText(frame, label, (cx - 50, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

def main():
    try:
        model = YOLO("yolov8m.pt")
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        print("Certifique-se de que yolov8m.pt existe no diretório")
        return
    
    try:
        slots = load_slots(SLOTS_JSON)
    except Exception as e:
        print(f"Erro ao carregar slots.json: {e}")
        return
    
    # Conecta ao Redis
    try:
        redis_conn = RedisConnectionHandle().connect()
        redis_repo = RedisRepository(redis_conn)
    except Exception as e:
        print(f"Erro ao conectar no Redis: {e}")
        return
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise Exception("Erro ao abrir o vídeo.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        results = model(frame, stream=False, imgsz=640, conf=CONF_MIN)
        detections = []

        for r in results:
            for box in r.boxes:
                cls_name = r.names[int(box.cls[0])]
                conf = float(box.conf[0])
                if cls_name in VALID_CLASSES and conf >= CONF_MIN:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append((x1, y1, x2, y2, cls_name, conf))

        occupied_ids = set()
        status = []

        for s in slots:
            sid = s["id"]
            poly = s["points"]
            occupied = False
            for (x1, y1, x2, y2, label, conf) in detections:
                overlap = rect_poly_overlap_ratio((x1, y1, x2, y2), poly, (w, h))
                if overlap > 0.2:
                    occupied = True
                    break
            if occupied:
                occupied_ids.add(sid)
            status.append({"vaga": sid, "status": 1 if occupied else 0})

        # Publica status no Redis
        ts = int(time.time())
        try:
            payload = json.dumps({"status": status, "ts": ts})
            redis_repo.publish('vagas:update', payload)
        except Exception as e:
            print(f"Erro ao publicar no Redis: {e}")

        frame = draw_slots(frame, slots, occupied_ids)
        cv2.imshow("Parking Detection", frame)
        if cv2.waitKey(20) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
