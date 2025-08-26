# main.py
import cv2
import json
import numpy as np
from ultralytics import YOLO

SLOTS_JSON = "slots.json"   # gerado pelo calibrate_slots.py
VIDEO_SOURCE = 0            # 0 = webcam | ou "data/estacionamento.mp4"
CONF_MIN = 0.60             # confiança mínima para considerar a detecção
VALID_CLASSES = {"car", "truck"}  # classes que contam como veículo

# ----------------------
# Utilidades de geometria
# ----------------------
def load_slots(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["slots"]

def point_in_poly(pt, poly_pts):
    """Ray casting: True se ponto (x,y) está dentro do polígono."""
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
    """Aproxima a área de interseção (retângulo x polígono) / área do polígono via máscaras."""
    x1, y1, x2, y2 = [int(v) for v in rect_xyxy]
    w, h = frame_wh
    x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return 0.0

    mask_poly = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask_poly, [np.array(poly_pts, dtype=np.int32)], 1)

    mask_rect = np.zeros_like(mask_poly)
    cv2.rectangle(mask_rect, (x1, y1), (x2, y2), 1, thickness=-1)

    inter = np.logical_and(mask_poly, mask_rect).sum()
    poly_area = mask_poly.sum() + 1e-9
    return float(inter) / float(poly_area)

def draw_slots(frame, slots, occupied_ids):
    """Desenha as vagas coloridas e seus rótulos."""
    for s in slots:
        sid = s["id"]
        pts = np.array(s["points"], dtype=np.int32)
        color = (0, 0, 255) if sid in occupied_ids else (0, 200, 70)  # BGR
        cv2.polylines(frame, [pts], True, color, 2)
        # label no centróide
        cx, cy = pts.mean(axis=0).astype(int)
        txt = f"Vaga {sid} - {'OCUP.' if sid in occupied_ids else 'LIVRE'}"
        cv2.putText(frame, txt, (cx - 40, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return frame

# ----------------------
# Main
# ----------------------
def main():
    # 1) Modelo
    model = YOLO("yolov8n.pt")

    # 2) Slots
    slots = load_slots(SLOTS_JSON)

    # 3) Vídeo
    src = VIDEO_SOURCE if isinstance(VIDEO_SOURCE, int) or str(VIDEO_SOURCE).isdigit() else VIDEO_SOURCE
    if isinstance(src, str) and src.isdigit():
        src = int(src)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir a fonte: {VIDEO_SOURCE}")

    cv2.namedWindow("Parking AI", cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]

        # 4) Inferência
        results = model(frame, stream=True, imgsz=640, conf=CONF_MIN)

        # 5) Coletar detecções filtradas
        detections = []  # lista de (x1,y1,x2,y2, label, conf)
        for r in results:
            names = r.names
            for box in r.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = names.get(cls, str(cls))
                if label in VALID_CLASSES and conf >= CONF_MIN:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append((x1, y1, x2, y2, label, conf))

        # 6) Decidir ocupação por vaga
        occupied_ids = set()
        for s in slots:
            sid = s["id"]
            poly = s["points"]
            occ = False
            for (x1, y1, x2, y2, label, conf) in detections:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                center_inside = point_in_poly((cx, cy), poly)
                overlap = rect_poly_overlap_ratio((x1, y1, x2, y2), poly, (w, h))
                if center_inside or overlap > 0.15:
                    occ = True
                    break
            if occ:
                occupied_ids.add(sid)

        # 7) Desenhar vagas e (opcional) caixas dos veículos
        frame = draw_slots(frame, slots, occupied_ids)
        for (x1, y1, x2, y2, label, conf) in detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (220, 220, 220), 1)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

        # 8) Contadores
        total = len(slots)
        occ_count = len(occupied_ids)
        free_count = total - occ_count
        cv2.rectangle(frame, (10, 10), (280, 80), (30, 30, 30), -1)
        cv2.putText(frame, f"Total: {total}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, f"Ocupadas: {occ_count}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 200, 50) if occ_count==0 else (50, 50, 200), 2)
        cv2.putText(frame, f"Livres: {free_count}", (150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 200, 50), 2)

        cv2.imshow("Parking AI", frame)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):  # q ou ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
