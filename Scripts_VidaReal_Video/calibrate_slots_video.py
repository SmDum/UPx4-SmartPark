import cv2
import json
import os
import numpy as np

VIDEO_PATH = "../videos/estacionamento1.mp4"
OUTPUT_JSON = "slots.json"

# Variáveis globais
drawing = False
current_points = []
slots = []

def draw_polygon(event, x, y, flags, param):
    global drawing, current_points, frame_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        current_points.append((x, y))
        cv2.circle(frame_copy, (x, y), 3, (0, 255, 0), -1)

    elif event == cv2.EVENT_RBUTTONDOWN and current_points:
        cv2.polylines(frame_copy, [np.array(current_points, dtype=np.int32)], True, (255, 0, 0), 2)
        vaga_id = len(slots) + 1
        slots.append({"id": vaga_id, "points": current_points.copy()})
        current_points.clear()

def show_instructions(img):
    """Mostra instruções na tela"""
    instructions = [
        "=== MENU DE INSTRUCOES ===",
        "Clique ESQUERDO: marcar pontos da vaga",
        "Clique DIREITO: salvar vaga atual",
        "Q: sair e salvar JSON",
    ]
    y_offset = 25
    for i, text in enumerate(instructions):
        color = (0, 255, 255) if i > 0 else (255, 255, 255)
        cv2.putText(img, text, (10, y_offset + (i * 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# Ler o primeiro frame do vídeo
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()

if not ret:
    raise Exception("Erro ao abrir o vídeo.")

frame_copy = frame.copy()
cv2.namedWindow("Selecione as vagas (clique esquerdo para marcar pontos, direito para salvar)")
cv2.setMouseCallback("Selecione as vagas (clique esquerdo para marcar pontos, direito para salvar)", draw_polygon)

while True:
    display_frame = frame_copy.copy()

    # Mostrar as vagas desenhadas
    for slot in slots:
        pts = np.array(slot["points"], dtype=np.int32)
        cv2.polylines(display_frame, [pts], True, (0, 255, 0), 2)
        cv2.putText(display_frame, f"ID {slot['id']}", tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Mostrar menu na tela
    show_instructions(display_frame)

    cv2.imshow("Selecione as vagas (clique esquerdo para marcar pontos, direito para salvar)", display_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # pressiona 'q' para sair
        break

cv2.destroyAllWindows()

# Salvar o resultado em JSON
if slots:
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"slots": slots}, f, indent=2)
    print(f"✅ Arquivo {OUTPUT_JSON} salvo com {len(slots)} vagas.")
else:
    print("Nenhuma vaga salva.")
