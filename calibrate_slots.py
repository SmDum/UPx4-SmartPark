import cv2
import json
import numpy as np
import argparse
from pathlib import Path

INSTRUCTIONS = """
Desenhe as vagas:
- Clique para adicionar pontos do polígono.
- ENTER (tecla Return) fecha o polígono atual (precisa >=3 pontos).
- N começa uma nova vaga (se já fechou a anterior).
- Z desfaz o último ponto.
- S salva em slots.json.
- Q sai sem salvar.
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="0 para webcam ou caminho do vídeo")
    ap.add_argument("--out", default="slots.json", help="Arquivo de saída")
    args = ap.parse_args()

    cap = cv2.VideoCapture(int(args.source) if str(args.source).isdigit() else args.source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir: {args.source}")

    window = "Calibrador - Desenhe as vagas"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    slots = []
    current = []
    h, w = None, None

    def on_mouse(event, x, y, flags, param):
        nonlocal current
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append((int(x), int(y)))

    cv2.setMouseCallback(window, on_mouse)

    print(INSTRUCTIONS)
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Não foi possível ler frame da câmera.")
            break

        if h is None or w is None:
            h, w = frame.shape[:2]
            cv2.resizeWindow(window, min(1200, w), min(700, h))

        disp = frame.copy()

        # Desenha os polígonos (vagas) já salvos
        for s in slots:
            pts = np.array(s["points"], dtype=np.int32)
            cv2.polylines(disp, [pts], True, (0, 200, 0), 2)
            cx, cy = pts.mean(axis=0).astype(int)
            cv2.putText(disp, f"Vaga {s['id']}", (cx-20, cy-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,0), 2)

        # Desenha o polígono em edição (ainda não salvo)
        if current:
            for i, p in enumerate(current):
                cv2.circle(disp, p, 4, (255,255,255), -1)
                if i > 0:
                    cv2.line(disp, current[i-1], current[i], (255,255,255), 2)

        # Escreve as instruções na imagem
        cv2.rectangle(disp, (0, 0), (disp.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(disp, "[ENTER] Salvar | [N] Nova Vaga | [Z] Desfazer | [S] Salvar | [Q] Sair",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

        cv2.imshow(window, disp)
        key = cv2.waitKey(20) & 0xFF

        if key == 13:  # ENTER fecha polígono
            if len(current) >= 3:
                slots.append({"id": len(slots) + 1, "points": [[int(x), int(y)] for (x,y) in current]})
                current = []
            else:
                print("Polígono precisa de pelo menos 3 pontos.")
        elif key == ord('n') or key == ord('N'):
            current = []
        elif key == ord('z') or key == ord('Z'):
            if current:
                current.pop()
        elif key == ord('s') or key == ord('S'):
            data = {"image_size": [w, h], "slots": slots}
            out_path = Path(args.out)
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"Salvo em {out_path}")
        elif key == ord('q') or key == ord('Q'):
            print("Saindo (sem salvar)...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()