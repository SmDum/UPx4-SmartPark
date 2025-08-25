# calibrate_slots.py

import cv2                # Importa a biblioteca OpenCV para processamento de imagens e interface gráfica
import json               # Importa para salvar e carregar dados em formato JSON
import numpy as np        # Importa para manipulação de arrays numéricos
import argparse           # Importa para processar argumentos de linha de comando
from pathlib import Path  # Importa para manipulação de caminhos de arquivos de forma mais segura

# Texto de instruções exibido ao usuário
INSTRUCTIONS = """
Desenhe as vagas:
- Clique para adicionar pontos do polígono.
- ENTER (tecla Return) fecha o polígono atual (precisa >=3 pontos).
- N começa uma nova vaga (se já fechou a anterior).
- Z desfaz o último ponto.
- S salva em slots.json.
- Q sai sem salvar.
"""

# Função para capturar o primeiro frame do vídeo ou webcam
def grab_first_frame(source):
    # Abre o vídeo ou webcam (se source for número, converte para int)
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir: {source}")  # Erro se não conseguir abrir
    ok, frame = cap.read()  # Lê o primeiro frame
    cap.release()           # Libera o recurso da câmera/vídeo
    if not ok:
        raise RuntimeError("Não foi possível ler frame inicial.")  # Erro se não conseguir ler frame
    return frame            # Retorna o frame capturado

# Função principal do script
def main():
    ap = argparse.ArgumentParser()  # Cria o parser de argumentos
    ap.add_argument("--source", required=True, help="0 para webcam ou caminho do vídeo")  # Argumento obrigatório: fonte do vídeo
    ap.add_argument("--out", default="slots.json", help="Arquivo de saída")               # Argumento opcional: arquivo de saída
    args = ap.parse_args()  # Faz o parsing dos argumentos

    frame = grab_first_frame(args.source)  # Captura o primeiro frame da fonte escolhida
    h, w = frame.shape[:2]                 # Obtém altura e largura da imagem

    window = "Calibrador - Desenhe as vagas"  # Nome da janela do OpenCV
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)  # Cria janela redimensionável
    cv2.resizeWindow(window, min(1200, w), min(700, h))  # Redimensiona a janela para caber na tela

    slots = []   # Lista para armazenar as vagas já desenhadas (cada vaga é um dicionário)
    current = [] # Lista para armazenar os pontos do polígono em edição

    # Função de callback para eventos do mouse
    def on_mouse(event, x, y, flags, param):
        nonlocal current
        if event == cv2.EVENT_LBUTTONDOWN:         # Se clicar com o botão esquerdo
            current.append((int(x), int(y)))       # Adiciona o ponto (x, y) à lista de pontos atuais

    cv2.setMouseCallback(window, on_mouse)         # Registra a função de callback para eventos do mouse

    print(INSTRUCTIONS)                            # Exibe as instruções no terminal
    while True:
        disp = frame.copy()                        # Cria uma cópia do frame para desenhar

        # Desenha os polígonos (vagas) já salvos
        for s in slots:
            pts = np.array(s["points"], dtype=np.int32)  # Converte lista de pontos para array numpy
            cv2.polylines(disp, [pts], True, (0, 200, 0), 2)  # Desenha o polígono em verde
            cx, cy = pts.mean(axis=0).astype(int)            # Calcula o centro do polígono
            cv2.putText(disp, f"Vaga {s['id']}", (cx-20, cy-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,0), 2)  # Escreve o número da vaga

        # Desenha o polígono em edição (ainda não salvo)
        if current:
            for i, p in enumerate(current):
                cv2.circle(disp, p, 4, (255,255,255), -1)        # Desenha um círculo em cada ponto
                if i > 0:
                    cv2.line(disp, current[i-1], current[i], (255,255,255), 2)  # Liga os pontos com linhas

        # Escreve as instruções na imagem
        cv2.putText(disp, "ENTER fechar | N nova vaga | Z desfazer | S salvar | Q sair",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

        cv2.imshow(window, disp)                 # Mostra a imagem na janela
        key = cv2.waitKey(20) & 0xFF             # Espera por uma tecla (20 ms)

        if key == 13:  # ENTER fecha polígono
            if len(current) >= 3:                # Só fecha se tiver pelo menos 3 pontos
                slots.append({"id": len(slots) + 1, "points": [[int(x), int(y)] for (x,y) in current]})  # Salva a vaga
                current = []                     # Limpa os pontos atuais
            else:
                print("Polígono precisa de pelo menos 3 pontos.")  # Avisa se não tiver pontos suficientes
        elif key == ord('n') or key == ord('N'): # Começa nova vaga (descarta edição atual)
            current = []
        elif key == ord('z') or key == ord('Z'): # Desfaz o último ponto adicionado
            if current:
                current.pop()
        elif key == ord('s') or key == ord('S'): # Salva as vagas em arquivo JSON
            data = {"image_size": [w, h], "slots": slots}
            out_path = Path(args.out)
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"Salvo em {out_path}")
        elif key == ord('q') or key == ord('Q'): # Sai do programa sem salvar
            print("Saindo (sem salvar)...")
            break

    cv2.destroyAllWindows()  # Fecha todas as janelas do OpenCV

if __name__ == "__main__":  # Executa a função principal se o script for chamado diretamente
    main()