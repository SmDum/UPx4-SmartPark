import cv2  # Biblioteca para processamento de imagens e acesso à webcam
import os
from datetime import datetime  # Para gerar timestamp nos nomes dos arquivos
from pathlib import Path  # Para manipulação de caminhos de arquivos


class PhotoSaver:
    """
    Classe responsável por gerar nomes de arquivos, garantir a existência da pasta
    e salvar frames (imagens) em disco.
    """
    def __init__(self, output_dir: str = "fotos_modelo"):
        # Define o diretório de saída e cria a pasta se não existir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _timestamp_name(self) -> str:
        # Gera um nome de arquivo baseado na data e hora atual
        # Exemplo: 2025-09-03_13-22-45-123456.jpg
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f") + ".jpg"

    def save(self, frame) -> str:
        # Salva o frame (imagem) no disco com nome único
        filename = self._timestamp_name()
        path = self.output_dir / filename
        ok = cv2.imwrite(str(path), frame)  # Salva a imagem usando OpenCV
        if not ok:
            raise IOError("Falha ao salvar a imagem em disco.")
        return str(path)  # Retorna o caminho do arquivo salvo


class Webcam:
    """
    Classe que encapsula o acesso à webcam via OpenCV.
    """
    def __init__(self, device_index: int = 0, width: int = 640, height: int = 480):
        # Inicializa os parâmetros da webcam
        self.device_index = device_index  # Índice do dispositivo da webcam
        self.width = width  # Largura desejada do frame
        self.height = height  # Altura desejada do frame
        self.cap = None  # Objeto de captura da webcam

    def open(self):
        # Abre a webcam usando OpenCV
        self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)  # CAP_DSHOW evita demora no Windows
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = cv2.VideoCapture(self.device_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir a câmera (índice {self.device_index}).")

        # Tenta ajustar a resolução da webcam
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read(self):
        # Lê um frame da webcam
        if self.cap is None:
            raise RuntimeError("Câmera não está aberta. Chame open() antes.")
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError("Falha ao ler frame da câmera.")
        return frame  # Retorna o frame capturado

    def release(self):
        # Libera o recurso da webcam
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class CameraController:
    """
    Classe que controla o loop principal: exibe o preview da webcam,
    escuta teclas e captura foto ao pressionar ENTER.
    """
    INSTRUCTIONS = "ENTER: tirar foto | Q/ESC: sair"  # Texto de instruções na tela

    def __init__(self, device_index: int = 0, output_dir: str = "fotos_modelo", window_name: str = "Preview da Webcam"):
        # Inicializa a webcam, o salvador de fotos e o nome da janela
        self.webcam = Webcam(device_index=device_index)
        self.saver = PhotoSaver(output_dir=output_dir)
        self.window_name = window_name

    @staticmethod
    def _is_enter(key: int) -> bool:
        # Verifica se a tecla pressionada foi ENTER
        return key in (10, 13)

    @staticmethod
    def _should_quit(key: int) -> bool:
        # Verifica se a tecla pressionada foi ESC ou Q (para sair)
        return key in (27, ord('q'), ord('Q'))

    def _draw_overlay(self, frame):
        # Desenha as instruções sobre o frame da webcam
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(frame, self.INSTRUCTIONS, (10, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        return frame

    def run(self):
        # Loop principal do programa
        try:
            self.webcam.open()  # Abre a webcam
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)  # Cria janela de preview
            cv2.resizeWindow(self.window_name, 900, 600)  # Ajusta tamanho da janela

            while True:
                frame = self.webcam.read()  # Lê frame da webcam
                frame_with_text = self._draw_overlay(frame.copy())  # Adiciona instruções ao frame
                cv2.imshow(self.window_name, frame_with_text)  # Mostra o frame na janela

                key = cv2.waitKey(1) & 0xFF  # Captura tecla pressionada

                if self._is_enter(key):
                    # Se ENTER pressionado, salva a imagem original
                    saved_path = self.saver.save(frame)
                    print(f"[OK] Foto salva em: {saved_path}")

                    # Feedback visual: desenha um retângulo verde rápido
                    flash = frame.copy()
                    cv2.rectangle(flash, (0, 0), (flash.shape[1]-1, flash.shape[0]-1), (0, 255, 0), 12)
                    cv2.imshow(self.window_name, flash)
                    cv2.waitKey(200)

                elif self._should_quit(key):
                    # Se ESC ou Q pressionado, sai do loop
                    break

        except Exception as e:
            print(f"[ERRO] {e}")  # Exibe erro caso ocorra
        finally:
            self.webcam.release()  # Libera webcam
            cv2.destroyAllWindows()  # Fecha todas as janelas do OpenCV


if __name__ == "__main__":
    # Ponto de entrada do programa
    controller = CameraController(device_index=0, output_dir="fotos_modelo", window_name="Preview da Webcam")
    controller.run()