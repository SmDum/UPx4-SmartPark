import cv2
import os
from datetime import datetime
from pathlib import Path


class PhotoSaver:
    """
    Responsável por gerar nomes de arquivos, garantir a pasta
    e salvar frames em disco.
    """
    def __init__(self, output_dir: str = "fotos"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _timestamp_name(self) -> str:
        # Ex.: 2025-09-03_13-22-45-123456.jpg
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f") + ".jpg"

    def save(self, frame) -> str:
        filename = self._timestamp_name()
        path = self.output_dir / filename
        # cv2.imwrite retorna True/False
        ok = cv2.imwrite(str(path), frame)
        if not ok:
            raise IOError("Falha ao salvar a imagem em disco.")
        return str(path)


class Webcam:
    """
    Encapsula o acesso à webcam via OpenCV.
    """
    def __init__(self, device_index: int = 0, width: int = 640, height: int = 480):
        self.device_index = device_index
        self.width = width
        self.height = height
        self.cap = None

    def open(self):
        self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)  # CAP_DSHOW evita demora no Windows
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = cv2.VideoCapture(self.device_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir a câmera (índice {self.device_index}).")

        # Tenta ajustar resolução (algumas câmeras podem ignorar)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read(self):
        if self.cap is None:
            raise RuntimeError("Câmera não está aberta. Chame open() antes.")
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError("Falha ao ler frame da câmera.")
        return frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class CameraController:
    """
    Controla o loop: exibe preview, escuta teclas e captura foto no ENTER.
    """
    INSTRUCTIONS = "ENTER: tirar foto | Q/ESC: sair"

    def __init__(self, device_index: int = 0, output_dir: str = "fotos", window_name: str = "Preview da Webcam"):
        self.webcam = Webcam(device_index=device_index)
        self.saver = PhotoSaver(output_dir=output_dir)
        self.window_name = window_name

    @staticmethod
    def _is_enter(key: int) -> bool:
        return key in (10, 13)

    @staticmethod
    def _should_quit(key: int) -> bool:
        return key in (27, ord('q'), ord('Q'))  # ESC ou Q

    def _draw_overlay(self, frame):
        # Desenha instruções no frame
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(frame, self.INSTRUCTIONS, (10, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        return frame

    def run(self):
        try:
            self.webcam.open()
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 900, 600)

            while True:
                frame = self.webcam.read()
                frame_with_text = self._draw_overlay(frame.copy())
                cv2.imshow(self.window_name, frame_with_text)

                # waitKey(1) retorna o código da tecla (ou -1 se nada)
                key = cv2.waitKey(1) & 0xFF

                if self._is_enter(key):
                    # Captura a imagem "bruta" (sem overlay) para salvar
                    saved_path = self.saver.save(frame)
                    print(f"[OK] Foto salva em: {saved_path}")

                    # Feedback visual
                    flash = frame.copy()
                    cv2.rectangle(flash, (0, 0), (flash.shape[1]-1, flash.shape[0]-1), (0, 255, 0), 12)
                    cv2.imshow(self.window_name, flash)
                    cv2.waitKey(200)

                elif self._should_quit(key):
                    break

        except Exception as e:
            print(f"[ERRO] {e}")
        finally:
            self.webcam.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    controller = CameraController(device_index=0, output_dir="fotos", window_name="Preview da Webcam")
    controller.run()