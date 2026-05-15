"""
capa1/camera.py
Implementación de ICameraInterface usando OpenCV.

El hilo productor llama a capture() en un loop y pasa los frames
a FifoFrameQueue. detect_change() evita encolar frames duplicados
cuando la pantalla no ha cambiado todavía.
"""
from __future__ import annotations

import cv2
import numpy as np

from common.exceptions import AdapterError
from transmision.interfaces import ICameraInterface


class OpenCVCamera(ICameraInterface):
    """
    Captura frames desde un dispositivo de cámara real o virtual
    (v4l2, DirectShow, AVFoundation) usando cv2.VideoCapture.

    Parámetros:
        device_id  : índice del dispositivo (0 = cámara por defecto)
        width, height : resolución deseada (la cámara puede ignorarla)
        fps        : FPS deseados
        diff_threshold : umbral de diferencia media para detect_change()
    """

    def __init__(
        self,
        device_id: int = 0,
        width: int = 1920,
        height: int = 1080,
        fps: float = 30.0,
        diff_threshold: float = 8.0,
    ) -> None:
        self._device_id = device_id
        self._width = width
        self._height = height
        self._fps = fps
        self._diff_threshold = diff_threshold
        self._cap: cv2.VideoCapture | None = None

    # ── ICameraInterface ──────────────────────────────────────────────────────

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self._device_id)
        if not self._cap.isOpened():
            raise AdapterError(f"No se pudo abrir la cámara con device_id={self._device_id}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS,          self._fps)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def capture(self) -> np.ndarray:
        """
        Lee un frame del dispositivo.
        Retorna array (H, W, 3) BGR uint8.
        Lanza AdapterError si la cámara no está abierta o el frame falla.
        """
        if self._cap is None or not self._cap.isOpened():
            raise AdapterError("Cámara no inicializada — llame a open() primero")
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise AdapterError("No se pudo leer frame de la cámara")
        return frame

    def detect_change(self, prev: np.ndarray, curr: np.ndarray) -> bool:
        """
        Retorna True si la diferencia media absoluta entre frames
        supera diff_threshold. Trabaja en escala de grises para eficiencia.
        """
        if prev is None or prev.shape != curr.shape:
            return True
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, curr_gray)
        return float(diff.mean()) > self._diff_threshold

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def resolution(self) -> tuple[int, int]:
        return self._width, self._height

    # ── contexto ──────────────────────────────────────────────────────────────

    def __enter__(self) -> OpenCVCamera:
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()