"""
transmision/implementation/qr_adapter.py

Implementación de INetworkAdapter usando transmisión visual
mediante grillas QR-NET mostradas en pantalla y capturadas por cámara.

Flujo TX:
    bytes -> Grid64Codec.encode_grid() -> imagen OpenCV -> pantalla

Flujo RX:
    cámara -> frame -> Grid64Codec.decode_grid() -> bytes

Este adaptador implementa el canal físico QR_LIGHT.
"""
from __future__ import annotations
import time

import cv2
import numpy as np
import screeninfo

from common.network_policies import AdapterType
from common.exceptions import AdapterError, GridDecodeError

from transmision.interfaces import (
    INetworkAdapter,
    ICameraInterface,
    IGridCodec,
    IColorCodec,
)


class QRLightAdapter(INetworkAdapter):
    """
    Adaptador físico QR/luz.

    Responsabilidades:
    - Mostrar grillas QR-NET en pantalla
    - Capturar imágenes desde cámara
    - Decodificar payloads desde frames reales

    NOTA:
    El adaptador administra automáticamente el ciclo de vida
    de la cámara cuando receive() es llamado.
    """

    WINDOW_NAME = "QR-NET TX"

    def __init__(
            self,
            camera: ICameraInterface,
            grid_codec: IGridCodec,
            color_codec: IColorCodec,
            *,
            display_ms: int = 3000,
            monitor_index: int = 0,
            fullscreen: bool = True,
            mac: bytes = b"\x10\x22\x33\x44\x55\x66",
            cost: int = 100,
    ) -> None:
        self._last_tx_image = None
        self._tx_visible_until = 0.0
        self._camera = camera
        self._grid_codec = grid_codec
        self._color_codec = color_codec
        self._monitor_index = monitor_index
        self._fullscreen = fullscreen
        self._display_ms = display_ms
        self._mac = mac
        self._cost = cost

        self._camera_started = False

    def update(self) -> None:
        """
        Mantiene viva la ventana TX mientras dure display_ms.
        """

        if self._last_tx_image is None:
            return

        if time.time() > self._tx_visible_until:
            cv2.destroyWindow(self.WINDOW_NAME)
            self._last_tx_image = None
            return

        cv2.imshow(self.WINDOW_NAME, self._last_tx_image)
        cv2.waitKey(1)

    def _build_fullscreen_frame(
            self,
            qr_image: np.ndarray,
    ) -> np.ndarray:
        """
        Construye una imagen fullscreen con la grilla centrada
        y fondo blanco puro.
        """

        monitors = screeninfo.get_monitors()

        if self._monitor_index >= len(monitors):
            raise AdapterError(
                f"Monitor {self._monitor_index} no existe"
            )

        monitor = monitors[self._monitor_index]

        screen_w = monitor.width
        screen_h = monitor.height

        # fondo blanco
        canvas = np.ones(
            (screen_h, screen_w, 3),
            dtype=np.uint8,
        ) * 255

        qr_h, qr_w = qr_image.shape[:2]

        # Escalar QR ocupando ~85% del monitor
        scale = min(
            (screen_w * 0.85) / qr_w,
            (screen_h * 0.85) / qr_h,
        )

        target_w = int(qr_w * scale)
        target_h = int(qr_h * scale)

        resized = cv2.resize(
            qr_image,
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST,
        )

        x = (screen_w - target_w) // 2
        y = (screen_h - target_h) // 2

        canvas[
            y:y + target_h,
            x:x + target_w,
        ] = resized

        return canvas
    # ──────────────────────────────────────────────────────────────────────
    # INetworkAdapter
    # ──────────────────────────────────────────────────────────────────────

    def send(self, data: bytes) -> bool:

        try:

            qr = self._grid_codec.encode_grid(
                payload=data,
                codec=self._color_codec,
            )

            frame = self._build_fullscreen_frame(qr)

            cv2.namedWindow(
                self.WINDOW_NAME,
                cv2.WINDOW_NORMAL,
            )

            if self._fullscreen:
                cv2.setWindowProperty(
                    self.WINDOW_NAME,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN,
                )

            cv2.imshow(self.WINDOW_NAME, frame)

            cv2.waitKey(1)

            self._last_tx_image = frame
            self._tx_visible_until = (
                    time.time() + (self._display_ms / 1000.0)
            )

            return True

        except Exception as e:
            raise AdapterError(
                f"Error enviando por QR_LIGHT: {e}"
            ) from e

    def receive(self) -> bytes | None:
        """
        Captura un único frame e intenta decodificarlo.
        No bloquea.
        """

        try:

            self._ensure_camera()

            frame = self._camera.capture()

            try:

                payload = self._grid_codec.decode_grid(
                    image=frame,
                    codec=self._color_codec,
                )

                print("[QR] Payload detectado")

                return payload

            except GridDecodeError:
                return None

        except Exception as e:
            raise AdapterError(
                f"Error recibiendo por QR_LIGHT: {e}"
            ) from e

    def is_available(self) -> bool:
        """
        El adaptador QR se considera disponible si OpenCV está funcional.
        """
        return True

    def get_mac(self) -> bytes:
        return self._mac

    def get_type(self) -> AdapterType:
        return AdapterType.QR_LIGHT

    def get_cost(self) -> int:
        return self._cost

    # ──────────────────────────────────────────────────────────────────────
    # Control de cámara
    # ──────────────────────────────────────────────────────────────────────

    def _ensure_camera(self) -> None:
        """
        Inicializa la cámara una sola vez.
        """

        if not self._camera_started:
            self._camera.open()
            self._camera_started = True

    def close(self) -> None:
        """
        Libera recursos del adaptador.
        """

        if self._camera_started:
            self._camera.close()
            self._camera_started = False

        cv2.destroyWindow(self.WINDOW_NAME)

    # ──────────────────────────────────────────────────────────────────────
    # Context manager
    # ──────────────────────────────────────────────────────────────────────

    def __enter__(self) -> "QRLightAdapter":
        return self

    def __exit__(self, *_) -> None:
        self.close()