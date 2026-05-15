import time

import cv2

from transmision.implementation.color_palette import ColorPalette
from transmision.implementation.grid import Grid64Codec
from transmision.implementation.camera import OpenCVCamera
from transmision.implementation.qr_adapter import QRLightAdapter


def main():

    codec = ColorPalette(n_colors=2)

    grid = Grid64Codec(module_px=24)

    camera = OpenCVCamera(
        device_id=700,
        width=1280,
        height=720,
    )

    adapter = QRLightAdapter(
        camera=camera,
        grid_codec=grid,
        color_codec=codec,
        fullscreen=True,
        display_ms=15000,
    )

    payload = b"Hola QR-NET"

    print(f"[TX] Enviando: {payload}")

    adapter.send(payload)

    print("[RX] Esperando lectura...")

    start = time.time()

    timeout = 15

    received = None

    while received is None:

        adapter.update()

        received = adapter.receive()

        cv2.waitKey(1)

        if time.time() - start > timeout:
            break

    print(f"[RX] Recibido: {received}")

    adapter.close()

    print("[OK] Test finalizado")


if __name__ == "__main__":
    main()