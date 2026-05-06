"""
color_test.py
Prueba visual de colores — independiente del detector de grilla.

Muestra los patches de calibración en Pantalla 1 y el feed de la
cámara en tiempo real. Permite ver qué colores distingue la cámara
antes de intentar decodificar la grilla completa.

Teclas:
    2 / 4 / 8 / F → cambiar a 2 / 4 / 8 / 16 colores
    Q             → salir
    G             → guardar captura
"""
from __future__ import annotations
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import cv2
import numpy as np
from transmision.camera_calibration import generate_calibration_image, PATCH_PX, GRID_COLS, BORDER_PX, GAP_PX, LABEL_H
from transmision.implementation.color_palette import ColorPalette, _rgb_to_hsv, _hsv_distance
import math

CAMERA_ID  = 700
MONITOR_W  = 1920
MONITOR_H  = 1080
N_COLORS   = 16

cap = cv2.VideoCapture(CAMERA_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Ventana TX en pantalla completa en monitor 1
cv2.namedWindow("COLOR CAL TX", cv2.WINDOW_NORMAL)
cv2.moveWindow("COLOR CAL TX", 1920, 0)
cv2.setWindowProperty("COLOR CAL TX", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Ventana RX en monitor principal
cv2.namedWindow("COLOR CAL RX", cv2.WINDOW_NORMAL)
cv2.resizeWindow("COLOR CAL RX", 1000, 600)

print("=" * 55)
print("  QR-NET — Prueba de calibración de colores")
print("=" * 55)
print("  Pantalla 1: patches de color")
print("  RX: feed de cámara + medición en tiempo real")
print()
print("  2 = 2 colores  4 = 4 colores")
print("  8 = 8 colores  F = 16 colores")
print("  Q = salir  G = guardar captura")
print("=" * 55)

def build_rx_display(frame: np.ndarray, palette: ColorPalette,
                     measured: list[tuple[int,int,int] | None],
                     n_colors: int) -> np.ndarray:
    """
    Construye el panel RX:
    Izquierda: feed de cámara
    Derecha: tabla de colores esperados vs observados
    """
    fh, fw = frame.shape[:2]
    canvas = np.zeros((600, 1000, 3), dtype=np.uint8)

    # Feed de cámara en la mitad izquierda
    cam_w, cam_h = 500, 400
    cam_resized = cv2.resize(frame, (cam_w, cam_h))
    canvas[0:cam_h, 0:cam_w] = cam_resized
    cv2.putText(canvas, f"Camara (device {CAMERA_ID})",
                (10, cam_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)

    # Tabla derecha: color esperado vs observado
    col_x = 510
    cv2.putText(canvas, f"{n_colors} colores — Esperado vs Observado",
                (col_x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    rows = math.ceil(n_colors / 4)
    patch_s = min(55, (580 - 40) // rows)

    for i in range(n_colors):
        row = i // 4
        col = i % 4
        x0 = col_x + col * (patch_s + 4)
        y0 = 40 + row * (patch_s * 2 + 8)

        # Color esperado (teoría)
        r, g, b = palette.encode(i)
        cv2.rectangle(canvas, (x0, y0), (x0+patch_s, y0+patch_s),
                      (b, g, r), -1)
        cv2.putText(canvas, f"{i}", (x0+2, y0+patch_s-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1)

        # Color observado por cámara
        if measured[i] is not None:
            ro, go, bo = measured[i]
            obs_bgr = (bo, go, ro)
            cv2.rectangle(canvas,
                          (x0, y0+patch_s+2),
                          (x0+patch_s, y0+patch_s*2+2),
                          obs_bgr, -1)
            # Medir distancia HSV
            h1,s1,v1 = _rgb_to_hsv((r,g,b))
            h2,s2,v2 = _rgb_to_hsv((ro,go,bo))
            dist = _hsv_distance(h1,s1,v1, h2,s2,v2)
            ok_color = (0,255,0) if dist < 0.15 else (0,140,255) if dist < 0.3 else (0,0,255)
            cv2.rectangle(canvas,
                          (x0, y0+patch_s+2),
                          (x0+patch_s, y0+patch_s*2+2),
                          ok_color, 1)
        else:
            cv2.rectangle(canvas,
                          (x0, y0+patch_s+2),
                          (x0+patch_s, y0+patch_s*2+2),
                          (40,40,40), -1)

    # Leyenda
    cv2.putText(canvas, "Arriba=esperado  Abajo=observado",
                (col_x, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120,120,120), 1)
    cv2.putText(canvas, "Verde=OK  Naranja=dudoso  Rojo=confundible",
                (col_x, 595), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120,120,120), 1)
    return canvas


def measure_patches_from_frame(
    frame: np.ndarray, n_colors: int, patch_px: int = PATCH_PX
) -> list[tuple[int,int,int] | None]:
    """
    Lee el color promedio de cada patch de calibración en el frame capturado.
    Asume que la cámara ve la pantalla completa (MONITOR_W × MONITOR_H).
    """
    fh, fw = frame.shape[:2]
    # Escalar frame a resolución del monitor
    if fw != MONITOR_W or fh != MONITOR_H:
        frame = cv2.resize(frame, (MONITOR_W, MONITOR_H))

    rows = math.ceil(n_colors / GRID_COLS)
    cols_n = min(n_colors, GRID_COLS)
    grid_w = cols_n * (patch_px + GAP_PX) - GAP_PX + 2 * BORDER_PX
    grid_h = rows * (patch_px + LABEL_H + GAP_PX) - GAP_PX + 2 * BORDER_PX
    grid_x0 = (MONITOR_W - grid_w) // 2
    grid_y0 = (MONITOR_H - grid_h) // 2

    results: list[tuple[int,int,int] | None] = []
    margin = patch_px // 5

    for i in range(n_colors):
        row = i // GRID_COLS
        col = i % GRID_COLS
        px0 = grid_x0 + BORDER_PX + col * (patch_px + GAP_PX)
        py0 = grid_y0 + BORDER_PX + row * (patch_px + LABEL_H + GAP_PX)

        region = frame[
            py0 + margin : py0 + patch_px - margin,
            px0 + margin : px0 + patch_px - margin,
        ]
        if region.size == 0:
            results.append(None)
            continue

        bgr = region.mean(axis=(0,1))
        results.append((int(bgr[2]), int(bgr[1]), int(bgr[0])))  # RGB

    return results


# ── loop principal ────────────────────────────────────────────────────────────

palette  = ColorPalette(n_colors=N_COLORS)
cal_img  = generate_calibration_image(N_COLORS, PATCH_PX, MONITOR_W, MONITOR_H)
measured: list = [None] * N_COLORS
save_idx = 0

cv2.imshow("COLOR CAL TX", cal_img)

while True:
    ok, frame = cap.read()
    if not ok:
        continue

    # Medir colores en este frame
    measured = measure_patches_from_frame(frame, N_COLORS)

    # Construir y mostrar panel RX
    rx = build_rx_display(frame, palette, measured, N_COLORS)
    cv2.imshow("COLOR CAL RX", rx)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q') or key == 27:
        break
    elif key == ord('2'):
        N_COLORS = 2
        palette  = ColorPalette(n_colors=N_COLORS)
        cal_img  = generate_calibration_image(N_COLORS, PATCH_PX, MONITOR_W, MONITOR_H)
        measured = [None] * N_COLORS
        cv2.imshow("COLOR CAL TX", cal_img)
        print(f"[CAL] Cambiado a {N_COLORS} colores")
    elif key == ord('4'):
        N_COLORS = 4
        palette  = ColorPalette(n_colors=N_COLORS)
        cal_img  = generate_calibration_image(N_COLORS, PATCH_PX, MONITOR_W, MONITOR_H)
        measured = [None] * N_COLORS
        cv2.imshow("COLOR CAL TX", cal_img)
        print(f"[CAL] Cambiado a {N_COLORS} colores")
    elif key == ord('8'):
        N_COLORS = 8
        palette  = ColorPalette(n_colors=N_COLORS)
        cal_img  = generate_calibration_image(N_COLORS, PATCH_PX, MONITOR_W, MONITOR_H)
        measured = [None] * N_COLORS
        cv2.imshow("COLOR CAL TX", cal_img)
        print(f"[CAL] Cambiado a {N_COLORS} colores")
    elif key == ord('f'):
        N_COLORS = 16
        palette  = ColorPalette(n_colors=N_COLORS)
        cal_img  = generate_calibration_image(N_COLORS, PATCH_PX, MONITOR_W, MONITOR_H)
        measured = [None] * N_COLORS
        cv2.imshow("COLOR CAL TX", cal_img)
        print(f"[CAL] Cambiado a {N_COLORS} colores")
    elif key == ord('g'):
        ts = int(time.time())
        cv2.imwrite(f"color_test_rx_{ts}.png", rx)
        print(f"[SAVE] color_test_rx_{ts}.png")
        save_idx += 1

cap.release()
cv2.destroyAllWindows()