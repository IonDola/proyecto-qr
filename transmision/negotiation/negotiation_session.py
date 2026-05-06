"""
negotiation_session.py
Sesión completa de negociación de color entre emisor y receptor.

Modo EMISOR (--mode tx):
  1. Muestra QR B/N con CalProbeFrame (anuncio)
  2. Muestra imagen de sonda con N_CANDIDATES colores
  3. Espera QR B/N de respuesta del receptor (CalResponseFrame)
  4. Construye paleta custom y la guarda en paleta_negociada.json
  5. Muestra resumen: cuántos colores, cuáles índices, color depth

Modo RECEPTOR (--mode rx):
  1. Escanea QR B/N de anuncio (CalProbeFrame)
  2. Mide colores de la imagen de sonda con la cámara
  3. Construye matriz de confusión
  4. Selecciona paleta óptima
  5. Muestra QR B/N con CalResponseFrame en pantalla para que el emisor lo lea

Uso en misma máquina (prueba local):
  python negotiation_session.py --mode both

Uso en dos máquinas:
  Máquina A (emisor):   python negotiation_session.py --mode tx
  Máquina B (receptor): python negotiation_session.py --mode rx
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import cv2
import numpy as np

from color_negotiation import (
    N_CANDIDATES, N_SAMPLES, PATCH_PX,
    generate_candidates,
    generate_probe_image,
    measure_candidates,
    build_confusion_matrix,
    select_optimal_palette,
    encode_response_as_qr,
    encode_probe_as_qr,
    decode_response_from_qr,
    decode_probe_from_qr,
    build_custom_palette,
)
from transmision.implementation.camera import OpenCVCamera
from transmision.implementation.color_palette import _rgb_to_hsv, _hsv_distance
from transmision.frames import CalProbeFrame, CalResponseFrame

# ── Configuración ─────────────────────────────────────────────────────────────

CAMERA_ID    = 700
MONITOR_TX_X = 1920    # offset X del monitor TX (Monitor 1 a la derecha)
MONITOR_W    = 1920
MONITOR_H    = 1080
MY_MAC       = os.urandom(6)
PEER_MAC     = b'\xff' * 6   # broadcast durante calibración

OUTPUT_FILE  = "paleta_negociada.json"


# ══════════════════════════════════════════════════════════════════════════════
# MODO EMISOR
# ══════════════════════════════════════════════════════════════════════════════

def run_tx(camera_id: int = CAMERA_ID) -> dict:
    """
    Flujo del emisor:
      1. Anuncia la sonda con QR B/N
      2. Muestra colores candidatos
      3. Espera y lee la respuesta del receptor
      4. Construye y guarda la paleta
    """
    candidates = generate_candidates(N_CANDIDATES)

    probe = CalProbeFrame(
        src_mac      = MY_MAC,
        dst_mac      = PEER_MAC,
        n_candidates = N_CANDIDATES,
        patch_px     = PATCH_PX,
        duration_ms  = 4000,
    )

    # ── Ventana TX en Monitor 1 ───────────────────────────────────────────────
    cv2.namedWindow("TX", cv2.WINDOW_NORMAL)
    cv2.moveWindow("TX", MONITOR_TX_X, 0)
    cv2.setWindowProperty("TX", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # ── Ventana RX (cámara local para leer respuesta) ─────────────────────────
    cv2.namedWindow("TX-CAM", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("TX-CAM", 640, 400)

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n[TX] ═══════════════════════════════════════════")
    print("[TX] FASE 1 — Mostrando QR de anuncio (CalProbeFrame)")
    print("[TX] El receptor debe escanear este QR primero")
    print("[TX] Presiona ENTER cuando el receptor esté listo...")

    # Mostrar QR de anuncio
    probe_qr = encode_probe_as_qr(probe, box_size=10)
    probe_canvas = _center_on_black(probe_qr, MONITOR_W, MONITOR_H,
                                    title="QR-NET CAL PROBE — Escanear primero")
    cv2.imshow("TX", probe_canvas)
    cv2.waitKey(1)
    input()   # esperar confirmación

    print("\n[TX] FASE 2 — Mostrando colores candidatos")
    print(f"[TX] {N_CANDIDATES} colores × {N_SAMPLES} muestras")
    print("[TX] El receptor está midiendo el canal...")

    probe_img = generate_probe_image(candidates, PATCH_PX, MONITOR_W, MONITOR_H)
    cv2.imshow("TX", probe_img)
    cv2.waitKey(1)

    # Esperar que el receptor termine de medir (duration_ms + margen)
    wait_ms = probe.duration_ms + 2000
    print(f"[TX] Esperando {wait_ms/1000:.1f}s...")
    deadline = time.time() + wait_ms / 1000
    while time.time() < deadline:
        ok, frame = cap.read()
        if ok:
            cv2.imshow("TX-CAM", frame)
        remaining = max(0, deadline - time.time())
        bar = "█" * int((1 - remaining / (wait_ms/1000)) * 30)
        print(f"\r[TX] [{bar:<30}] {remaining:.1f}s ", end="", flush=True)
        cv2.waitKey(50)
    print()

    print("\n[TX] FASE 3 — Esperando QR de respuesta del receptor")
    print("[TX] El receptor mostrará un QR B/N con su respuesta")
    print("[TX] Apunta la cámara al QR de respuesta...")

    response = _wait_for_response_qr(cap, timeout=60.0)

    cap.release()
    cv2.destroyAllWindows()

    if response is None:
        print("[TX] ✗ No se recibió respuesta en el tiempo límite")
        return {}

    # ── Mostrar resumen ───────────────────────────────────────────────────────
    _print_tx_summary(response, candidates)

    # ── Guardar paleta ────────────────────────────────────────────────────────
    palette_data = _save_palette(candidates, response)
    return palette_data


def _wait_for_response_qr(
    cap: cv2.VideoCapture,
    timeout: float = 60.0,
) -> CalResponseFrame | None:
    """Espera hasta timeout segundos leyendo frames hasta encontrar el QR de respuesta."""
    try:
        from pyzbar import pyzbar
    except ImportError:
        print("[TX] ✗ Instala pyzbar: pip install pyzbar")
        return None

    deadline = time.time() + timeout
    last_found = None

    while time.time() < deadline:
        ok, frame = cap.read()
        if not ok:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        codes = pyzbar.decode(gray)

        found = False
        for code in codes:
            try:
                text = code.data.decode("utf-8")
                d = json.loads(text)
                if d.get("frame_type") == "CAL_RESPONSE":
                    response = CalResponseFrame.from_json(text)
                    # Dibujar borde verde
                    pts = np.array([p for p in code.polygon], dtype=np.int32)
                    cv2.polylines(frame, [pts], True, (0, 255, 0), 3)
                    cv2.putText(frame, f"RESPUESTA OK — {len(response.selected)} colores",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                    cv2.imshow("TX-CAM", frame)
                    cv2.waitKey(500)
                    print(f"\n[TX] ✓ CalResponseFrame recibido — {len(response.selected)} colores seleccionados")
                    return response
            except Exception:
                pass

        # Overlay de búsqueda
        remaining = max(0, deadline - time.time())
        cv2.putText(frame, f"Buscando QR respuesta... {remaining:.0f}s",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)
        cv2.imshow("TX-CAM", frame)
        cv2.waitKey(30)

    return None


# ══════════════════════════════════════════════════════════════════════════════
# MODO RECEPTOR
# ══════════════════════════════════════════════════════════════════════════════

def run_rx(camera_id: int = CAMERA_ID) -> dict:
    """
    Flujo del receptor:
      1. Captura y decodifica el CalProbeFrame del emisor
      2. Mide los colores candidatos con la cámara
      3. Construye la matriz de confusión
      4. Selecciona la paleta óptima
      5. Muestra el QR B/N con CalResponseFrame
    """
    camera = OpenCVCamera(device_id=camera_id, width=1280, height=720)
    camera.open()

    # Ventanas
    cv2.namedWindow("RX-FEED", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RX-FEED", 700, 450)
    cv2.namedWindow("RX-RESPONSE", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RX-RESPONSE", 500, 500)

    print("\n[RX] ═══════════════════════════════════════════")
    print("[RX] FASE 1 — Esperando QR de anuncio del emisor")
    print("[RX] Apunta la cámara al QR de anuncio...")

    # Leer CalProbeFrame
    probe = _wait_for_probe_qr(camera, timeout=60.0)

    if probe is None:
        print("\n[RX] No se encontró QR de anuncio.")
        print("[RX] Usando configuración por defecto:")
        probe = CalProbeFrame(
            src_mac      = PEER_MAC,
            dst_mac      = MY_MAC,
            n_candidates = N_CANDIDATES,
            patch_px     = PATCH_PX,
            duration_ms  = 4000,
        )
        print(f"[RX] n_candidates={probe.n_candidates}, patch_px={probe.patch_px}")

    candidates = generate_candidates(probe.n_candidates)

    print(f"\n[RX] FASE 2 — Midiendo {probe.n_candidates} colores candidatos")
    print(f"[RX] Asegúrate de que la pantalla del emisor muestre los patches de color")
    print(f"[RX] Tomando {N_SAMPLES} muestras durante {probe.duration_ms/1000:.1f}s...")

    # Medir candidatos
    observed = _measure_with_preview(camera, candidates, probe, N_SAMPLES)

    print("\n[RX] FASE 3 — Construyendo matriz de confusión")
    confusion = build_confusion_matrix(observed)

    # Mostrar resumen de confusión
    _print_confusion_summary(confusion, candidates)

    # Seleccionar paleta óptima (máximo 16 colores para CD=3)
    target = min(16, probe.n_candidates)
    selected = select_optimal_palette(candidates, observed, confusion, target_n=target)

    print(f"\n[RX] FASE 4 — Paleta óptima seleccionada: {len(selected)} colores")
    print(f"[RX] Índices: {selected}")

    # Determinar recommended_cd
    n_sel = len(selected)
    if n_sel >= 16: cd = 3
    elif n_sel >= 8: cd = 2
    elif n_sel >= 4: cd = 1
    else: cd = 0

    response = CalResponseFrame(
        src_mac        = MY_MAC,
        dst_mac        = probe.src_mac,
        selected       = selected,
        confusion      = confusion,
        recommended_cd = cd,
        interval_ms    = 150,
    )

    print(f"\n[RX] COLOR_DEPTH recomendado: {cd} ({2**(cd+1)} colores)")
    print(f"[RX] Intervalo recomendado: {response.interval_ms}ms")

    print("\n[RX] FASE 5 — Mostrando QR de respuesta")
    print("[RX] El emisor debe leer este QR con su cámara")

    # Mostrar QR de respuesta
    response_qr = encode_response_as_qr(response, box_size=7)
    response_canvas = _center_on_black(
        response_qr, 900, 900,
        title=f"CalResponseFrame | {len(selected)} colores | CD={cd}"
    )
    cv2.imshow("RX-RESPONSE", response_canvas)

    print("[RX] Presiona Q cuando el emisor confirme la recepción")
    while True:
        key = cv2.waitKey(100) & 0xFF
        if key == ord('q') or key == 27:
            break
        ok, frame = camera.capture() if False else (True, np.zeros((100,100,3), np.uint8))
        try:
            frame = camera.capture()
            cv2.imshow("RX-FEED", frame)
        except Exception:
            pass

    camera.close()
    cv2.destroyAllWindows()

    palette_data = _save_palette(candidates, response)
    return palette_data


def _wait_for_probe_qr(
    camera: OpenCVCamera,
    timeout: float = 60.0,
) -> CalProbeFrame | None:
    """Espera y decodifica el CalProbeFrame del emisor."""
    try:
        from pyzbar import pyzbar
    except ImportError:
        print("[RX] pyzbar no disponible — saltando lectura de probe")
        return None

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            frame = camera.capture()
        except Exception:
            time.sleep(0.1)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        codes = pyzbar.decode(gray)
        for code in codes:
            try:
                text = code.data.decode("utf-8")
                d = json.loads(text)
                if d.get("frame_type") == "CAL_PROBE":
                    probe = CalProbeFrame.from_json(text)
                    pts = np.array([p for p in code.polygon], dtype=np.int32)
                    cv2.polylines(frame, [pts], True, (0,255,0), 3)
                    cv2.putText(frame, "PROBE QR detectado",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                    cv2.imshow("RX-FEED", frame)
                    cv2.waitKey(300)
                    print(f"\n[RX] ✓ CalProbeFrame recibido — {probe.n_candidates} candidatos")
                    return probe
            except Exception:
                pass

        remaining = max(0, deadline - time.time())
        cv2.putText(frame, f"Buscando QR probe... {remaining:.0f}s",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)
        cv2.imshow("RX-FEED", frame)
        cv2.waitKey(30)

    return None


def _measure_with_preview(
    camera: OpenCVCamera,
    candidates: list,
    probe: CalProbeFrame,
    n_samples: int,
) -> list:
    """Mide colores mostrando preview en tiempo real del progreso."""
    from color_negotiation import (
        GRID_COLS, BORDER_PX, GAP_PX, LABEL_H,
        measure_candidates,
    )
    import math

    # Generar imagen de sonda sintética para referencia visual
    n = len(candidates)
    cols = GRID_COLS
    rows = math.ceil(n / cols)
    patch_px = probe.patch_px

    grid_w = cols * (patch_px + GAP_PX) - GAP_PX + 2 * BORDER_PX
    grid_h = rows * (patch_px + LABEL_H + GAP_PX) - GAP_PX + 2 * BORDER_PX
    grid_x0 = max(0, (MONITOR_W - grid_w) // 2)
    grid_y0 = max(0, (MONITOR_H - grid_h) // 2)

    accum: list[list] = [[] for _ in range(n)]
    margin = max(1, patch_px // 5)

    for sample_idx in range(n_samples):
        try:
            frame = camera.capture()
        except Exception:
            time.sleep(0.1)
            continue

        fh, fw = frame.shape[:2]
        if fw != MONITOR_W or fh != MONITOR_H:
            frame_scaled = cv2.resize(frame, (MONITOR_W, MONITOR_H))
        else:
            frame_scaled = frame

        for i in range(n):
            row = i // cols
            col = i % cols
            px0 = grid_x0 + BORDER_PX + col * (patch_px + GAP_PX)
            py0 = grid_y0 + BORDER_PX + row * (patch_px + LABEL_H + GAP_PX)
            region = frame_scaled[
                py0 + margin: py0 + patch_px - margin,
                px0 + margin: px0 + patch_px - margin,
            ]
            if region.size == 0:
                continue
            bgr = region.mean(axis=(0,1))
            h, s, v = _rgb_to_hsv((int(bgr[2]), int(bgr[1]), int(bgr[0])))
            accum[i].append((h, s, v))

        # Preview
        preview = cv2.resize(frame, (700, 440))
        pct = int((sample_idx + 1) / n_samples * 100)
        bar = "█" * (pct // 3)
        cv2.putText(preview, f"Midiendo... {pct}% [{bar:<33}]",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,128), 2)
        cv2.imshow("RX-FEED", preview)
        cv2.waitKey(50)
        time.sleep(0.05)

    # Promediar
    observed = []
    for patch_samples in accum:
        if not patch_samples:
            observed.append((0.0, 0.0, 0.0))
        else:
            h = sum(s[0] for s in patch_samples) / len(patch_samples)
            s_ = sum(s[1] for s in patch_samples) / len(patch_samples)
            v = sum(s[2] for s in patch_samples) / len(patch_samples)
            observed.append((h, s_, v))

    return observed


# ══════════════════════════════════════════════════════════════════════════════
# MODO AMBOS (prueba local en una sola máquina)
# ══════════════════════════════════════════════════════════════════════════════

def run_both(camera_id: int = CAMERA_ID) -> dict:
    """
    Modo de prueba local: emisor y receptor en la misma máquina.
    El emisor muestra en Monitor 1, el receptor lee con la cámara.
    No hay intercambio de QR B/N — la respuesta se genera y aplica localmente.
    """
    candidates = generate_candidates(N_CANDIDATES)
    camera = OpenCVCamera(device_id=camera_id, width=1280, height=720)
    camera.open()

    cv2.namedWindow("TX-PROBE", cv2.WINDOW_NORMAL)
    cv2.moveWindow("TX-PROBE", MONITOR_TX_X, 0)
    cv2.setWindowProperty("TX-PROBE", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cv2.namedWindow("RX-PREVIEW", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RX-PREVIEW", 700, 450)

    print("\n[BOTH] Mostrando imagen de sonda...")
    probe_img = generate_probe_image(candidates, PATCH_PX, MONITOR_W, MONITOR_H)
    cv2.imshow("TX-PROBE", probe_img)
    cv2.waitKey(500)   # dar tiempo al monitor para renderizar

    print(f"[BOTH] Midiendo {N_CANDIDATES} colores ({N_SAMPLES} muestras)...")

    probe = CalProbeFrame(src_mac=MY_MAC, dst_mac=PEER_MAC,
                          n_candidates=N_CANDIDATES, patch_px=PATCH_PX, duration_ms=3000)
    observed = _measure_with_preview(camera, candidates, probe, N_SAMPLES)

    print("\n[BOTH] Construyendo matriz de confusión...")
    confusion = build_confusion_matrix(observed)
    _print_confusion_summary(confusion, candidates)

    selected = select_optimal_palette(candidates, observed, confusion, target_n=16)
    n_sel = len(selected)
    cd = 3 if n_sel >= 16 else 2 if n_sel >= 8 else 1 if n_sel >= 4 else 0

    response = CalResponseFrame(
        src_mac=MY_MAC, dst_mac=PEER_MAC,
        selected=selected, confusion=confusion,
        recommended_cd=cd, interval_ms=150,
    )

    _print_tx_summary(response, candidates)

    # Mostrar QR de respuesta
    response_qr = encode_response_as_qr(response, box_size=7)
    resp_canvas = _center_on_black(response_qr, 700, 700,
                                   title=f"Paleta negociada | {n_sel} colores | CD={cd}")
    cv2.imshow("RX-PREVIEW", resp_canvas)
    print("\n[BOTH] Presiona Q para continuar")
    while cv2.waitKey(100) & 0xFF not in (ord('q'), 27):
        pass

    camera.close()
    cv2.destroyAllWindows()
    return _save_palette(candidates, response)


# ══════════════════════════════════════════════════════════════════════════════
# UTILIDADES
# ══════════════════════════════════════════════════════════════════════════════

def _center_on_black(
    img: np.ndarray,
    w: int, h: int,
    title: str = "",
) -> np.ndarray:
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    ih, iw = img.shape[:2]
    scale = min((w - 20) / iw, (h - 50) / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    scaled = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_NEAREST)
    y0 = (h - nh) // 2
    x0 = (w - nw) // 2
    canvas[y0: y0+nh, x0: x0+nw] = scaled
    if title:
        cv2.putText(canvas, title, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
    return canvas


def _print_confusion_summary(
    confusion: list[list[float]],
    candidates: list[tuple[int,int,int]],
) -> None:
    n = len(confusion)
    print(f"\n[CAL] Matriz de confusión ({n}×{n}):")
    confused_pairs = []
    for i in range(n):
        for j in range(n):
            if i != j and confusion[i][j] > 0.1:
                confused_pairs.append((i, j, confusion[i][j]))
    if confused_pairs:
        confused_pairs.sort(key=lambda x: -x[2])
        print(f"  Pares confundibles (prob > 10%):")
        for i, j, p in confused_pairs[:10]:
            ri, gi, bi = candidates[i]
            rj, gj, bj = candidates[j]
            print(f"    color {i:02d} ({ri:3d},{gi:3d},{bi:3d}) → "
                  f"color {j:02d} ({rj:3d},{gj:3d},{bj:3d})  p={p:.2f}")
    else:
        print("  Sin confusiones significativas ✓")


def _print_tx_summary(
    response: CalResponseFrame,
    candidates: list[tuple[int,int,int]],
) -> None:
    print(f"\n{'═'*50}")
    print(f"  PALETA NEGOCIADA")
    print(f"{'═'*50}")
    print(f"  Colores seleccionados : {len(response.selected)}")
    print(f"  COLOR_DEPTH           : {response.recommended_cd} "
          f"({2**(response.recommended_cd+1)} colores efectivos)")
    print(f"  Intervalo recomendado : {response.interval_ms}ms")
    print(f"\n  Índices y colores RGB:")
    for rank, idx in enumerate(response.selected):
        r, g, b = candidates[idx]
        print(f"    [{rank:02d}] candidato #{idx:02d}  RGB({r:3d},{g:3d},{b:3d})")
    print(f"{'═'*50}")


def _save_palette(
    candidates: list[tuple[int,int,int]],
    response: CalResponseFrame,
) -> dict:
    data = {
        "recommended_cd" : response.recommended_cd,
        "interval_ms"    : response.interval_ms,
        "n_colors"       : len(response.selected),
        "selected_indices": response.selected,
        "palette_rgb"    : [list(candidates[i]) for i in response.selected],
        "confusion_diagonal": [
            round(response.confusion[i][i], 3)
            for i in range(len(response.confusion))
        ],
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n[CAL] Paleta guardada en {OUTPUT_FILE}")
    return data


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QR-NET Color Negotiation")
    parser.add_argument("--mode", choices=["tx", "rx", "both"], default="both",
                        help="tx=emisor  rx=receptor  both=prueba local")
    parser.add_argument("--camera", type=int, default=CAMERA_ID,
                        help="device_id de la cámara")
    parser.add_argument("--monitor-x", type=int, default=MONITOR_TX_X,
                        help="offset X del monitor TX")
    args = parser.parse_args()

    MONITOR_TX_X = args.monitor_x

    print("╔══════════════════════════════════════════╗")
    print("║   QR-NET — Negociación de Color          ║")
    print(f"║   Modo: {args.mode.upper():<33}║")
    print(f"║   Cámara: device_id={args.camera:<21}║")
    print("╚══════════════════════════════════════════╝")

    if args.mode == "tx":
        result = run_tx(args.camera)
    elif args.mode == "rx":
        result = run_rx(args.camera)
    else:
        result = run_both(args.camera)

    if result:
        print(f"\n✓ Listo. Paleta guardada con {result.get('n_colors')} colores.")
        print(f"  Usa paleta_negociada.json en camera_test.py para transmitir.")
