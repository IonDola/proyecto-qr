"""
camera_test.py
Prueba completa del pipeline de Capa 1 con cámara real y dos pantallas.

Uso:
    python camera_test.py

Ventana TX : muestra la grilla QR — arrástrala a la Pantalla 1
Ventana RX : muestra el feed de la cámara con overlay — déjala en Pantalla 2
             (o ábrela en la pantalla donde está la cámara apuntando a TX)

Controles:
    Q       → salir
    C       → cambiar color depth (2 → 4 → 8 → 16 colores)
    +/-     → aumentar/reducir intervalo entre frames (50ms steps)
    S       → guardar screenshot del frame actual
    ESPACIO → pausar/reanudar TX
"""
from __future__ import annotations
import sys
import os
import time
import threading
import queue
import math
from dataclasses import dataclass, field

import cv2
import numpy as np


# ── path del proyecto ─────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
NEGOTIATED_FILE = "transmision/negotiation/paleta_negociada.json"

from transmision.implementation.color_palette import ColorPalette
from transmision.implementation.grid import Grid64Codec, SILENCE_MODULES
from transmision.implementation.queue import FifoFrameQueue, DropPolicy

# ── configuración ─────────────────────────────────────────────────────────────

CAMERA_ID       = 700     
N_COLORS        = 2      # color depth inicial (se sobreescribe si hay paleta negociada)
INTERVAL_MS     = 200     # ms entre frames QR
FIFO_SIZE       = 32
MODULE_PX       = 10      # tamaño de módulo — la imagen mide (64+4)*10 = 680px

# ── cargar paleta negociada si existe ────────────────────────────────────────
_negotiated_palette: ColorPalette | None = None
if os.path.exists(NEGOTIATED_FILE):
    try:
        _negotiated_palette = ColorPalette.from_negotiated(NEGOTIATED_FILE)
        import json as _json
        _ndata = _json.load(open(NEGOTIATED_FILE))
        N_COLORS  = _ndata.get("n_colors", 16)
        INTERVAL_MS = _ndata.get("interval_ms", INTERVAL_MS)
        print(f"[CAL] Paleta negociada cargada: {N_COLORS} colores, {INTERVAL_MS}ms/frame")
    except Exception as e:
        print(f"[CAL] No se pudo cargar paleta negociada: {e} — usando paleta teórica")
        _negotiated_palette = None
else:
    print(f"[CAL] Sin paleta negociada — usando distribución HSV teórica ({N_COLORS} colores)")
    print(f"[CAL] Ejecuta negotiation_session.py --mode both para negociar primero")

# Payloads de prueba en loop
TEST_PAYLOADS = [
    b"QR-NET CAMERA TEST 2026",
    #b"HOLA DESDE CAPA 1",
    #b"REDES TEC ABRIL 2026",
    #b"TRANSMISION OPTICA OK",
]

# ── estadísticas ──────────────────────────────────────────────────────────────

@dataclass
class Stats:
    frames_sent:     int   = 0
    frames_decoded:  int   = 0
    frames_error:    int   = 0
    frames_captured: int   = 0
    start_time:      float = field(default_factory=time.time)
    last_decoded:    bytes = b""
    last_error:      str   = ""

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    @property
    def tx_fps(self) -> float:
        return self.frames_sent / max(self.elapsed, 0.001)

    @property
    def rx_fps(self) -> float:
        return self.frames_decoded / max(self.elapsed, 0.001)

    @property
    def error_rate(self) -> float:
        total = self.frames_decoded + self.frames_error
        return self.frames_error / max(total, 1)


stats = Stats()
stats_lock = threading.Lock()

# ── estado global ─────────────────────────────────────────────────────────────

current_n_colors  = N_COLORS
current_interval  = INTERVAL_MS
paused            = False
frame_queue: FifoFrameQueue | None = None


# ══════════════════════════════════════════════════════════════════════════════
# HILO TX — genera y muestra grillas QR
# ══════════════════════════════════════════════════════════════════════════════

# Resolución del monitor TX — ajustar si el monitor 1 tiene otra resolución
TX_MONITOR_W = 1920
TX_MONITOR_H = 1080


def tx_loop(grid: Grid64Codec, running: threading.Event) -> None:
    global current_n_colors, current_interval, paused

    payload_idx = 0

    # Crear ventana en pantalla completa en monitor 1
    # En Windows con dos monitores, mover la ventana a x=-TX_MONITOR_W la pone en monitor 1
    cv2.namedWindow("QR-NET TX", cv2.WINDOW_NORMAL)
    cv2.moveWindow("QR-NET TX", TX_MONITOR_W, 0)   # monitor izquierdo
    cv2.setWindowProperty("QR-NET TX", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while running.is_set():
        if paused:
            blank = np.zeros((TX_MONITOR_H, TX_MONITOR_W, 3), dtype=np.uint8)
            cv2.putText(blank, "PAUSADO — SPACE para reanudar",
                        (TX_MONITOR_W//4, TX_MONITOR_H//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
            cv2.imshow("QR-NET TX", blank)
            cv2.waitKey(100)
            continue

        # Usar paleta negociada si está disponible, sino la teórica
        palette = (_negotiated_palette
                   if _negotiated_palette is not None
                   else ColorPalette(n_colors=current_n_colors))
        payload = TEST_PAYLOADS[payload_idx % len(TEST_PAYLOADS)]

        try:
            img = grid.encode_grid(payload, palette)
            _, bpc_enc = grid._read_format_info(img)  # verificar que encode escribe bien
            print(f"[DEBUG TX] bpc_escrito={bpc_enc}  palette_bpc={palette.bits_per_cell}")

            # Borde negro generoso para separar la grilla del fondo
            border = 40
            img = cv2.copyMakeBorder(img, border, border, border, border,
                                     cv2.BORDER_CONSTANT, value=(0, 0, 0))

            # Escalar al tamaño máximo que cabe en el monitor manteniendo aspecto
            h_img, w_img = img.shape[:2]
            scale = min((TX_MONITOR_W - 20) / w_img, (TX_MONITOR_H - 80) / h_img)
            new_w = int(w_img * scale)
            new_h = int(h_img * scale)
            scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            # Canvas negro del tamaño exacto del monitor
            canvas = np.full((TX_MONITOR_H, TX_MONITOR_W, 3), 255, dtype=np.uint8)
            y0 = (TX_MONITOR_H - new_h) // 2
            x0 = (TX_MONITOR_W - new_w) // 2
            canvas[y0:y0+new_h, x0:x0+new_w] = scaled

            _draw_tx_overlay(canvas, payload, current_n_colors, current_interval)
            cv2.imshow("QR-NET TX", canvas)

        except Exception as e:
            with stats_lock:
                stats.last_error = f"TX encode error: {e}"

        with stats_lock:
            stats.frames_sent += 1

        payload_idx += 1
        key = cv2.waitKey(current_interval)
        _handle_key(key, running)


def _draw_tx_overlay(img: np.ndarray, payload: bytes, n_colors: int, interval: int) -> None:
    h, w = img.shape[:2]
    # Barra inferior semitransparente
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - 60), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    txt = f"TX | {n_colors} colores | {interval}ms | {payload.decode(errors='replace')}"
    cv2.putText(img, txt, (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    with stats_lock:
        fps_txt = f"Enviados: {stats.frames_sent} | {stats.tx_fps:.1f} fps"
    cv2.putText(img, fps_txt, (10, h - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)


# ══════════════════════════════════════════════════════════════════════════════
# HILO PRODUCTOR — captura cámara → FIFO
# ══════════════════════════════════════════════════════════════════════════════

def producer_loop(cap: cv2.VideoCapture, fq: FifoFrameQueue,
                  running: threading.Event) -> None:
    prev = None
    while running.is_set():
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        with stats_lock:
            stats.frames_captured += 1
        # Solo encolar si el frame cambió
        if prev is None or _frame_changed(prev, frame):
            fq.put(frame)
            prev = frame.copy()


def _frame_changed(prev: np.ndarray, curr: np.ndarray, threshold: float = 6.0) -> bool:
    p = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    c = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    return float(cv2.absdiff(p, c).mean()) > threshold


# ══════════════════════════════════════════════════════════════════════════════
# HILO CONSUMIDOR — decodifica frames de la FIFO
# ══════════════════════════════════════════════════════════════════════════════
def _debug_align(grid, image):
    h, w = image.shape[:2]

    # Reducir tamaño si es muy grande
    if w > 1280:
        scale = 1280 / w
        image = cv2.resize(image, (1280, int(h * scale)))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold binario
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Invertir SIN morphology close
    closed = cv2.bitwise_not(binary)

    contours, _ = cv2.findContours(
        closed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    img_area = image.shape[0] * image.shape[1]

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )

    print(f"[DEBUG ALIGN] frame={w}x{h}  img_area={img_area}  contornos={len(contours)}")

    dbg = image.copy()

    best_quad = None
    best_area = 0

    for i, cnt in enumerate(contours[:20]):

        area = cv2.contourArea(cnt)
        pct = area / img_area * 100

        # Ignorar ruido pequeño
        if pct < 1:
            continue

        # Ignorar regiones absurdamente grandes
        if pct > 70:
            continue

        peri = cv2.arcLength(cnt, True)

        approx = cv2.approxPolyDP(
            cnt,
            0.03 * peri,
            True
        )

        print(f"  [{i}] area={area:.0f} ({pct:.1f}%)  vertices={len(approx)}")

        # Dibujar todos los contornos válidos
        cv2.drawContours(dbg, [approx], -1, (0, 255, 0), 2)

        # Queremos cuadriláteros
        if len(approx) != 4:
            continue

        if area > best_area:
            best_area = area
            best_quad = approx

    # Dibujar el mejor cuadrilátero
    if best_quad is not None:
        cv2.drawContours(dbg, [best_quad], -1, (0, 0, 255), 4)

    cv2.imshow("DEBUG CONTOURS", dbg)

    # Continuar usando tu align original
    return grid._align_grid(image)

def consumer_loop(grid: Grid64Codec, fq: FifoFrameQueue,
                  running: threading.Event, decoded_q: queue.Queue) -> None:
    while running.is_set():
        try:
            frame_img = fq.get(timeout=1.0)
        except queue.Empty:
            continue

        palette = (_negotiated_palette
                   if _negotiated_palette is not None
                   else ColorPalette(n_colors=current_n_colors))

        # Intentar alinear y decodificar
        aligned = _debug_align(grid, frame_img)

        if aligned is None:
            with stats_lock:
                stats.frames_error += 1
                stats.last_error = "Grilla no detectada"

            decoded_q.put(("error", frame_img, "Grilla no detectada"))
            continue

        payload_len, bpc = grid._read_format_info(aligned)

        print(
            f"[DEBUG] payload_len={payload_len}  "
            f"bpc_leido={bpc}  "
            f"codec_bpc={palette.bits_per_cell}  "
            f"aligned_shape={aligned.shape}"
        )
        try:
            cal_patch = grid._extract_cal_patch(aligned)
            palette.calibrate(cal_patch)
            raw = grid.decode_grid(aligned, palette)

            with stats_lock:
                stats.frames_decoded += 1
                stats.last_decoded = raw
                stats.last_error = ""

            decoded_q.put(("ok", frame_img, raw))

        except Exception as e:
            with stats_lock:
                stats.frames_error += 1
                stats.last_error = str(e)
            decoded_q.put(("error", frame_img, str(e)))


# ══════════════════════════════════════════════════════════════════════════════
# VENTANA RX — muestra feed + overlay de decodificación
# ══════════════════════════════════════════════════════════════════════════════

def rx_display_loop(cap: cv2.VideoCapture, decoded_q: queue.Queue,
                    running: threading.Event) -> None:
    cv2.namedWindow("QR-NET RX", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("QR-NET RX", 900, 600)

    last_frame = None
    last_status = ("waiting", b"", "")

    while running.is_set():
        # Leer frame más reciente de la cámara para display en vivo
        ok, live = cap.read()
        if ok:
            last_frame = live.copy()

        # Consumir todos los resultados pendientes (nos quedamos con el último)
        while True:
            try:
                status, frame_img, data = decoded_q.get_nowait()
                last_status = (status, data if isinstance(data, bytes) else b"", 
                               "" if isinstance(data, bytes) else data)
            except queue.Empty:
                break

        if last_frame is not None:
            display = last_frame.copy()
            _draw_rx_overlay(display, last_status)
            cv2.imshow("QR-NET RX", display)

        key = cv2.waitKey(30)
        _handle_key(key, running)


def _draw_rx_overlay(img: np.ndarray, status: tuple) -> None:
    h, w = img.shape[:2]
    st, decoded_bytes, error_msg = status

    # Color del borde según estado
    border_color = (0, 255, 0) if st == "ok" else (0, 0, 255) if st == "error" else (128, 128, 128)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), border_color, 4)

    # Panel inferior
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - 90), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

    with stats_lock:
        s = Stats.__new__(Stats)
        s.__dict__.update(stats.__dict__)

    # Línea 1: estado
    if st == "ok":
        txt1 = f"OK  |  {decoded_bytes.decode(errors='replace')}"
        color1 = (0, 255, 0)
    elif st == "error":
        txt1 = f"ERROR: {error_msg[:60]}"
        color1 = (0, 100, 255)
    else:
        txt1 = "Esperando grilla QR..."
        color1 = (200, 200, 200)

    cv2.putText(img, txt1, (10, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color1, 2)

    # Línea 2: métricas
    txt2 = (f"RX: {s.frames_decoded} ok / {s.frames_error} err  |  "
            f"{s.rx_fps:.1f} fps  |  "
            f"Error rate: {s.error_rate * 100:.1f}%  |  "
            f"Queue: {frame_queue.qsize() if frame_queue else 0}")
    cv2.putText(img, txt2, (10, h - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # Línea 3: config
    txt3 = (f"Capturados: {s.frames_captured}  |  "
            f"{current_n_colors} colores  |  "
            f"{current_interval}ms/frame  |  "
            f"[Q]salir [C]colores [+/-]intervalo [SPACE]pausa")
    cv2.putText(img, txt3, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)


# ══════════════════════════════════════════════════════════════════════════════
# CONTROL DE TECLADO
# ══════════════════════════════════════════════════════════════════════════════

def _handle_key(key: int, running: threading.Event) -> None:
    global current_n_colors, current_interval, paused

    if key == ord('q') or key == 27:   # Q o ESC
        running.clear()
    elif key == ord('c'):              # cambiar color depth
        cycle = {2: 4, 4: 8, 8: 16, 16: 2}
        current_n_colors = cycle[current_n_colors]
        print(f"[CONFIG] Color depth → {current_n_colors} colores ({int(math.log2(current_n_colors))} bits/módulo)")
    elif key == ord('+') or key == ord('='):
        current_interval = min(current_interval + 50, 2000)
        print(f"[CONFIG] Intervalo → {current_interval}ms")
    elif key == ord('-'):
        current_interval = max(current_interval - 50, 50)
        print(f"[CONFIG] Intervalo → {current_interval}ms")
    elif key == ord(' '):
        paused = not paused
        print(f"[CONFIG] {'PAUSADO' if paused else 'REANUDADO'}")
    elif key == ord('s'):
        fname = f"screenshot_{int(time.time())}.png"
        print(f"[INFO] Screenshot guardado: {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    global frame_queue

    print("=" * 60)
    print("  QR-NET — Prueba de cámara real")
    print("=" * 60)
    print(f"  Cámara TX display : ventana 'QR-NET TX'  → mover a Pantalla 1")
    print(f"  Cámara RX feed    : ventana 'QR-NET RX'  → dejar en Pantalla 2")
    print(f"  Color depth inicial: {N_COLORS} colores ({int(math.log2(N_COLORS))} bits/módulo)")
    print(f"  Intervalo inicial  : {INTERVAL_MS}ms por frame")
    print(f"  Módulo px          : {MODULE_PX}px")
    print()
    print("  Controles: Q=salir  C=colores  +/-=intervalo  SPACE=pausa")
    print("=" * 60)

    # ── abrir cámara ──────────────────────────────────────────────────────────
    cap_rx = cv2.VideoCapture(CAMERA_ID)
    if not cap_rx.isOpened():
        print(f"[ERROR] No se pudo abrir cámara device_id={CAMERA_ID}")
        print("        Prueba cambiar CAMERA_ID a 0 en este script.")
        return

    cap_rx.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap_rx.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap_rx.set(cv2.CAP_PROP_FPS, 30)

    w = int(cap_rx.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_rx.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_cam = cap_rx.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] Cámara abierta: {w}×{h} @ {fps_cam:.0f}fps  (device_id={CAMERA_ID})")

    # ── inicializar componentes ───────────────────────────────────────────────
    grid = Grid64Codec(module_px=MODULE_PX)
    frame_queue = FifoFrameQueue(maxsize=FIFO_SIZE, policy=DropPolicy.DROP_OLDEST)
    decoded_q: queue.Queue = queue.Queue(maxsize=16)

    # ── segunda cámara para feed de display (puede ser la misma) ─────────────
    # Para el display en vivo de RX abrimos una segunda instancia
    cap_display = cv2.VideoCapture(CAMERA_ID)
    cap_display.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap_display.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # ── hilos ─────────────────────────────────────────────────────────────────
    running = threading.Event()
    running.set()

    threads = [
        threading.Thread(target=tx_loop,
                         args=(grid, running),
                         name="TX", daemon=True),
        threading.Thread(target=producer_loop,
                         args=(cap_rx, frame_queue, running),
                         name="Producer", daemon=True),
        threading.Thread(target=consumer_loop,
                         args=(grid, frame_queue, running, decoded_q),
                         name="Consumer", daemon=True),
    ]

    for t in threads:
        t.start()
    print("[INFO] Hilos iniciados — mueve 'QR-NET TX' a la Pantalla 1\n")

    # El display RX corre en el hilo principal (OpenCV requiere GUI en main thread)
    try:
        rx_display_loop(cap_display, decoded_q, running)
    except KeyboardInterrupt:
        running.clear()
    finally:
        running.clear()
        for t in threads:
            t.join(timeout=2.0)
        cap_rx.release()
        cap_display.release()
        cv2.destroyAllWindows()

        # ── resumen final ─────────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("  Resumen de la sesión")
        print("=" * 60)
        with stats_lock:
            s = stats
        print(f"  Duración          : {s.elapsed:.1f}s")
        print(f"  Frames enviados   : {s.frames_sent}  ({s.tx_fps:.2f} fps)")
        print(f"  Frames capturados : {s.frames_captured}")
        print(f"  Frames decodificados: {s.frames_decoded}  ({s.rx_fps:.2f} fps)")
        print(f"  Frames con error  : {s.frames_error}")
        print(f"  Tasa de error     : {s.error_rate * 100:.1f}%")
        if s.last_decoded:
            print(f"  Último decodificado: {s.last_decoded.decode(errors='replace')}")
        print("=" * 60)


if __name__ == "__main__":
    main()