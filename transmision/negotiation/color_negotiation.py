"""
capa1/color_negotiation.py
Negociación de paleta de colores basada en medición real del canal.

Flujo completo:
  1. Emisor genera N colores candidatos (más de los que va a usar)
  2. Emisor muestra cada candidato como patch grande en pantalla
  3. Receptor captura, mide color real de cada patch, construye
     la matriz de confusión C[i][j]
  4. Receptor selecciona el subconjunto óptimo de K colores donde
     la confusión es mínima
  5. Receptor responde con CalResponseFrame (QR B/N estándar)
  6. Emisor lee la respuesta y construye su paleta custom

La selección óptima es un problema de Maximum Independent Set
sobre el grafo de confusión — NP-hard en general, pero con
N≤64 un greedy funciona perfectamente.
"""
from __future__ import annotations
import json
import math
import time
from dataclasses import dataclass

import cv2
import numpy as np
import qrcode
import base64
from PIL import Image
from pyzbar import pyzbar

import os
import sys
from common.exceptions import DecompressionError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from frames import CalProbeFrame, CalResponseFrame
from implementation.color_palette import ColorPalette, _rgb_to_hsv, _hsv_distance
from implementation.compressor import ZstdCompressor


# ── Parámetros de la sonda ────────────────────────────────────────────────────

N_CANDIDATES   = 32     # colores candidatos que muestra el emisor
N_SAMPLES      = 15     # capturas por color para promediar
PATCH_PX       = 120    # tamaño del patch en pantalla (px)
GRID_COLS      = 8      # columnas en la pantalla de sonda
BORDER_PX      = 60     # margen negro alrededor
GAP_PX         = 8      # separación entre patches
LABEL_H        = 22     # altura del label numérico

# Umbral de confusión: distancia HSV bajo la cual dos colores
# se consideran indistinguibles por el canal
CONFUSION_THRESHOLD = 0.18


# ── Generación de candidatos ──────────────────────────────────────────────────

def generate_candidates(n: int = N_CANDIDATES) -> list[tuple[int, int, int]]:
    """
    Genera N colores candidatos distribuidos en el espacio HSV.
    Usa más colores que los que se necesitan para que el receptor
    pueda elegir los mejores según su canal.

    Estrategia: grilla 2D en (Hue × Saturation) con Value fijo alto.
    """
    import colorsys
    candidates: list[tuple[int, int, int]] = []

    # Distribuir en hue con variaciones de saturación
    hue_steps = max(8, n // 2)
    sat_levels = [0.95, 0.70]   # saturación alta y media

    for sat in sat_levels:
        for i in range(hue_steps):
            hue = i / hue_steps
            r, g, b = colorsys.hsv_to_rgb(hue, sat, 0.90)
            candidates.append((int(r * 255), int(g * 255), int(b * 255)))
            if len(candidates) >= n:
                break
        if len(candidates) >= n:
            break

    # Completar si faltan con variaciones de value
    if len(candidates) < n:
        for i in range(n - len(candidates)):
            hue = i / (n - len(candidates))
            r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.60)
            candidates.append((int(r * 255), int(g * 255), int(b * 255)))

    return candidates[:n]


# ── Imagen de sonda del emisor ────────────────────────────────────────────────

def generate_probe_image(
    candidates: list[tuple[int, int, int]],
    patch_px: int = PATCH_PX,
    monitor_w: int = 1920,
    monitor_h: int = 1080,
) -> np.ndarray:
    """
    Genera la imagen de sonda con todos los colores candidatos.
    Cada color aparece como un patch grande con su índice visible.
    Fondo negro garantizado para separación visual.
    """
    n = len(candidates)
    cols = GRID_COLS
    rows = math.ceil(n / cols)

    grid_w = cols * (patch_px + GAP_PX) - GAP_PX + 2 * BORDER_PX
    grid_h = rows * (patch_px + LABEL_H + GAP_PX) - GAP_PX + 2 * BORDER_PX

    canvas = np.zeros((monitor_h, monitor_w, 3), dtype=np.uint8)
    grid   = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    for i, (r, g, b) in enumerate(candidates):
        row = i // cols
        col = i % cols
        x0 = BORDER_PX + col * (patch_px + GAP_PX)
        y0 = BORDER_PX + row * (patch_px + LABEL_H + GAP_PX)

        # Patch de color (BGR)
        grid[y0: y0 + patch_px, x0: x0 + patch_px] = (b, g, r)

        # Borde blanco para delimitar
        cv2.rectangle(grid, (x0-1, y0-1), (x0+patch_px, y0+patch_px),
                      (255, 255, 255), 1)

        # Índice
        cv2.putText(grid, f"{i:02d}",
                    (x0 + patch_px//2 - 12, y0 + patch_px + LABEL_H - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    # Título
    cv2.putText(grid, f"QR-NET CAL PROBE | {n} candidatos | midiendo canal...",
                (BORDER_PX, BORDER_PX - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

    # Centrar en canvas
    y0c = max(0, (monitor_h - grid_h) // 2)
    x0c = max(0, (monitor_w - grid_w) // 2)
    h_fit = min(grid_h, monitor_h - y0c)
    w_fit = min(grid_w, monitor_w - x0c)
    canvas[y0c: y0c + h_fit, x0c: x0c + w_fit] = grid[:h_fit, :w_fit]

    return canvas


# ── Medición del receptor ─────────────────────────────────────────────────────

def measure_candidates(
    camera,                              # ICameraInterface
    candidates: list[tuple[int,int,int]],
    probe_img: np.ndarray,
    n_samples: int = N_SAMPLES,
    patch_px: int = PATCH_PX,
    monitor_w: int = 1920,
    monitor_h: int = 1080,
) -> list[tuple[float, float, float]]:
    """
    Captura N_SAMPLES frames mientras la imagen de sonda está en pantalla
    y mide el color promedio observado de cada patch.

    Retorna lista de colores observados en HSV, uno por candidato.
    """
    n = len(candidates)
    cols = GRID_COLS
    rows = math.ceil(n / cols)

    # Calcular posición de cada patch en el canvas
    grid_w = cols * (patch_px + GAP_PX) - GAP_PX + 2 * BORDER_PX
    grid_h = rows * (patch_px + LABEL_H + GAP_PX) - GAP_PX + 2 * BORDER_PX
    grid_x0 = max(0, (monitor_w - grid_w) // 2)
    grid_y0 = max(0, (monitor_h - grid_h) // 2)

    # Acumular muestras HSV por patch
    accum: list[list[tuple[float,float,float]]] = [[] for _ in range(n)]

    for _ in range(n_samples):
        frame = camera.capture()
        fh, fw = frame.shape[:2]
        if fw != monitor_w or fh != monitor_h:
            frame = cv2.resize(frame, (monitor_w, monitor_h))

        margin = max(1, patch_px // 5)

        for i in range(n):
            row = i // cols
            col = i % cols
            px0 = grid_x0 + BORDER_PX + col * (patch_px + GAP_PX)
            py0 = grid_y0 + BORDER_PX + row * (patch_px + LABEL_H + GAP_PX)

            region = frame[
                py0 + margin : py0 + patch_px - margin,
                px0 + margin : px0 + patch_px - margin,
            ]
            if region.size == 0:
                continue

            bgr = region.mean(axis=(0, 1))
            r, g, b = float(bgr[2]), float(bgr[1]), float(bgr[0])
            h, s, v = _rgb_to_hsv((int(r), int(g), int(b)))
            accum[i].append((h, s, v))

        time.sleep(0.05)

    # Promediar muestras por patch
    observed: list[tuple[float, float, float]] = []
    for patch_samples in accum:
        if not patch_samples:
            observed.append((0.0, 0.0, 0.0))
        else:
            h = sum(s[0] for s in patch_samples) / len(patch_samples)
            s_ = sum(s[1] for s in patch_samples) / len(patch_samples)
            v = sum(s[2] for s in patch_samples) / len(patch_samples)
            observed.append((h, s_, v))

    return observed


# ── Matriz de confusión ───────────────────────────────────────────────────────

def build_confusion_matrix(
    observed: list[tuple[float, float, float]],
    threshold: float = CONFUSION_THRESHOLD,
) -> list[list[float]]:
    """
    Construye la matriz de confusión C[i][j] ∈ [0,1].

    C[i][j] = probabilidad de que el color i sea clasificado como j
    usando distancia mínima en HSV.

    Implementación:
      Para cada par (i,j), si dist(obs[i], obs[j]) < threshold
      asignamos una probabilidad de confusión proporcional a la
      proximidad. C[i][i] = 1 - sum(C[i][j] para j≠i).
    """
    n = len(observed)
    C = [[0.0] * n for _ in range(n)]

    for i in range(n):
        hi, si, vi = observed[i]
        distances = []
        for j in range(n):
            if i == j:
                distances.append(0.0)
            else:
                hj, sj, vj = observed[j]
                distances.append(_hsv_distance(hi, si, vi, hj, sj, vj))

        # Clasificación por distancia mínima (como hace decode())
        min_dist = min(d for k, d in enumerate(distances) if k != i)
        nearest = distances.index(min_dist)

        if min_dist < threshold:
            # Confusión proporcional a la proximidad
            confusion_prob = max(0.0, 1.0 - min_dist / threshold)
            C[i][nearest] = round(confusion_prob, 3)
            C[i][i] = round(1.0 - confusion_prob, 3)
        else:
            C[i][i] = 1.0

    return C


# ── Selección óptima de colores ───────────────────────────────────────────────

def select_optimal_palette(
    candidates: list[tuple[int, int, int]],
    observed: list[tuple[float, float, float]],
    confusion: list[list[float]],
    target_n: int = 16,
    confusion_threshold: float = CONFUSION_THRESHOLD,
) -> list[int]:
    """
    Selecciona los índices de los mejores colores del conjunto candidato.

    Algoritmo greedy de Maximum Independent Set sobre el grafo de confusión:
      1. Construir grafo G donde hay arista (i,j) si C[i][j] > 0 o C[j][i] > 0
      2. Ordenar nodos por grado ascendente (menos confundibles primero)
      3. Seleccionar greedy: agregar nodo si no tiene vecinos ya seleccionados
      4. Si el resultado es menor que target_n, relajar el umbral y repetir

    Retorna lista de índices en el orden original de candidates.
    """
    n = len(candidates)

    # Construir grafo de confusión
    def build_graph(thresh: float) -> list[set[int]]:
        adj: list[set[int]] = [set() for _ in range(n)]
        for i in range(n):
            hi, si, vi = observed[i]
            for j in range(i + 1, n):
                hj, sj, vj = observed[j]
                if _hsv_distance(hi, si, vi, hj, sj, vj) < thresh:
                    adj[i].add(j)
                    adj[j].add(i)
        return adj

    # Greedy MIS
    def greedy_mis(adj: list[set[int]]) -> list[int]:
        # Ordenar por grado ascendente
        order = sorted(range(n), key=lambda x: len(adj[x]))
        selected: list[int] = []
        excluded: set[int] = set()
        for node in order:
            if node not in excluded:
                selected.append(node)
                excluded.update(adj[node])
                if len(selected) >= target_n:
                    break
        return sorted(selected)

    # Intentar con umbral estricto, relajar si no hay suficientes
    for multiplier in [1.0, 1.3, 1.6, 2.0]:
        adj = build_graph(confusion_threshold * multiplier)
        selected = greedy_mis(adj)
        if len(selected) >= min(target_n, n // 2):
            break

    # Si aún tenemos menos que target_n, agregar los menos confusos
    if len(selected) < target_n:
        remaining = [i for i in range(n) if i not in selected]
        # Agregar los que tienen menor confusión total
        remaining.sort(key=lambda i: sum(confusion[i]))
        for r in remaining:
            if len(selected) >= target_n:
                break
            selected.append(r)
        selected.sort()

    return selected[:target_n]


# ── QR estándar B/N para CalResponse ─────────────────────────────────────────

def encode_response_as_qr(
    response: CalResponseFrame,
    box_size: int = 6,
    border: int = 4,
) -> np.ndarray:
    """
    Codifica un CalResponseFrame como QR estándar B/N (biblioteca qrcode).
    Retorna imagen BGR numpy array.

    El QR B/N puede leerse con pyzbar, con el celular, o con cualquier
    lector estándar — no depende de nuestra grilla custom.
    """
    json_str = response.to_json()

    compressor = ZstdCompressor(level=3)
    compressed_bytes = compressor.compress(json_str.encode('utf-8'))
    b64_str = base64.b64encode(compressed_bytes).decode('utf-8')
    
    qr = qrcode.QRCode(
        version=None,                       # auto-seleccionar versión
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=box_size,
        border=border,
    )
    qr.add_data(b64_str)
    qr.make(fit=True)

    pil_img = qr.make_image(fill_color="black", back_color="white")
    arr = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def decode_response_from_qr(image: np.ndarray) -> CalResponseFrame | None:
    """
    Decodifica un CalResponseFrame desde una imagen que contiene un QR B/N.
    Usa pyzbar para leer el QR.
    Retorna None si no se puede decodificar.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    codes = pyzbar.decode(gray)

    decompressor = ZstdCompressor()
    
    for code in codes:
        try:
            b64_data = code.data.decode("utf-8")
            compressed_bytes = base64.b64decode(b64_data)

            json_bytes = decompressor.decompress(compressed_bytes)
            json_str = json_bytes.decode('utf-8')
            
            d = json.loads(json_str)
            if d.get("frame_type") == "CAL_RESPONSE":
                return CalResponseFrame.from_json(json_str)
        except Exception as e:
            raise DecompressionError("Error al decodificar CalResponseFrame desde QR: {e}") from e
    return None


def encode_probe_as_qr(
    probe: CalProbeFrame,
    box_size: int = 8,
    border: int = 4,
) -> np.ndarray:
    """
    Codifica CalProbeFrame como QR B/N estándar.
    El receptor lo lee antes de medir los colores.
    """
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=box_size,
        border=border,
    )
    qr.add_data(probe.to_json())
    qr.make(fit=True)
    pil_img = qr.make_image(fill_color="black", back_color="white")
    arr = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def decode_probe_from_qr(image: np.ndarray) -> CalProbeFrame | None:
    """Decodifica CalProbeFrame desde imagen con QR B/N."""
    try:
        from pyzbar import pyzbar
    except ImportError:
        raise ImportError("Instala pyzbar: pip install pyzbar")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    codes = pyzbar.decode(gray)
    for code in codes:
        try:
            text = code.data.decode("utf-8")
            d = json.loads(text)
            if d.get("frame_type") == "CAL_PROBE":
                return CalProbeFrame.from_json(text)
        except Exception:
            continue
    return None


# ── Paleta custom desde CalResponse ──────────────────────────────────────────

def build_custom_palette(
    candidates: list[tuple[int, int, int]],
    response: CalResponseFrame,
) -> ColorPalette:
    """
    Construye una ColorPalette usando exactamente los colores seleccionados
    por el receptor en el CalResponseFrame.

    La paleta resultante es no-uniforme en HSV — usa los colores reales
    que el canal puede distinguir, no la distribución teórica equidistante.
    """
    selected_colors = [candidates[i] for i in response.selected]
    n = len(selected_colors)

    # Ajustar a potencia de 2 más cercana por debajo
    valid_ns = [2, 4, 8, 16]
    n_used = max(v for v in valid_ns if v <= n) if n >= 2 else 2

    palette = ColorPalette(n_colors=n_used)
    # Sobreescribir la paleta interna con los colores negociados
    palette._palette    = selected_colors[:n_used]
    palette._calibrated = selected_colors[:n_used]
    palette._n_colors   = n_used

    return palette
