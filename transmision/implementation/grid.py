"""
capa1/grid.py
Implementación de IGridCodec para la grilla propietaria de 64×64 módulos.

Estructura de la grilla (en módulos):
  - Zona de silencio : 2 módulos en cada borde (no se renderiza en la grilla,
                       es margen en la imagen final)
  - Finder patterns  : 3 bloques 7×7 en esquinas sup-izq, sup-der, inf-izq
  - Separadores      : 1 módulo blanco alrededor de cada finder
  - Timing patterns  : fila 6 y columna 6 (alternado negro/blanco)
  - Celda calibración: bloque 4×4 en esquina inf-der (colores de referencia)
  - Payload          : resto de módulos disponibles

Renderizado:
  - Cada módulo se renderiza como un cuadrado de MODULE_PX píxeles.
  - La imagen final incluye la zona de silencio como margen vacío.
  - Total imagen = (64 + 2*SILENCE) * MODULE_PX píxeles por lado.
"""
from __future__ import annotations
import math
from typing import ClassVar

import cv2
import numpy as np

from common.exceptions import GridDecodeError
from ..interfaces import IColorCodec, IGridCodec

# ── Constantes de la grilla ───────────────────────────────────────────────────

GRID_MODULES:    int = 64    # módulos por lado de la grilla
SILENCE_MODULES: int = 2     # zona de silencio en módulos
MODULE_PX:       int = 12    # píxeles por módulo en la imagen renderizada
FINDER_SIZE:     int = 7     # tamaño del finder pattern en módulos
CAL_SIZE:        int = 4     # tamaño del parche de calibración en módulos

# Área útil (sin bordes ni finders)
_TOTAL_MODULES   = GRID_MODULES * GRID_MODULES
_FINDER_AREA     = 3 * (FINDER_SIZE + 1) ** 2   # aprox. con separadores
_TIMING_AREA     = 2 * (GRID_MODULES - 2 * (FINDER_SIZE + 1))
_CAL_AREA        = CAL_SIZE * CAL_SIZE
_FORMAT_AREA     = 31
_RESERVED_APPROX = _FINDER_AREA + _TIMING_AREA + _CAL_AREA + _FORMAT_AREA

# Imagen final en píxeles
_IMG_SIDE = (GRID_MODULES + 2 * SILENCE_MODULES) * MODULE_PX


class Grid64Codec(IGridCodec):
    """
    Codifica bytes en una imagen BGR de la grilla 64×64 y los recupera.
    Cada módulo almacena bits_per_cell bits usando ColorPalette.
    """

    def __init__(self, module_px: int = MODULE_PX) -> None:
        self._module_px = module_px
        # Máscara booleana: True = módulo disponible para payload
        self._payload_mask: np.ndarray = self._build_payload_mask()
        self._payload_positions: list[tuple[int, int]] = [
            (r, c)
            for r in range(GRID_MODULES)
            for c in range(GRID_MODULES)
            if self._payload_mask[r, c]
        ]

    # ── IGridCodec ────────────────────────────────────────────────────────────

    def encode_grid(self, payload: bytes, codec: IColorCodec) -> np.ndarray:
        """
        Genera imagen BGR (H, W, 3) de la grilla codificada.
        payload: hasta max_payload_bytes bytes.
        """
        max_b = self.max_payload_bytes_for(codec)
        if len(payload) > max_b:
            raise ValueError(f"Payload {len(payload)} B excede máximo {max_b} B")

        # Convertir payload a lista de nibbles (o menos bits según color depth)
        bpc = codec.bits_per_cell
        symbols = _bytes_to_symbols(payload, bpc)

        # Crear imagen base blanca (zona de silencio = blanco)
        img = np.ones((_IMG_SIDE, _IMG_SIDE, 3), dtype=np.uint8) * 255

        # Rellenar módulos de payload
        sym_iter = iter(symbols)
        for (r, c) in self._payload_positions:
            sym = next(sym_iter, 0)   # 0 = padding si no hay más datos
            color = codec.encode(sym)
            self._draw_module(img, r, c, color)

        # Dibujar estructura fija
        self._draw_finder_patterns(img)
        self._draw_timing_patterns(img)
        self._draw_cal_patch(img, codec)
        self._draw_format_info(img, len(payload), bpc)

        return img

    def decode_grid(self, image: np.ndarray, codec: IColorCodec) -> bytes:
        """
        Extrae payload desde la imagen capturada.
        Primero localiza la grilla, luego calibra colores, luego lee módulos.
        """
        aligned = self._align_grid(image)
        if aligned is None:
            raise GridDecodeError("No se pudo localizar la grilla en la imagen")

        # Calibrar codec con el parche de referencia
        cal_patch = self._extract_cal_patch(aligned)
        codec.calibrate(cal_patch)

        # Leer longitud del payload desde los módulos de formato
        payload_len, bpc = self._read_format_info(aligned)
        if bpc != codec.bits_per_cell:
            raise GridDecodeError(f"bits_per_cell en imagen ({bpc}) != codec ({codec.bits_per_cell})")

        # Leer símbolos de los módulos de payload
        n_symbols = math.ceil(payload_len * 8 / bpc)
        symbols: list[int] = []
        for (r, c) in self._payload_positions[:n_symbols]:
            raw_color = self._read_module_color(aligned, r, c)
            symbols.append(codec.decode(raw_color))

        return _symbols_to_bytes(symbols, bpc, payload_len)

    @property
    def max_payload_bytes(self) -> int:
        """Máximo con 4 bits/módulo (16 colores) — caso óptimo."""
        return self.max_payload_bytes_for_bpc(4)

    @property
    def grid_px(self) -> int:
        return _IMG_SIDE

    def max_payload_bytes_for(self, codec: IColorCodec) -> int:
        return self.max_payload_bytes_for_bpc(codec.bits_per_cell)

    def max_payload_bytes_for_bpc(self, bpc: int) -> int:
        total_bits = len(self._payload_positions) * bpc
        return total_bits // 8

    # ── construcción de la máscara de payload ─────────────────────────────────

    @staticmethod
    def _build_payload_mask() -> np.ndarray:
        """
        Genera máscara booleana (64, 64).
        True = módulo disponible para datos.
        """
        mask = np.ones((GRID_MODULES, GRID_MODULES), dtype=bool)

        # Finder patterns + separadores (7+1 = 8 módulos)
        fs = FINDER_SIZE + 1  # 8
        # sup-izq
        mask[:fs, :fs] = False
        # sup-der
        mask[:fs, GRID_MODULES - fs:] = False
        # inf-izq
        mask[GRID_MODULES - fs:, :fs] = False

        # Timing patterns (fila 6 y columna 6)
        mask[6, fs: GRID_MODULES - fs] = False
        mask[fs: GRID_MODULES - fs, 6] = False

        # Parche de calibración 4×4 en esquina inf-der
        mask[GRID_MODULES - CAL_SIZE:, GRID_MODULES - CAL_SIZE:] = False

        # Módulos de formato (fila y col 8, cerca de finders)
        mask[8, :9]  = False
        mask[:9, 8]  = False
        mask[8, GRID_MODULES - 8:] = False
        mask[GRID_MODULES - 8:, 8] = False

        return mask

    # ── dibujo de la grilla ───────────────────────────────────────────────────

    def _draw_module(
        self,
        img: np.ndarray,
        row: int,
        col: int,
        color: tuple[int, int, int],
        *,
        bgr: bool = True,
    ) -> None:
        """Pinta un módulo (row, col) en la imagen con el color dado."""
        offset = SILENCE_MODULES * self._module_px
        y = offset + row * self._module_px
        x = offset + col * self._module_px
        # color llega en RGB; OpenCV usa BGR
        b, g, r = (color[2], color[1], color[0]) if bgr else color
        img[y: y + self._module_px, x: x + self._module_px] = (b, g, r)

    def _draw_finder_patterns(self, img: np.ndarray) -> None:
        """Dibuja los 3 finder patterns (7×7) en las esquinas correspondientes."""
        corners = [
            (0, 0),
            (0, GRID_MODULES - FINDER_SIZE),
            (GRID_MODULES - FINDER_SIZE, 0),
        ]
        for (r0, c0) in corners:
            for dr in range(FINDER_SIZE):
                for dc in range(FINDER_SIZE):
                    # Borde exterior + borde interior + centro
                    on_outer = (dr == 0 or dr == 6 or dc == 0 or dc == 6)
                    on_inner = (2 <= dr <= 4 and 2 <= dc <= 4)
                    color = (0, 0, 0) if (on_outer or on_inner) else (255, 255, 255)
                    self._draw_module(img, r0 + dr, c0 + dc, color, bgr=False)
            # Separador (1 módulo blanco alrededor del finder)
            # ya es blanco por defecto, no hace falta dibujarlo explícitamente

    def _draw_timing_patterns(self, img: np.ndarray) -> None:
        """Dibuja los timing patterns en fila 6 y columna 6."""
        fs = FINDER_SIZE + 1
        for i in range(fs, GRID_MODULES - fs):
            color = (0, 0, 0) if i % 2 == 0 else (255, 255, 255)
            self._draw_module(img, 6, i, color, bgr=False)
            self._draw_module(img, i, 6, color, bgr=False)

    def _draw_cal_patch(self, img: np.ndarray, codec: IColorCodec) -> None:
        """
        Dibuja el parche de calibración 4×4 en la esquina inferior derecha.
        Cada celda muestra un color de la paleta en orden.
        """
        r0 = GRID_MODULES - CAL_SIZE
        c0 = GRID_MODULES - CAL_SIZE
        idx = 0
        for dr in range(CAL_SIZE):
            for dc in range(CAL_SIZE):
                color = codec.encode(idx % codec.n_colors)
                self._draw_module(img, r0 + dr, c0 + dc, color)
                idx += 1

    def _draw_format_info(self, img: np.ndarray, payload_len: int, bpc: int) -> None:
        """
        Escribe longitud del payload y bits_per_cell en los módulos de formato.
        Usa los primeros 24 módulos de la fila/col 8.
        payload_len: hasta 2^20 ≈ 1 MB por frame (más que suficiente).
        bpc: 1-4.
        """
        # Empaquetamos 20 bits de longitud + 2 bits de bpc + 2 de ECC simple
        info = ((payload_len & 0xFFFFF) << 4) | ((bpc - 1) & 0x3)
        bits = [(info >> i) & 1 for i in range(23, -1, -1)]
        positions = [(8, c) for c in range(9, 9 + 24)]
        for (r, c), bit in zip(positions, bits):
            # Negro=módulo 1, Blanco=módulo 0. bgr=True por defecto → (B,G,R)
            # Como el color es neutro (gris extremo) B==G==R, no importa el orden
            color = (0, 0, 0) if bit else (255, 255, 255)
            self._draw_module(img, r, c, color)

    # ── lectura de la grilla ──────────────────────────────────────────────────

    def _align_grid(self, image: np.ndarray) -> np.ndarray | None:
        """
        Detecta y alinea la grilla en la imagen capturada usando
        los finder patterns como referencia.
        Retorna la imagen recortada y perspectiva-corregida, o None.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(
            cv2.bitwise_not(binary), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        # Heurística: el contorno más grande que se aproxima a un cuadrado
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for cnt in contours[:5]:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                return self._perspective_correct(image, approx.reshape(4, 2))

        return None

    def _perspective_correct(
        self, image: np.ndarray, corners: np.ndarray
    ) -> np.ndarray:
        """Corrige perspectiva de la grilla usando los 4 vértices detectados."""
        dst_size = _IMG_SIDE
        dst_pts = np.array([
            [0, 0], [dst_size - 1, 0],
            [dst_size - 1, dst_size - 1], [0, dst_size - 1],
        ], dtype=np.float32)

        # Ordenar corners: sup-izq, sup-der, inf-der, inf-izq
        src_pts = _order_corners(corners.astype(np.float32))
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return cv2.warpPerspective(image, M, (dst_size, dst_size))

    def _extract_cal_patch(self, aligned: np.ndarray) -> np.ndarray:
        """Extrae el parche de calibración 4×4 de la imagen alineada."""
        r0 = GRID_MODULES - CAL_SIZE
        c0 = GRID_MODULES - CAL_SIZE
        patch = np.zeros((CAL_SIZE, CAL_SIZE, 3), dtype=np.uint8)
        for dr in range(CAL_SIZE):
            for dc in range(CAL_SIZE):
                r_val, g_val, b_val = self._read_module_color(aligned, r0 + dr, c0 + dc)
                # _read_module_color retorna RGB; calibrate() espera BGR (formato OpenCV)
                patch[dr, dc] = (b_val, g_val, r_val)
        return patch

    def _read_module_color(
        self, aligned: np.ndarray, row: int, col: int
    ) -> tuple[int, int, int]:
        """
        Lee el color promedio del módulo (row, col) en la imagen alineada.
        Toma el 50% central del módulo para evitar bordes anti-alias.
        """
        offset = SILENCE_MODULES * self._module_px
        y = offset + row * self._module_px
        x = offset + col * self._module_px
        m = self._module_px
        pad = m // 4
        region = aligned[y + pad: y + m - pad, x + pad: x + m - pad]
        bgr_mean = region.mean(axis=(0, 1)).astype(int)
        b, g, r = int(bgr_mean[0]), int(bgr_mean[1]), int(bgr_mean[2])
        return (r, g, b)   # retornamos en RGB

    def _read_format_info(self, aligned: np.ndarray) -> tuple[int, int]:
        """Extrae payload_len y bpc desde los módulos de formato."""
        positions = [(8, c) for c in range(9, 9 + 24)]
        bits = []
        for (r, c) in positions:
            r_val, g_val, b_val = self._read_module_color(aligned, r, c)
            # _read_module_color retorna RGB; luminancia estándar
            lum = 0.299 * r_val + 0.587 * g_val + 0.114 * b_val
            bits.append(1 if lum < 128 else 0)

        info = 0
        for bit in bits:
            info = (info << 1) | bit

        payload_len = (info >> 4) & 0xFFFFF
        bpc = (info & 0x3) + 1
        return payload_len, bpc


# ── utilidades de conversión ──────────────────────────────────────────────────

def _bytes_to_symbols(data: bytes, bpc: int) -> list[int]:
    """
    Convierte bytes a lista de símbolos de bpc bits cada uno.
    bpc debe ser 1, 2, 3 o 4.
    """
    symbols: list[int] = []
    mask = (1 << bpc) - 1
    bit_buf = 0
    bits_in_buf = 0
    for byte in data:
        bit_buf = (bit_buf << 8) | byte
        bits_in_buf += 8
        while bits_in_buf >= bpc:
            bits_in_buf -= bpc
            symbols.append((bit_buf >> bits_in_buf) & mask)
    if bits_in_buf > 0:
        symbols.append((bit_buf << (bpc - bits_in_buf)) & mask)
    return symbols


def _symbols_to_bytes(symbols: list[int], bpc: int, length: int) -> bytes:
    """Convierte lista de símbolos de bpc bits a bytes (length bytes)."""
    bit_buf = 0
    bits_in_buf = 0
    result = bytearray()
    for sym in symbols:
        bit_buf = (bit_buf << bpc) | sym
        bits_in_buf += bpc
        while bits_in_buf >= 8:
            bits_in_buf -= 8
            result.append((bit_buf >> bits_in_buf) & 0xFF)
        if len(result) >= length:
            break
    return bytes(result[:length])


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """
    Ordena 4 puntos en: sup-izq, sup-der, inf-der, inf-izq.
    """
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    order = np.argsort(angles)
    pts = pts[order]
    # rotar para que sup-izq sea primero
    top = pts[np.argmin(pts[:, 1])]
    idx = np.where((pts == top).all(axis=1))[0][0]
    return np.roll(pts, -idx, axis=0)