"""
capa1/color_palette.py
Implementación de IColorCodec con 16 colores separados en espacio HSV.

Diseño:
- 16 colores equidistantes en el eje Hue (22.5° de separación).
- Saturation=0.85, Value=0.90 fijos → colores vivos y distinguibles.
- Calibración dinámica desde el parche de referencia 4×4 en cada frame.
- Decodificación por distancia mínima en espacio HSV (más robusto que RGB
  ante variaciones de iluminación).
"""
from __future__ import annotations
import colorsys
import math
from typing import ClassVar

import numpy as np

from common.other import RGB
from transmision.interfaces import IColorCodec


class ColorPalette(IColorCodec):
    """
    Paleta de N colores (2, 4, 8 o 16) bien separados en HSV.
    La calibración ajusta la paleta interna a las condiciones reales
    de iluminación y sensor de la cámara.
    """

    # Posición del parche de calibración en la grilla (módulos desde esquina inf-der)
    CAL_PATCH_MODULES: ClassVar[int] = 4   # bloque 4×4
    # Saturación y valor base de la paleta
    BASE_SAT: ClassVar[float] = 0.85
    BASE_VAL: ClassVar[float] = 0.90

    def __init__(self, n_colors: int = 16) -> None:
        if n_colors not in (2, 4, 8, 16):
            raise ValueError(f"n_colors debe ser 2, 4, 8 o 16; recibió {n_colors}")
        self._n_colors = n_colors
        # Paleta teórica inicial (RGB uint8)
        self._palette: list[RGB] = self._build_palette(n_colors)
        # Paleta calibrada (se actualiza en calibrate())
        self._calibrated: list[RGB] = list(self._palette)

    # ── IColorCodec ───────────────────────────────────────────────────────────

    def encode(self, nibble: int) -> RGB:
        """Retorna el color RGB calibrado para el nibble dado (0..n_colors-1)."""
        if not (0 <= nibble < self._n_colors):
            raise ValueError(f"nibble {nibble} fuera de rango [0, {self._n_colors})")
        return self._calibrated[nibble]

    def decode(self, color: RGB) -> int:
        """
        Clasifica un color RGB en el nibble más cercano.
        Convierte a HSV para mayor robustez ante cambios de iluminación.
        """
        h, s, v = _rgb_to_hsv(color)
        best_idx, best_dist = 0, float("inf")
        for i, ref in enumerate(self._calibrated):
            rh, rs, rv = _rgb_to_hsv(ref)
            dist = _hsv_distance(h, s, v, rh, rs, rv)
            if dist < best_dist:
                best_dist, best_idx = dist, i
        return best_idx

    def calibrate(self, ref_patch: np.ndarray) -> None:
        """
        Ajusta la paleta usando el parche de referencia 4×4 capturado.
        ref_patch: array (4, 4, 3) BGR uint8 (formato OpenCV).
        Los 16 colores están dispuestos en orden de izquierda a derecha,
        de arriba a abajo en el parche.
        """
        if ref_patch.shape != (4, 4, 3):
            raise ValueError(f"ref_patch debe ser (4,4,3), recibió {ref_patch.shape}")

        observed: list[RGB] = []
        for row in range(4):
            for col in range(4):
                b, g, r = ref_patch[row, col]   # OpenCV es BGR
                observed.append((int(r), int(g), int(b)))

        # Solo calibramos los N colores que usamos
        for i in range(self._n_colors):
            self._calibrated[i] = observed[i]

    @property
    def bits_per_cell(self) -> int:
        return int(math.log2(self._n_colors))

    @property
    def n_colors(self) -> int:
        return self._n_colors

    # ── helpers públicos ──────────────────────────────────────────────────────

    @classmethod
    def from_negotiated(cls, path: str) -> "ColorPalette":
        """
        Construye una ColorPalette desde paleta_negociada.json
        generado por negotiation_session.py.
        """
        import json, math
        with open(path) as f:
            data = json.load(f)
        colors_rgb = [tuple(c) for c in data["palette_rgb"]]
        return cls.from_rgb_list(colors_rgb)

    @classmethod
    def from_rgb_list(cls, colors: list) -> "ColorPalette":
        """
        Construye una ColorPalette desde una lista de colores RGB explícita.
        Trunca a la potencia de 2 más cercana por debajo.
        """
        valid_ns = [2, 4, 8, 16]
        n = len(colors)
        n_used = max(v for v in valid_ns if v <= n) if n >= 2 else 2
        instance = cls.__new__(cls)
        instance._n_colors   = n_used
        instance._palette    = [tuple(c) for c in colors[:n_used]]
        instance._calibrated = [tuple(c) for c in colors[:n_used]]
        return instance

    def reset_calibration(self) -> None:
        """Restaura la paleta calibrada a los valores teóricos originales."""
        self._calibrated = list(self._palette)

    def palette_rgb(self) -> list[RGB]:
        """Retorna copia de la paleta calibrada actual (útil para debug y tests)."""
        return list(self._calibrated)

    # ── helpers internos ──────────────────────────────────────────────────────

    @staticmethod
    def _build_palette(n: int) -> list[RGB]:
        """
        Genera n colores equidistantes en hue, con sat=BASE_SAT, val=BASE_VAL.
        Separación angular = 360/n grados.
        """
        palette: list[RGB] = []
        for i in range(n):
            hue = i / n
            r, g, b = colorsys.hsv_to_rgb(hue, ColorPalette.BASE_SAT, ColorPalette.BASE_VAL)
            palette.append((int(r * 255), int(g * 255), int(b * 255)))
        return palette


# ── funciones auxiliares ──────────────────────────────────────────────────────

def _rgb_to_hsv(color: RGB) -> tuple[float, float, float]:
    r, g, b = color
    return colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)


def _hsv_distance(h1: float, s1: float, v1: float,
                  h2: float, s2: float, v2: float) -> float:
    """
    Distancia en espacio HSV ponderada.
    El hue es circular (0 y 1 son el mismo color),
    por eso se usa la diferencia angular mínima.
    """
    dh = min(abs(h1 - h2), 1.0 - abs(h1 - h2))  # distancia circular
    ds = abs(s1 - s2)
    dv = abs(v1 - v2)
    # Ponderamos hue más fuerte porque es la dimensión más discriminante
    return math.sqrt((2 * dh) ** 2 + ds ** 2 + dv ** 2)