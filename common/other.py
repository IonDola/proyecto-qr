from __future__ import annotations
from enum import IntEnum, auto
from typing import NewType

RGB    = tuple[int, int, int]       # (r, g, b) cada canal 0-255

class CompressionAlgorithm(IntEnum):
    """Algoritmo de compresión negociado en HandshakeFrame."""
    NONE   = 0x0
    ZSTD   = 0x1
    LZ4    = 0x2
    BROTLI = 0x3