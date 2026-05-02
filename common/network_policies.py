from __future__ import annotations
from enum import IntEnum, auto
from typing import NewType

MAC    = NewType("MAC",    bytes)   # 6 bytes  — dirección física análoga a MAC
NodeID = NewType("NodeID", bytes)   # 16 bytes — identificador de nodo en la mesh


def make_mac(b: bytes) -> MAC:
    if len(b) != 6:
        raise ValueError(f"MAC debe tener 6 bytes, recibió {len(b)}")
    return MAC(b)


def make_node_id(b: bytes) -> NodeID:
    if len(b) != 16:
        raise ValueError(f"NodeID debe tener 16 bytes, recibió {len(b)}")
    return NodeID(b)


class AdapterType(IntEnum):
    """Canal físico disponible para un NetworkAdapter."""
    QR_LIGHT     = 0
    ETHERNET     = 1
    WIFI         = 2
    VIDEO_STREAM = 3


class SelectionPolicy(IntEnum):
    """Política de selección de canal en AdapterSelector."""
    PREFER_QR       = 0
    PREFER_TCP      = 1
    LOWEST_COST     = 2
    FIRST_AVAILABLE = 3


class FrameType(IntEnum):
    """Tipo de frame en el protocolo de Capa 1."""
    # Handshake
    SYN     = 0x0
    SYN_ACK = 0x1
    ACK     = 0x2
    # Datos
    DATA    = 0x3
    NACK    = 0x4
    FIN     = 0x5
    FIN_ACK = 0x6