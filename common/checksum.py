"""
common/checksum.py
Funciones de verificación de integridad usadas en Capa 1 y Capa 2/3.
Sin estado — todas son funciones puras.
"""
from __future__ import annotations
import binascii
import hashlib
import struct


def crc16(data: bytes) -> int:
    """
    CRC-16/CCITT-FALSE.
    Usado en DataFrame y HandshakeFrame (Capa 1) por su bajo costo computacional.
    Retorna un entero de 16 bits (0–65535).
    """
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc <<= 1
            crc &= 0xFFFF
    return crc


def crc32(data: bytes) -> int:
    """
    CRC-32 estándar (IEEE 802.3).
    Usado en Packet (Capa 2/3) para mayor cobertura sobre payloads grandes.
    Retorna un entero de 32 bits sin signo.
    """
    return binascii.crc32(data) & 0xFFFF_FFFF


def sha128(data: bytes) -> bytes:
    """
    SHA-256 truncado a 128 bits (16 bytes).
    Usado en HandshakeFrame para verificar integridad del archivo completo
    al finalizar la sesión de datacasting.
    Retorna 16 bytes.
    """
    digest = hashlib.sha256(data).digest()
    return digest[:16]


def verify_crc16(data: bytes, expected: int) -> bool:
    return crc16(data) == expected


def verify_crc32(data: bytes, expected: int) -> bool:
    return crc32(data) == expected


def verify_sha128(data: bytes, expected: bytes) -> bool:
    return sha128(data) == expected