"""
capa1/frames.py
Estructuras de datos para los frames de Capa 1.

HandshakeFrame — QR #0, negocia capacidades antes de transmitir datos.
DataFrame      — frames de datos, NACK y FIN durante la transmisión.

Ambas son dataclasses con serialización/deserialización a bytes
y verificación de checksum CRC-16.
"""
from __future__ import annotations
import struct
from dataclasses import dataclass, field

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.network_policies import FrameType
from common.other import CompressionAlgorithm
from common.exceptions import ChecksumError
from common.checksum import crc16

# ── Constantes de protocolo ───────────────────────────────────────────────────

MAGIC: int = 0x4E51          # "QN" — identifica el protocolo QR-NET
PROTOCOL_VERSION: int = 0x1  # versión actual del protocolo


# ── HandshakeFrame ────────────────────────────────────────────────────────────
#
# Layout (big-endian):
#   VERSION(4b) | FRAME_TYPE(4b) | MAGIC(16b)         = 3 bytes
#   SRC_MAC(48b)                                       = 6 bytes
#   DST_MAC(48b)                                       = 6 bytes
#   CD(2b) | GRID(6b) | ECC(2b) | SYNC(2b) | RSVD(4b) = 2 bytes
#   FRAME_INTERVAL_MS(16b)                             = 2 bytes
#   COMP(4b) | RSVD(4b)                                = 1 byte
#   RSVD(7b)                                           = 1 byte (alineación)
#   TOTAL_FRAMES(32b)                                  = 4 bytes
#   FILE_SIZE(64b)                                     = 8 bytes
#   FILE_HASH(128b)                                    = 16 bytes
#   RSVD(8b) | CHECKSUM(16b)                           = 3 bytes
#                                               TOTAL = 52 bytes

_HS_STRUCT = struct.Struct(">B B H 6s 6s B H B B I Q 16s B H")
#             ver_ftype magic src dst cd_grid interval comp rsvd total fsize hash rsvd2 csum

HANDSHAKE_SIZE = _HS_STRUCT.size  # 52 bytes


@dataclass
class HandshakeFrame:
    # Identificación
    version:      int           = PROTOCOL_VERSION
    frame_type:   FrameType     = FrameType.SYN
    src_mac:      bytes         = field(default=b'\x00' * 6)
    dst_mac:      bytes         = field(default=b'\xff' * 6)  # broadcast por defecto
    # Capacidades negociadas
    color_depth:  int           = 0      # 0-3 → 2/4/8/16 colores  (0 = B/N)
    grid_size:    int           = 8      # múltiplo de 8; 8 → 64×64
    ecc_level:    int           = 2      # 0=L 1=M 2=Q 3=H
    sync_method:  int           = 3      # 0=timer 1=visual 2=tick 3=híbrido
    interval_ms:  int           = 150    # ms entre frames
    compression:  CompressionAlgorithm = CompressionAlgorithm.ZSTD
    # Metadatos del archivo
    total_frames: int           = 0
    file_size:    int           = 0
    file_hash:    bytes         = field(default=b'\x00' * 16)

    # ── serialización ─────────────────────────────────────────────────────────

    def to_bytes(self) -> bytes:
        """Serializa el frame a bytes con CRC-16 al final."""
        ver_ftype = ((self.version & 0xF) << 4) | (int(self.frame_type) & 0xF)
        cd_grid   = ((self.color_depth & 0x3) << 6) | (self.grid_size & 0x3F)
        ecc_sync  = ((self.ecc_level & 0x3) << 6) | ((self.sync_method & 0x3) << 4)
        comp_byte = (int(self.compression) & 0xF) << 4

        body = _HS_STRUCT.pack(
            ver_ftype,
            0,                  # reservado (segundo byte del primer campo)
            MAGIC,
            self.src_mac,
            self.dst_mac,
            cd_grid,
            self.interval_ms,
            comp_byte,
            ecc_sync,
            self.total_frames,
            self.file_size,
            self.file_hash,
            0,                  # reservado
            0,                  # checksum placeholder
        )
        checksum = crc16(body[:-2])
        return body[:-2] + struct.pack(">H", checksum)

    @classmethod
    def from_bytes(cls, data: bytes) -> HandshakeFrame:
        """Deserializa desde bytes y verifica CRC-16."""
        if len(data) < HANDSHAKE_SIZE:
            raise ValueError(f"HandshakeFrame requiere {HANDSHAKE_SIZE} bytes, recibió {len(data)}")

        expected = crc16(data[:-2])
        (actual,) = struct.unpack(">H", data[-2:])
        if expected != actual:
            raise ChecksumError(f"HandshakeFrame CRC-16 inválido: esperado {expected:#06x}, recibido {actual:#06x}")

        (ver_ftype, _, magic, src, dst,
         cd_grid, interval, comp_byte, ecc_sync,
         total, fsize, fhash, _, _) = _HS_STRUCT.unpack(data[:HANDSHAKE_SIZE])

        if magic != MAGIC:
            raise ValueError(f"MAGIC inválido: {magic:#06x}")

        return cls(
            version      = (ver_ftype >> 4) & 0xF,
            frame_type   = FrameType((ver_ftype) & 0xF),
            src_mac      = src,
            dst_mac      = dst,
            color_depth  = (cd_grid >> 6) & 0x3,
            grid_size    = cd_grid & 0x3F,
            ecc_level    = (ecc_sync >> 6) & 0x3,
            sync_method  = (ecc_sync >> 4) & 0x3,
            interval_ms  = interval,
            compression  = CompressionAlgorithm((comp_byte >> 4) & 0xF),
            total_frames = total,
            file_size    = fsize,
            file_hash    = fhash,
        )

    def verify(self) -> bool:
        """Verifica el checksum interno del frame."""
        try:
            HandshakeFrame.from_bytes(self.to_bytes())
            return True
        except (ChecksumError, ValueError):
            return False

    # ── helpers de negociación ─────────────────────────────────────────────────

    @property
    def n_colors(self) -> int:
        """Número de colores según color_depth negociado."""
        return 2 ** (self.color_depth + 1)   # 0→2, 1→4, 2→8, 3→16

    def negotiate_with(self, remote: HandshakeFrame) -> HandshakeFrame:
        """
        Retorna un nuevo HandshakeFrame con los valores mínimos entre
        self (local) y remote (propuesta del otro extremo).
        Usado por el receptor para construir el SYN-ACK.
        """
        return HandshakeFrame(
            frame_type   = FrameType.SYN_ACK,
            src_mac      = self.dst_mac,
            dst_mac      = self.src_mac,
            color_depth  = min(self.color_depth,  remote.color_depth),
            grid_size    = min(self.grid_size,     remote.grid_size),
            ecc_level    = min(self.ecc_level,     remote.ecc_level),
            sync_method  = remote.sync_method,
            interval_ms  = max(self.interval_ms,   remote.interval_ms),
            compression  = remote.compression,
            total_frames = remote.total_frames,
            file_size    = remote.file_size,
            file_hash    = remote.file_hash,
        )


# ── DataFrame ─────────────────────────────────────────────────────────────────
#
# Layout (big-endian):
#   VERSION(4b) | FRAME_TYPE(4b)  = 1 byte
#   SRC_MAC(48b)                  = 6 bytes
#   DST_MAC(48b)                  = 6 bytes
#   SEQ_NUM(16b)                  = 2 bytes
#   PAYLOAD_LEN(8b)               = 1 byte
#   PAYLOAD(variable, ≤112 bytes)
#   CHECKSUM(16b)                 = 2 bytes
#                       HEADER   = 18 bytes (sin payload)
#                       MAX TOTAL = 18 + 112 = 130 bytes (dentro de 128+overhead)

_DF_HEADER = struct.Struct(">B 6s 6s H B")  # ver_ftype src dst seq plen
DATAFRAME_HEADER_SIZE = _DF_HEADER.size      # 16 bytes
MAX_PAYLOAD = 112                            # bytes de payload neto por frame


@dataclass
class DataFrame:
    version:     int        = PROTOCOL_VERSION
    frame_type:  FrameType  = FrameType.DATA
    src_mac:     bytes      = field(default=b'\x00' * 6)
    dst_mac:     bytes      = field(default=b'\xff' * 6)
    seq_num:     int        = 0
    payload:     bytes      = b''

    def __post_init__(self) -> None:
        if len(self.payload) > MAX_PAYLOAD:
            raise ValueError(f"Payload excede {MAX_PAYLOAD} bytes: {len(self.payload)}")

    # ── serialización ─────────────────────────────────────────────────────────

    def to_bytes(self) -> bytes:
        ver_ftype = ((self.version & 0xF) << 4) | (int(self.frame_type) & 0xF)
        header = _DF_HEADER.pack(
            ver_ftype,
            self.src_mac,
            self.dst_mac,
            self.seq_num,
            len(self.payload),
        )
        body = header + self.payload
        checksum = crc16(body)
        return body + struct.pack(">H", checksum)

    @classmethod
    def from_bytes(cls, data: bytes) -> DataFrame:
        if len(data) < DATAFRAME_HEADER_SIZE + 2:
            raise ValueError("DataFrame demasiado corto")

        (ver_ftype, src, dst, seq, plen) = _DF_HEADER.unpack(data[:DATAFRAME_HEADER_SIZE])
        payload = data[DATAFRAME_HEADER_SIZE: DATAFRAME_HEADER_SIZE + plen]

        body = data[:DATAFRAME_HEADER_SIZE + plen]
        expected = crc16(body)
        (actual,) = struct.unpack(">H", data[DATAFRAME_HEADER_SIZE + plen: DATAFRAME_HEADER_SIZE + plen + 2])
        if expected != actual:
            raise ChecksumError(f"DataFrame CRC-16 inválido seq={seq}: esperado {expected:#06x}, recibido {actual:#06x}")

        return cls(
            version    = (ver_ftype >> 4) & 0xF,
            frame_type = FrameType(ver_ftype & 0xF),
            src_mac    = src,
            dst_mac    = dst,
            seq_num    = seq,
            payload    = payload,
        )

    def verify(self) -> bool:
        try:
            DataFrame.from_bytes(self.to_bytes())
            return True
        except (ChecksumError, ValueError):
            return False

    @classmethod
    def nack(cls, src_mac: bytes, dst_mac: bytes, missing_seq: int) -> DataFrame:
        """Factory: crea un frame NACK solicitando retransmisión de missing_seq."""
        return cls(
            frame_type = FrameType.NACK,
            src_mac    = src_mac,
            dst_mac    = dst_mac,
            seq_num    = missing_seq,
            payload    = struct.pack(">H", missing_seq),
        )

    @classmethod
    def fin(cls, src_mac: bytes, dst_mac: bytes, seq: int) -> DataFrame:
        """Factory: crea un frame FIN para cerrar la sesión."""
        return cls(
            frame_type = FrameType.FIN,
            src_mac    = src_mac,
            dst_mac    = dst_mac,
            seq_num    = seq,
        )

# ── Frames de calibración de color ────────────────────────────────────────────
#
# CalProbeFrame — el emisor anuncia los colores candidatos que va a mostrar.
#   Serializado como JSON compacto dentro de un QR estándar (qrcode library).
#   Campos:
#     version      : uint8
#     frame_type   : "CAL_PROBE"
#     src_mac      : hex string
#     dst_mac      : hex string
#     n_candidates : int  — cuántos colores mostrará (16–64)
#     patch_px     : int  — tamaño de cada patch en px
#     duration_ms  : int  — cuánto tiempo mostrará cada ronda
#
# CalResponseFrame — el receptor responde con los índices que distingue.
#   También serializado como JSON en QR estándar B/N.
#   Campos:
#     version        : uint8
#     frame_type     : "CAL_RESPONSE"
#     src_mac        : hex string
#     dst_mac        : hex string
#     selected       : list[int]  — índices de colores distinguibles
#     confusion      : list[list[float]]  — matriz de confusión NxN (N=n_candidates)
#     recommended_cd : int  — COLOR_DEPTH recomendado (0-3)
#     interval_ms    : int  — intervalo recomendado entre frames
#
# Ambos se transportan como JSON → bytes UTF-8 → QR estándar.
# El QR estándar B/N se puede leer con pyzbar o cualquier lector de QR.

import json as _json

CAL_PROBE_TYPE    = "CAL_PROBE"
CAL_RESPONSE_TYPE = "CAL_RESPONSE"


@dataclass
class CalProbeFrame:
    """
    Anuncio del emisor antes de mostrar los colores candidatos.
    Viaja como QR estándar B/N (legible por cualquier dispositivo).
    """
    src_mac      : bytes
    dst_mac      : bytes
    n_candidates : int   = 32      # colores candidatos a mostrar
    patch_px     : int   = 120     # tamaño del patch en pantalla
    duration_ms  : int   = 3000    # ms que se muestra cada ronda de colores
    version      : int   = PROTOCOL_VERSION

    def to_json(self) -> str:
        return _json.dumps({
            "version"      : self.version,
            "frame_type"   : CAL_PROBE_TYPE,
            "src_mac"      : self.src_mac.hex(),
            "dst_mac"      : self.dst_mac.hex(),
            "n_candidates" : self.n_candidates,
            "patch_px"     : self.patch_px,
            "duration_ms"  : self.duration_ms,
        }, separators=(",", ":"))

    @classmethod
    def from_json(cls, s: str) -> CalProbeFrame:
        d = _json.loads(s)
        if d.get("frame_type") != CAL_PROBE_TYPE:
            raise ValueError(f"frame_type inválido: {d.get('frame_type')}")
        return cls(
            version      = d["version"],
            src_mac      = bytes.fromhex(d["src_mac"]),
            dst_mac      = bytes.fromhex(d["dst_mac"]),
            n_candidates = d["n_candidates"],
            patch_px     = d["patch_px"],
            duration_ms  = d["duration_ms"],
        )


@dataclass
class CalResponseFrame:
    """
    Respuesta del receptor con los colores que puede distinguir.
    También viaja como QR estándar B/N.
    """
    src_mac        : bytes
    dst_mac        : bytes
    selected       : list          # list[int] — índices distinguibles
    confusion      : list          # list[list[float]] — matriz NxN
    recommended_cd : int   = 0     # COLOR_DEPTH recomendado (0 = B/N)
    interval_ms    : int   = 150
    version        : int   = PROTOCOL_VERSION

    def to_json(self) -> str:
        # Redondear matriz a 3 decimales para compactar el QR
        rounded = [[round(v, 3) for v in row] for row in self.confusion]
        return _json.dumps({
            "version"        : self.version,
            "frame_type"     : CAL_RESPONSE_TYPE,
            "src_mac"        : self.src_mac.hex(),
            "dst_mac"        : self.dst_mac.hex(),
            "selected"       : self.selected,
            "confusion"      : rounded,
            "recommended_cd" : self.recommended_cd,
            "interval_ms"    : self.interval_ms,
        }, separators=(",", ":"))

    @classmethod
    def from_json(cls, s: str) -> CalResponseFrame:
        d = _json.loads(s)
        if d.get("frame_type") != CAL_RESPONSE_TYPE:
            raise ValueError(f"frame_type inválido: {d.get('frame_type')}")
        return cls(
            version        = d["version"],
            src_mac        = bytes.fromhex(d["src_mac"]),
            dst_mac        = bytes.fromhex(d["dst_mac"]),
            selected       = d["selected"],
            confusion      = d["confusion"],
            recommended_cd = d["recommended_cd"],
            interval_ms    = d["interval_ms"],
        )

    @property
    def n_usable_colors(self) -> int:
        """Número de colores seleccionados como distinguibles."""
        return len(self.selected)

    @property
    def color_depth(self) -> int:
        """COLOR_DEPTH efectivo basado en n_usable_colors."""
        n = self.n_usable_colors
        if n >= 16: return 3
        if n >= 8:  return 2
        if n >= 4:  return 1
        return 0