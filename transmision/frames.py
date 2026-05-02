"""
Estructuras de datos para los frames de Capa 1.

HandshakeFrame — QR #0, negocia capacidades antes de transmitir datos.
DataFrame      — frames de datos, NACK y FIN durante la transmisión.

Ambas son dataclasses con serialización/deserialización a bytes
y verificación de checksum CRC-16.
"""
from __future__ import annotations
import struct
from dataclasses import dataclass, field

from common.types import FrameType, CompressionMode
from common.exceptions import ChecksumError
from common.checksum import crc16

# ── Constantes de protocolo ───────────────────────────────────────────────────

MAGIC: int = 0x4E51          # "QN" — identifica el protocolo QR-NET
PROTOCOL_VERSION: int = 0x1  # versión actual del protocolo


# ── HandshakeFrame ────────────────────────────────────────────────────────────
#
# Layout (big-endian):
#   VERSION(4b) | FRAME_TYPE(4b) | MAGIC(16b)          = 3 bytes
#   SRC_MAC(48b)                                       = 6 bytes
#   DST_MAC(48b)                                       = 6 bytes
#   CD(2b) | GRID(6b) | ECC(2b) | SYNC(2b) | RSVD(4b)  = 2 bytes
#   FRAME_INTERVAL_MS(16b)                             = 2 bytes
#   COMP(4b) | RSVD(4b)                                = 1 byte
#   RSVD(7b)                                           = 1 byte (alineación)
#   TOTAL_FRAMES(32b)                                  = 4 bytes
#   FILE_SIZE(64b)                                     = 8 bytes
#   FILE_HASH(128b)                                    = 16 bytes
#   RSVD(8b) | CHECKSUM(16b)                           = 3 bytes
#                                                TOTAL = 52 bytes

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
    color_depth:  int           = 3      # 0-3 → 2/4/8/16 colores
    grid_size:    int           = 8      # múltiplo de 8; 8 → 64×64
    ecc_level:    int           = 2      # 0=L 1=M 2=Q 3=H
    sync_method:  int           = 3      # 0=timer 1=visual 2=tick 3=híbrido
    interval_ms:  int           = 150    # ms entre frames
    compression:  CompressionMode = CompressionMode.ZSTD
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
            compression  = CompressionAlgo((comp_byte >> 4) & 0xF),
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
    def end(cls, src_mac: bytes, dst_mac: bytes, seq: int) -> DataFrame:
        """Factory: crea un frame END para cerrar la sesión."""
        return cls(
            frame_type = FrameType.END,
            src_mac    = src_mac,
            dst_mac    = dst_mac,
            seq_num    = seq,
        )