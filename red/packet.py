"""
red/packet.py
Estructura del paquete de red de QR-NET (Capa 3).

Layout big-endian:
  VERSION(8b)       = 1 byte
  PACKET_TYPE(8b)   = 1 byte   (DATA / HELLO / BYE / CIRCUIT_SETUP / CIRCUIT_TEARDOWN)
  TTL(8b)           = 1 byte
  RESERVED(8b)      = 1 byte   (alineación, siempre 0x00)
  SRC_ID(128b)      = 16 bytes  — puede ser el circuit_id para paquetes anónimos
  DST_ID(128b)      = 16 bytes  — destino final o broadcast (0xFF * 16)
  CIRCUIT_ID(32b)   = 4 bytes   — 0 si el paquete no pertenece a un circuito
  PAYLOAD_LEN(16b)  = 2 bytes
  PAYLOAD(variable)
  CHECKSUM(32b)     = 4 bytes   — CRC-32 de todo lo anterior
                   HEADER = 46 bytes (sin payload)
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import IntEnum

from common.checksum import crc32
from common.exceptions import ChecksumError

# Constantes
PACKET_VERSION:  int = 0x1
DEFAULT_TTL:     int = 16
BROADCAST_ID:    bytes = b'\xff' * 16

_PKT_HEADER = struct.Struct(">B B B B 16s 16s I H")
PACKET_HEADER_SIZE = _PKT_HEADER.size   # 42 bytes
# + 4 bytes checksum al final

class PacketType(IntEnum):
    DATA             = 0x00   # datos de usuario
    HELLO            = 0x01   # anuncio de presencia (descubrimiento)
    BYE              = 0x02   # nodo abandona la mesh
    CIRCUIT_SETUP    = 0x03   # solicitud de circuito anónimo
    CIRCUIT_TEARDOWN = 0x04   # cierre de circuito anónimo
    ROUTE_UPDATE     = 0x05   # actualización de tabla de ruteo (distance vector)


# Packet
@dataclass
class Packet:
    """
    Paquete de Capa 3 de QR-NET.
    Para ruteo anónimo: src_id lleva el circuit_id codificado en 16 bytes
    en lugar del node_id real del originador.
    """
    version:     int        = PACKET_VERSION
    packet_type: PacketType = PacketType.DATA
    ttl:         int        = DEFAULT_TTL
    src_id:      bytes      = field(default=b'\x00' * 16)
    dst_id:      bytes      = field(default=BROADCAST_ID)
    circuit_id:  int        = 0          # 0 → paquete no anónimo
    payload:     bytes      = b''

    def __post_init__(self) -> None:
        if len(self.src_id) != 16:
            raise ValueError(f"src_id debe tener 16 bytes, recibió {len(self.src_id)}")
        if len(self.dst_id) != 16:
            raise ValueError(f"dst_id debe tener 16 bytes, recibió {len(self.dst_id)}")

    # serialización
    def to_bytes(self) -> bytes:
        """Serializa el paquete a bytes con CRC-32 al final."""
        header = _PKT_HEADER.pack(
            self.version,
            int(self.packet_type),
            self.ttl,
            0,                      # reservado
            self.src_id,
            self.dst_id,
            self.circuit_id,
            len(self.payload),
        )
        body = header + self.payload
        checksum = crc32(body)
        return body + struct.pack(">I", checksum)

    @classmethod
    def from_bytes(cls, data: bytes) -> Packet:
        """Deserializa desde bytes y verifica CRC-32."""
        if len(data) < PACKET_HEADER_SIZE + 4:
            raise ValueError(
                f"Packet requiere al menos {PACKET_HEADER_SIZE + 4} bytes, recibió {len(data)}"
            )
        (version, ptype, ttl, _reserved,
         src_id, dst_id, circuit_id, plen) = _PKT_HEADER.unpack(
            data[:PACKET_HEADER_SIZE]
        )
        payload = data[PACKET_HEADER_SIZE: PACKET_HEADER_SIZE + plen]
        body    = data[:PACKET_HEADER_SIZE + plen]
        expected = crc32(body)
        (actual,) = struct.unpack(">I", data[PACKET_HEADER_SIZE + plen: PACKET_HEADER_SIZE + plen + 4])
        if expected != actual:
            raise ChecksumError(
                f"Packet CRC-32 inválido: esperado {expected:#010x}, recibido {actual:#010x}"
            )

        return cls(
            version     = version,
            packet_type = PacketType(ptype),
            ttl         = ttl,
            src_id      = src_id,
            dst_id      = dst_id,
            circuit_id  = circuit_id,
            payload     = payload,
        )

    def verify(self) -> bool:
        """Verifica la integridad del paquete serializando y comparando."""
        try:
            Packet.from_bytes(self.to_bytes())
            return True
        except (ChecksumError, ValueError):
            return False

    # helpers

    @property
    def is_broadcast(self) -> bool:
        """True si el paquete va dirigido a todos los nodos de la mesh."""
        return self.dst_id == BROADCAST_ID
    @property
    def is_anonymous(self) -> bool:
        """True si el paquete usa un circuito virtual (circuit_id != 0)."""
        return self.circuit_id != 0
    def with_decremented_ttl(self) -> Packet:
        """Retorna una copia del paquete con TTL decrementado en 1."""
        from dataclasses import replace
        return replace(self, ttl=max(0, self.ttl - 1))

    # factories
    @classmethod
    def hello(cls, src_id: bytes, payload: bytes = b'') -> Packet:
        """Factory: paquete HELLO broadcast para descubrimiento de pares."""
        return cls(
            packet_type = PacketType.HELLO,
            src_id      = src_id,
            dst_id      = BROADCAST_ID,
            payload     = payload,
        )
    @classmethod
    def bye(cls, src_id: bytes) -> Packet:
        """Factory: paquete BYE broadcast al abandonar la mesh."""
        return cls(
            packet_type = PacketType.BYE,
            src_id      = src_id,
            dst_id      = BROADCAST_ID,
        )
    @classmethod
    def circuit_setup(cls, src_id: bytes, dst_id: bytes, circuit_id: int) -> Packet:
        """Factory: solicitud de establecimiento de circuito anónimo."""
        return cls(
            packet_type = PacketType.CIRCUIT_SETUP,
            src_id      = src_id,
            dst_id      = dst_id,
            circuit_id  = circuit_id,
        )
    @classmethod
    def circuit_teardown(cls, src_id: bytes, dst_id: bytes, circuit_id: int) -> Packet:
        """Factory: cierre explícito de circuito anónimo."""
        return cls(
            packet_type = PacketType.CIRCUIT_TEARDOWN,
            src_id      = src_id,
            dst_id      = dst_id,
            circuit_id  = circuit_id,
        )
