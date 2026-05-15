"""
capa1/interfaces.py
Contratos de todas las abstracciones de Capa 1.
Las clases concretas implementan estas interfaces; las capas superiores
solo dependen de estas interfaces, nunca de las implementaciones.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .frames import HandshakeFrame, DataFrame
    from common.other import RGB
    from common.network_policies import AdapterType, SelectionPolicy


# ── Color ─────────────────────────────────────────────────────────────────────

class IColorCodec(ABC):
    """
    Codifica y decodifica valores (nibbles 0-15) en colores RGB.
    La implementación concreta define la paleta y la calibración.
    """

    @abstractmethod
    def encode(self, nibble: int) -> RGB:
        """Convierte un nibble (0-15) al color RGB correspondiente."""

    @abstractmethod
    def decode(self, color: RGB) -> int:
        """
        Clasifica un color RGB observado y retorna el nibble más cercano.
        Usa distancia euclídea en espacio HSV sobre la paleta calibrada.
        """

    @abstractmethod
    def calibrate(self, ref_patch: np.ndarray) -> None:
        """
        Ajusta la paleta interna usando el parche de referencia 4x4
        capturado desde la imagen real.
        ref_patch: array shape (4, 4, 3) dtype uint8 en BGR (OpenCV).
        """

    @property
    @abstractmethod
    def bits_per_cell(self) -> int:
        """Bits codificados por módulo: 1, 2, 3 o 4 según color depth."""

    @property
    @abstractmethod
    def n_colors(self) -> int:
        """Número de colores distintos en la paleta (2, 4, 8 o 16)."""


# ── Cámara ────────────────────────────────────────────────────────────────────

class ICameraInterface(ABC):
    """Abstracción de captura de imagen. Permite sustituir OpenCV por mock en tests."""

    @abstractmethod
    def open(self) -> None:
        """Inicializa el dispositivo de captura."""

    @abstractmethod
    def close(self) -> None:
        """Libera el dispositivo de captura."""

    @abstractmethod
    def capture(self) -> np.ndarray:
        """
        Captura un frame. Retorna array shape (H, W, 3) dtype uint8 BGR.
        Lanza AdapterError si el dispositivo no está disponible.
        """

    @abstractmethod
    def detect_change(self, prev: np.ndarray, curr: np.ndarray) -> bool:
        """
        Retorna True si los dos frames son suficientemente distintos
        para considerarse un nuevo QR (umbral configurable en la implementación).
        """

    @property
    @abstractmethod
    def fps(self) -> float:
        """FPS nominal del dispositivo."""

    @property
    @abstractmethod
    def resolution(self) -> tuple[int, int]:
        """Resolución (ancho, alto) en píxeles."""


# ── Grilla ────────────────────────────────────────────────────────────────────

class IGridCodec(ABC):
    """
    Codifica bytes en una imagen de grilla 64x64 y los recupera de una imagen.
    El formato de grilla es propietario (no estándar QR).
    """

    @abstractmethod
    def encode_grid(self, payload: bytes, codec: IColorCodec) -> np.ndarray:
        """
        Genera imagen BGR de la grilla codificada.
        payload: hasta MAX_PAYLOAD_BYTES bytes.
        Retorna array shape (GRID_PX, GRID_PX, 3) dtype uint8.
        """

    @abstractmethod
    def decode_grid(self, image: np.ndarray, codec: IColorCodec) -> bytes:
        """
        Extrae bytes del payload desde la imagen capturada.
        Lanza GridDecodeError si no se puede alinear la grilla.
        """

    @property
    @abstractmethod
    def max_payload_bytes(self) -> int:
        """Máximo de bytes de payload por frame según color depth actual."""

    @property
    @abstractmethod
    def grid_px(self) -> int:
        """Tamaño de la imagen renderizada en píxeles por lado."""


# ── Compresión ────────────────────────────────────────────────────────────────

class ICompressor(ABC):
    """Compresión y descompresión de streams de bytes."""

    @abstractmethod
    def compress(self, data: bytes) -> bytes:
        """Comprime data y retorna el bloque comprimido."""

    @abstractmethod
    def decompress(self, data: bytes) -> bytes:
        """Descomprime data y retorna los bytes originales."""

    @abstractmethod
    def compress_stream(self, chunks):
        """
        Generador. Recibe iterable de bytes chunks, yield de chunks comprimidos.
        Permite procesar archivos grandes sin cargar todo en RAM.
        """

    @abstractmethod
    def decompress_stream(self, chunks):
        """Generador inverso de compress_stream."""

    @property
    @abstractmethod
    def compression_algorithm(self):
        """CompressionAlgorithm que implementa esta clase."""


# ── Cola FIFO ─────────────────────────────────────────────────────────────────

class IFrameQueue(ABC):
    """
    Cola thread-safe de imágenes capturadas.
    El hilo productor (cámara) y el hilo consumidor (decodificador)
    solo interactúan a través de esta interfaz.
    """

    @abstractmethod
    def put(self, frame: np.ndarray, block: bool = False) -> bool:
        """
        Encola un frame. Si block=False y la cola está llena,
        aplica la política configurada (drop-oldest o discard-new).
        Retorna True si el frame fue encolado.
        """

    @abstractmethod
    def get(self, timeout: float = 2.0) -> np.ndarray:
        """
        Desencola y retorna el frame más antiguo.
        Bloquea hasta timeout segundos. Lanza queue.Empty si no hay frame.
        """

    @abstractmethod
    def qsize(self) -> int:
        """Número de frames actualmente en la cola."""

    @abstractmethod
    def clear(self) -> None:
        """Vacía la cola sin procesar los frames pendientes."""

    @property
    @abstractmethod
    def maxsize(self) -> int:
        """Capacidad máxima de la cola en frames."""


# ── Adaptador de red ──────────────────────────────────────────────────────────

class INetworkAdapter(ABC):
    """
    Contrato común para todos los canales físicos:
    QR/luz, Ethernet/WiFi (TCP) y video compartido.
    QRNetNode solo conoce esta interfaz.
    """

    @abstractmethod
    def send(self, data: bytes) -> bool:
        """
        Transmite data por el canal. Retorna True si fue enviado correctamente.
        """

    @abstractmethod
    def receive(self) -> bytes | None:
        """
        Intenta recibir datos del canal.
        Retorna bytes o None si no hay datos disponibles.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """True si el canal está operativo y puede enviar/recibir."""

    @abstractmethod
    def get_mac(self) -> bytes:
        """Retorna los 6 bytes de la dirección física del adaptador."""

    @abstractmethod
    def get_type(self) -> AdapterType:
        """Tipo de canal (QR_LIGHT, ETHERNET, WIFI, VIDEO_STREAM)."""

    @abstractmethod
    def get_cost(self) -> int:
        """
        Costo relativo del canal. Menor = preferido por AdapterSelector.
        Permite que la política LOWEST_COST elija el canal más eficiente.
        """


# ── Selector de canal ─────────────────────────────────────────────────────────

class IAdapterSelector(ABC):
    """
    Elige el INetworkAdapter adecuado para cada paquete saliente.
    Es el único punto donde se decide qué canal físico usar.
    """

    @abstractmethod
    def select(self, dst_mac: bytes) -> INetworkAdapter:
        """
        Retorna el adaptador más adecuado para el destino dado.
        Lanza AdapterError si ningún adaptador está disponible.
        """

    @abstractmethod
    def register(self, adapter: INetworkAdapter) -> None:
        """Registra un adaptador en el pool de candidatos."""

    @abstractmethod
    def set_policy(self, policy: SelectionPolicy) -> None:
        """Cambia la política de selección en tiempo de ejecución."""

# Frame
class IFrameRenderer(ABC):

    @abstractmethod
    def show(self, frame: np.ndarray) -> None:
        """
        Muestra un frame visualmente.
        """

    @abstractmethod
    def close(self) -> None:
        """
        Libera recursos gráficos.
        """