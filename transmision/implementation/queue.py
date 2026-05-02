"""
capa1/fifo_queue.py
Implementación de IFrameQueue con cola thread-safe productor-consumidor.

El hilo PRODUCTOR (cámara) llama a put() — nunca bloquea.
El hilo CONSUMIDOR (decodificador) llama a get() — bloquea hasta timeout.

Política cuando la cola está llena:
  DROP_OLDEST  → descarta el frame más antiguo para dar lugar al nuevo (default)
  DISCARD_NEW  → descarta el frame entrante y continúa

Esta política es crítica para evitar que el decodificador procese frames
obsoletos cuando está más lento que la cámara.
"""
from __future__ import annotations
import queue
import threading
from enum import auto, Enum

import numpy as np

from ..interfaces import IFrameQueue


class DropPolicy(Enum):
    DROP_OLDEST  = auto()   # descarta el más antiguo si la cola está llena
    DISCARD_NEW  = auto()   # descarta el entrante si la cola está llena


class FifoFrameQueue(IFrameQueue):
    """
    Cola FIFO thread-safe de imágenes numpy.

    Parámetros:
        maxsize    : capacidad máxima en frames (default 32 ≈ 16 MB a 1080p)
        policy     : comportamiento cuando la cola está llena
        hash_dedup : si True, descarta frames duplicados por hash de imagen
    """

    def __init__(
        self,
        maxsize: int = 32,
        policy: DropPolicy = DropPolicy.DROP_OLDEST,
        hash_dedup: bool = True,
    ) -> None:
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=maxsize)
        self._policy = policy
        self._hash_dedup = hash_dedup
        self._last_hash: int | None = None
        self._lock = threading.Lock()
        # estadísticas
        self._dropped: int = 0
        self._enqueued: int = 0

    # ── IFrameQueue ───────────────────────────────────────────────────────────

    def put(self, frame: np.ndarray, block: bool = False) -> bool:
        """
        Intenta encolar el frame.
        Si hash_dedup=True y el frame es idéntico al anterior, lo descarta.
        Si la cola está llena aplica la política configurada.
        Retorna True si fue encolado.
        """
        if self._hash_dedup:
            h = _fast_hash(frame)
            with self._lock:
                if h == self._last_hash:
                    return False          # frame duplicado, descartado
                self._last_hash = h

        if self._q.full():
            if self._policy == DropPolicy.DROP_OLDEST:
                try:
                    self._q.get_nowait()  # descarta el más antiguo
                    self._dropped += 1
                except queue.Empty:
                    pass
            else:
                # DISCARD_NEW: simplemente no encola
                self._dropped += 1
                return False

        try:
            self._q.put_nowait(frame)
            self._enqueued += 1
            return True
        except queue.Full:
            self._dropped += 1
            return False

    def get(self, timeout: float = 2.0) -> np.ndarray:
        """
        Retorna el frame más antiguo.
        Lanza queue.Empty si no hay frame en timeout segundos.
        """
        return self._q.get(block=True, timeout=timeout)

    def qsize(self) -> int:
        return self._q.qsize()

    def clear(self) -> None:
        """Vacía la cola sin procesar los frames pendientes."""
        with self._lock:
            while not self._q.empty():
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    break
            self._last_hash = None

    @property
    def maxsize(self) -> int:
        return self._q.maxsize

    # ── estadísticas ──────────────────────────────────────────────────────────

    @property
    def dropped(self) -> int:
        """Número total de frames descartados desde la creación de la cola."""
        return self._dropped

    @property
    def enqueued(self) -> int:
        """Número total de frames encolados exitosamente."""
        return self._enqueued

    @property
    def drop_rate(self) -> float:
        """Fracción de frames perdidos sobre el total recibido (0.0 – 1.0)."""
        total = self._enqueued + self._dropped
        return self._dropped / total if total > 0 else 0.0


# ── utilidad ──────────────────────────────────────────────────────────────────

def _fast_hash(frame: np.ndarray) -> int:
    step_y = max(1, frame.shape[0] // 64)
    step_x = max(1, frame.shape[1] // 64)
    small = frame[::step_y, ::step_x]
    return hash(small.tobytes())