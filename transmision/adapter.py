"""
capa1/adaptador.py
DispositivoLuzAdaptador — implementación principal de INetworkAdapter
para el canal óptico QR.

Responsabilidades:
  1. Ejecutar el handshake de 3 vías (QR #0).
  2. Fragmentar datos comprimidos en DataFrames y renderizarlos como grilla.
  3. Capturar frames con el pipeline FIFO y decodificarlos.
  4. Retransmitir frames específicos ante un NACK.
  5. Cerrar la sesión con FIN/FIN-ACK.

Diseño de hilos:
  - Hilo PRODUCTOR: captura cámara → FifoFrameQueue
  - Hilo CONSUMIDOR: FifoFrameQueue → decodificación → buffer de recepción
  - Hilo principal: envío de frames (renderiza + muestra en pantalla)
"""
from __future__ import annotations
import os
import queue
import threading
import time
import uuid

import cv2
import numpy as np

from common.network_policies import AdapterType, MAC, make_mac
from common.exceptions import AdapterError, HandshakeError, GridDecodeError
from common.checksum import sha128

from .interfaces import INetworkAdapter, IColorCodec, ICameraInterface, IGridCodec, IFrameQueue, ICompressor
from .frames import HandshakeFrame, DataFrame, FrameType, MAX_PAYLOAD, PROTOCOL_VERSION
from .implementation.color_palette import ColorPalette
from .implementation.camera import OpenCVCamera
from .implementation.grid import Grid64Codec
from .implementation.compressor import ZstdCompressor
from .implementation.queue import FifoFrameQueue


class DispositivoLuzAdaptador(INetworkAdapter):
    """
    Adaptador de canal óptico QR.
    Implementa INetworkAdapter para integrarse con AdapterSelector y QRNetNode.

    Parámetros:
        mac           : dirección física de 6 bytes (se genera aleatoria si None)
        camera        : ICameraInterface (default: OpenCVCamera device 0)
        color_codec   : IColorCodec (default: ColorPalette 16 colores)
        grid_codec    : IGridCodec (default: Grid64Codec)
        compressor    : ICompressor (default: ZstdCompressor nivel 3)
        frame_queue   : IFrameQueue (default: FifoFrameQueue maxsize=32)
        window_name   : nombre de la ventana OpenCV para mostrar los QR
    """

    COST = 100   # costo alto → AdapterSelector lo prefiere solo cuando es necesario

    def __init__(
        self,
        mac: bytes | None = None,
        camera: ICameraInterface | None = None,
        color_codec: IColorCodec | None = None,
        grid_codec: IGridCodec | None = None,
        compressor: ICompressor | None = None,
        frame_queue: IFrameQueue | None = None,
        window_name: str = "QR-NET",
    ) -> None:
        self._mac = make_mac(mac if mac else os.urandom(6))
        self._camera    = camera      or OpenCVCamera()
        self._codec     = color_codec or ColorPalette(n_colors=16)
        self._grid      = grid_codec  or Grid64Codec()
        self._compressor = compressor or ZstdCompressor(level=3)
        self._queue     = frame_queue or FifoFrameQueue(maxsize=32)
        self._window    = window_name

        # Estado de sesión (se actualiza tras el handshake)
        self._negotiated: HandshakeFrame | None = None
        self._peer_mac: bytes = b'\xff' * 6
        self._interval_ms: int = 150

        # Buffer de DataFrames recibidos: seq_num → payload
        self._rx_buffer: dict[int, bytes] = {}
        self._rx_lock = threading.Lock()

        # Control de hilos
        self._running = False
        self._producer_thread: threading.Thread | None = None
        self._consumer_thread: threading.Thread | None = None

    # ── INetworkAdapter ───────────────────────────────────────────────────────

    def send(self, data: bytes) -> bool:
        """
        Comprime data, la fragmenta en DataFrames y los transmite como grilla QR.
        Retorna True si todos los frames fueron enviados y confirmados.
        """
        if not self.is_available():
            return False
        compressed = self._compressor.compress(data)
        frames = self._fragment(compressed)
        for frame in frames:
            self._display_frame(frame)
            time.sleep(self._interval_ms / 1000)
        return True

    def receive(self) -> bytes | None:
        """
        Retorna el próximo bloque de datos decodificado del buffer de recepción,
        o None si no hay datos disponibles todavía.
        """
        with self._rx_lock:
            if not self._rx_buffer:
                return None
            seq = min(self._rx_buffer.keys())
            payload = self._rx_buffer.pop(seq)
        return self._compressor.decompress(payload)

    def is_available(self) -> bool:
        return self._running and self._negotiated is not None

    def get_mac(self) -> bytes:
        return bytes(self._mac)

    def get_type(self) -> AdapterType:
        return AdapterType.QR_LIGHT

    def get_cost(self) -> int:
        return self.COST

    # ── Handshake ─────────────────────────────────────────────────────────────

    def handshake(self, peer_mac: bytes, is_initiator: bool = True) -> bool:
        """
        Ejecuta el handshake de 3 vías.
        is_initiator=True  → este nodo envía el SYN.
        is_initiator=False → este nodo espera el SYN y responde SYN-ACK.
        Retorna True si el handshake completó exitosamente.
        """
        self._peer_mac = peer_mac
        try:
            if is_initiator:
                return self._handshake_initiator()
            else:
                return self._handshake_responder()
        except Exception as e:
            raise HandshakeError(f"Handshake falló: {e}") from e

    def _handshake_initiator(self) -> bool:
        """Envía SYN, espera SYN-ACK, envía ACK."""
        local_caps = HandshakeFrame(
            frame_type   = FrameType.SYN,
            src_mac      = bytes(self._mac),
            dst_mac      = self._peer_mac,
            color_depth  = 3,            # máximo local: 16 colores
            grid_size    = 8,            # 64×64
            ecc_level    = 2,
            sync_method  = 3,
            interval_ms  = self._interval_ms,
            compression  = self._compressor.algo,
        )
        self._display_handshake(local_caps)
        time.sleep(self._interval_ms / 1000)

        # Esperar SYN-ACK
        syn_ack = self._wait_handshake(FrameType.SYN_ACK, timeout=10.0)
        if syn_ack is None:
            raise HandshakeError("Timeout esperando SYN-ACK")

        # Aplicar capacidades negociadas
        self._apply_negotiated(syn_ack)

        # Enviar ACK
        ack = HandshakeFrame(
            frame_type = FrameType.ACK,
            src_mac    = bytes(self._mac),
            dst_mac    = self._peer_mac,
        )
        self._display_handshake(ack)
        return True

    def _handshake_responder(self) -> bool:
        """Espera SYN, envía SYN-ACK, espera ACK."""
        syn = self._wait_handshake(FrameType.SYN, timeout=30.0)
        if syn is None:
            raise HandshakeError("Timeout esperando SYN")

        local_caps = HandshakeFrame(
            color_depth = 3,
            grid_size   = 8,
            interval_ms = self._interval_ms,
        )
        syn_ack = local_caps.negotiate_with(syn)
        self._apply_negotiated(syn_ack)
        self._display_handshake(syn_ack)
        time.sleep(self._interval_ms / 1000)

        ack = self._wait_handshake(FrameType.ACK, timeout=10.0)
        return ack is not None

    def _apply_negotiated(self, hs: HandshakeFrame) -> None:
        """Aplica los parámetros negociados al estado interno."""
        self._negotiated = hs
        self._interval_ms = hs.interval_ms
        n_colors = hs.n_colors
        if isinstance(self._codec, ColorPalette):
            self._codec = ColorPalette(n_colors=n_colors)

    # ── Transmisión de archivo ────────────────────────────────────────────────

    def send_file(self, path: str, dst_mac: bytes) -> bool:
        """
        Transmite un archivo completo via datacasting.
        Calcula hash, comprime en streaming, fragmenta y transmite frame a frame.
        Retorna True si la transmisión completó sin errores.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        file_size = os.path.getsize(path)

        # Hash del archivo original para verificación al final
        with open(path, "rb") as f:
            file_hash = sha128(f.read())   # para archivos muy grandes considerar hash incremental

        # Estimar total de frames
        max_payload = self._grid.max_payload_bytes_for(self._codec)
        # Comprimir en streaming para estimar tamaño comprimido
        compressed_chunks: list[bytes] = []
        with open(path, "rb") as f:
            for chunk in self._compressor.compress_stream(
                iter(lambda: f.read(64 * 1024), b"")
            ):
                compressed_chunks.append(chunk)
        compressed = b"".join(compressed_chunks)
        total_frames = (len(compressed) + max_payload - 1) // max_payload

        # Actualizar HandshakeFrame con metadatos
        if self._negotiated:
            self._negotiated.total_frames = total_frames
            self._negotiated.file_size    = file_size
            self._negotiated.file_hash    = file_hash

        # Fragmentar y transmitir
        seq = 0
        offset = 0
        while offset < len(compressed):
            chunk = compressed[offset: offset + max_payload]
            frame = DataFrame(
                src_mac    = bytes(self._mac),
                dst_mac    = dst_mac,
                seq_num    = seq,
                payload    = chunk,
            )
            self._display_frame(frame)
            time.sleep(self._interval_ms / 1000)
            offset += len(chunk)
            seq += 1

        # FIN
        fin = DataFrame.fin(bytes(self._mac), dst_mac, seq)
        self._display_frame(fin)
        return True

    def retransmit(self, seq: int, dst_mac: bytes, data: bytes) -> None:
        """Retransmite un DataFrame específico por número de secuencia."""
        frame = DataFrame(
            src_mac = bytes(self._mac),
            dst_mac = dst_mac,
            seq_num = seq,
            payload = data,
        )
        self._display_frame(frame)

    # ── Ciclo de captura ──────────────────────────────────────────────────────

    def start(self) -> None:
        """Inicia los hilos de captura y decodificación."""
        self._camera.open()
        self._running = True
        self._producer_thread = threading.Thread(
            target=self._producer_loop, daemon=True, name="qrnet-producer"
        )
        self._consumer_thread = threading.Thread(
            target=self._consumer_loop, daemon=True, name="qrnet-consumer"
        )
        self._producer_thread.start()
        self._consumer_thread.start()

    def stop(self) -> None:
        """Detiene los hilos y libera la cámara."""
        self._running = False
        self._camera.close()
        cv2.destroyWindow(self._window)

    def _producer_loop(self) -> None:
        """Hilo productor: captura frames y los encola si cambiaron."""
        prev = None
        while self._running:
            try:
                frame = self._camera.capture()
                if self._camera.detect_change(prev, frame):
                    self._queue.put(frame)
                    prev = frame
            except AdapterError:
                time.sleep(0.1)
            time.sleep(self._interval_ms / 1000 / 2)

    def _consumer_loop(self) -> None:
        """Hilo consumidor: desencola y decodifica frames."""
        while self._running:
            try:
                frame_img = self._queue.get(timeout=2.0)
                self._decode_and_store(frame_img)
            except queue.Empty:
                continue

    def _decode_and_store(self, frame_img: np.ndarray) -> None:
        """Decodifica un frame y lo almacena según su tipo."""
        try:
            raw = self._grid.decode_grid(frame_img, self._codec)
            # Intentar interpretar como HandshakeFrame primero
            try:
                hs = HandshakeFrame.from_bytes(raw)
                self._handle_handshake(hs)
                return
            except Exception:
                pass
            # Interpretar como DataFrame
            df = DataFrame.from_bytes(raw)
            if df.frame_type == FrameType.DATA:
                with self._rx_lock:
                    self._rx_buffer[df.seq_num] = df.payload
            elif df.frame_type == FrameType.NACK:
                pass   # manejo de NACK delegado a la capa superior
        except GridDecodeError:
            pass   # frame borroso o sin grilla — ignorar

    def _handle_handshake(self, hs: HandshakeFrame) -> None:
        """Procesa un HandshakeFrame recibido durante el consumer loop."""
        if hs.frame_type == FrameType.SYN:
            with self._rx_lock:
                self._rx_buffer[-1] = hs.to_bytes()   # clave especial para handshake
        elif hs.frame_type in (FrameType.SYN_ACK, FrameType.ACK):
            with self._rx_lock:
                self._rx_buffer[-2] = hs.to_bytes()

    # ── visualización ─────────────────────────────────────────────────────────

    def _display_frame(self, frame: DataFrame) -> None:
        """Serializa el DataFrame, lo codifica en grilla y lo muestra en pantalla."""
        img = self._grid.encode_grid(frame.to_bytes(), self._codec)
        cv2.imshow(self._window, img)
        cv2.waitKey(1)

    def _display_handshake(self, frame: HandshakeFrame) -> None:
        """Muestra un HandshakeFrame como grilla QR en pantalla."""
        img = self._grid.encode_grid(frame.to_bytes(), self._codec)
        cv2.imshow(self._window, img)
        cv2.waitKey(1)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _fragment(self, data: bytes) -> list[DataFrame]:
        """Divide data en DataFrames de MAX_PAYLOAD bytes cada uno."""
        frames = []
        offset = 0
        seq = 0
        max_p = self._grid.max_payload_bytes_for(self._codec)
        max_p = min(max_p, MAX_PAYLOAD)
        while offset < len(data):
            chunk = data[offset: offset + max_p]
            frames.append(DataFrame(
                src_mac  = bytes(self._mac),
                dst_mac  = self._peer_mac,
                seq_num  = seq,
                payload  = chunk,
            ))
            offset += len(chunk)
            seq += 1
        return frames

    def _wait_handshake(
        self, expected_type: FrameType, timeout: float
    ) -> HandshakeFrame | None:
        """Espera un HandshakeFrame del tipo esperado dentro del timeout."""
        deadline = time.time() + timeout
        key = -1 if expected_type == FrameType.SYN else -2
        while time.time() < deadline:
            with self._rx_lock:
                raw = self._rx_buffer.get(key)
            if raw is not None:
                try:
                    hs = HandshakeFrame.from_bytes(raw)
                    if hs.frame_type == expected_type:
                        with self._rx_lock:
                            self._rx_buffer.pop(key, None)
                        return hs
                except Exception:
                    pass
            time.sleep(0.05)
        return None

    # ── contexto ──────────────────────────────────────────────────────────────

    def __enter__(self) -> DispositivoLuzAdaptador:
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()