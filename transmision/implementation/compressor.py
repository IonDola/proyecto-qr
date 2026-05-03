"""
capa1/compression.py
Implementación de ICompressor con zstd (Zstandard) en modo streaming.

Ventajas de zstd para este caso de uso:
- Soporte nativo de streaming → archivos de 1 GB sin cargar en RAM
- ~500 MB/s de compresión, ~1700 MB/s de descompresión
- Ratio típico 40-70% según tipo de dato
- compress_stream / decompress_stream son generadores — se integran
  directamente con el pipeline de fragmentación del adaptador.
"""
from __future__ import annotations
from typing import Generator, Iterable

import zstandard as zstd

from common.other import CompressionAlgorithm
from common.exceptions import CompressionError, DecompressionError
from ..interfaces import ICompressor

# Tamaño de chunk para el streaming (64 KB — balance entre latencia y ratio)
CHUNK_SIZE = 64 * 1024


class ZstdCompressor(ICompressor):
    """
    Compresor zstd con nivel configurable usando zstandard 0.25.x.
    Nivel 3 es el punto óptimo para QR-NET:
    buen ratio sin sacrificar velocidad de compresión.
    """

    def __init__(self, level: int = 3) -> None:
        if not (1 <= level <= 22):
            raise ValueError(f"Nivel zstd debe estar entre 1 y 22, recibió {level}")
        self._level = level
        # Instancias reutilizables — thread-safe en zstandard 0.25
        self._cctx = zstd.ZstdCompressor(level=level, write_content_size=True)
        self._dctx = zstd.ZstdDecompressor()

    # ── ICompressor ───────────────────────────────────────────────────────────

    def compress(self, data: bytes) -> bytes:
        """Comprime data en memoria. write_content_size=True permite decompress() sin max_length."""
        try:
            return self._cctx.compress(data)
        except Exception as e:
            raise CompressionError(f"Error al comprimir con zstd: {e}") from e

    def decompress(self, data: bytes) -> bytes:
        """
        Descomprime data en memoria.
        Soporta frames con y sin content_size en el header:
        compress() produce frames con content_size,
        compress_stream() produce frames sin él.
        """
        import io
        try:
            return self._dctx.decompress(data)
        except zstd.ZstdError:
            # Frame sin content_size — stream_reader no requiere conocer el tamaño
            try:
                with self._dctx.stream_reader(io.BytesIO(data)) as reader:
                    return reader.read()
            except Exception as e2:
                raise DecompressionError(f"Error al descomprimir con zstd: {e2}") from e2

    def compress_stream(
        self, chunks: Iterable[bytes]
    ) -> Generator[bytes, None, None]:
        """
        Generador: recibe iterable de chunks de bytes, yield de chunks comprimidos.
        Permite comprimir un archivo de 1 GB chunk a chunk sin cargarlo en RAM.

        Uso:
            with open("archivo.bin", "rb") as f:
                src = iter(lambda: f.read(CHUNK_SIZE), b"")
                for compressed_chunk in compressor.compress_stream(src):
                    queue.put(compressed_chunk)

        Nota: write_content_size=False en streaming porque el tamaño
        total no se conoce de antemano. decompress_stream usa read_to_iter
        que no requiere content_size en el header.
        """
        try:
            cctx = zstd.ZstdCompressor(level=self._level, write_content_size=False)
            chunker = cctx.chunker(chunk_size=CHUNK_SIZE)
            for chunk in chunks:
                for out in chunker.compress(chunk):
                    yield out
            for out in chunker.finish():
                yield out
        except Exception as e:
            raise CompressionError(f"Error en streaming zstd compress: {e}") from e

    def decompress_stream(
        self, chunks: Iterable[bytes]
    ) -> Generator[bytes, None, None]:
        """
        Generador inverso: recibe chunks comprimidos, yield de chunks originales.
        Usa read_to_iter que no requiere content_size en el frame header.
        """
        try:
            import io
            buf = b"".join(chunks)
            reader = self._dctx.stream_reader(io.BytesIO(buf))
            while True:
                out = reader.read(CHUNK_SIZE)
                if not out:
                    break
                yield out
        except Exception as e:
            raise CompressionError(f"Error en streaming zstd decompress: {e}") from e

    @property
    def compression_algorithm(self) -> CompressionAlgorithm:
        return CompressionAlgorithm.ZSTD

    @property
    def level(self) -> int:
        return self._level