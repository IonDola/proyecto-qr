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

    Compresor zstd con nivel configurable.

    Nivel 3 es el punto óptimo para QR-NET:

    buen ratio sin sacrificar velocidad de compresión.
    """


    def __init__(self, level: int = 3) -> None:

        if not (1 <= level <= 22):

            raise ValueError(f"Nivel zstd debe estar entre 1 y 22, recibió {level}")

        self._level = level


    # ── ICompressor ───────────────────────────────────────────────────────────


    def compress(self, data: bytes) -> bytes:
        """Comprime data en memoria. Para bloques pequeños (cabeceras, etc.)."""
        try:
            cctx = zstd.ZstdCompressor(level=self._level, write_content_size=True)
            return cctx.compress(data)
        except Exception as e:

            raise CompressionError(f"Error al comprimir con zstd: {e}") from e


    def decompress(self, data: bytes) -> bytes:
        try:
            dctx = zstd.ZstdDecompressor()
            return dctx.decompress(data)
        except Exception as e:
            raise CompressionError(f"Error al descomprimir con zstd: {e}") from e


    def compress_stream(

        self, chunks: Iterable[bytes]

    ) -> Generator[bytes, None, None]:
        """

        Generador: recibe iterable de chunks de bytes, yield de chunks comprimidos.

        Permite comprimir un archivo de 1 GB chunk a chunk sin cargarlo en RAM.


        Uso:

            with open("archivo.bin", "rb") as f:

                gen = compressor.compress_stream(iter(lambda: f.read(CHUNK_SIZE), b""))

                for compressed_chunk in gen:

                    queue.put(compressed_chunk)
        """

        try:
            cctx = zstd.ZstdCompressor(level=self._level, write_content_size=False)
            chunker = cctx.chunker(chunk_size=CHUNK_SIZE)
            for chunk in chunks:
                for compressed in chunker.compress(chunk):
                    yield compressed
            for compressed in chunker.finish():
                yield compressed
        except Exception as e:
            raise CompressionError(f"Error en streaming zstd compress: {e}") from e


    def decompress_stream(

        self, chunks: Iterable[bytes]

    ) -> Generator[bytes, None, None]:
        """

        Generador inverso: recibe chunks comprimidos, yield de chunks originales.
        """

        try:

            dctx = zstd.ZstdDecompressor()

            with dctx.stream_writer(None) as writer:

                for chunk in chunks:

                    writer.write(chunk)

                    buf = writer.flush()   # type: ignore[attr-defined]

                    if buf:

                        yield buf

        except Exception as e:

            raise DecompressionError(f"Error en streaming zstd decompress: {e}") from e


    @property

    def compression_algorithm(self) -> CompressionAlgorithm:

        return CompressionAlgorithm.ZSTD


    @property

    def level(self) -> int:

        return self._level