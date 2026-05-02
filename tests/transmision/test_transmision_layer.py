"""

tests/transmision/test_transmision_layer.py

Suite de tests unitarios para toda la Capa 1 de QR-NET.


Cubre:

  - common/checksum.py

  - common/other.py

  - transmision/frames.py        (HandshakeFrame, DataFrame)

  - transmision/color_palette.py (ColorPalette)

  - transmision/grid.py          (Grid64Codec, _bytes_to_symbols, _symbols_to_bytes)

  - transmision/compression.py   (ZstdCompressor)

  - transmision/fifo_queue.py    (FifoFrameQueue)

  - transmision/selector.py      (AdapterSelector)


No requiere cámara física — los tests de grilla usan imágenes sintéticas.

"""

from __future__ import annotations

import queue

import struct

import threading
import time

from unittest.mock import MagicMock


import numpy as np
import pytest


# ── imports del proyecto ──────────────────────────────────────────────────────

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from common.checksum import crc16, crc32, sha128, verify_crc16

from common.network_policies import AdapterType, FrameType, SelectionPolicy

from common.other import CompressionAlgorithm

from common.exceptions import ChecksumError, GridDecodeError, AdapterError


from transmision.frames import HandshakeFrame, DataFrame, MAX_PAYLOAD

from transmision.implementation.color_palette import ColorPalette, _rgb_to_hsv, _hsv_distance

from transmision.implementation.grid import Grid64Codec, _bytes_to_symbols, _symbols_to_bytes, GRID_MODULES

from transmision.implementation.compressor import ZstdCompressor

from transmision.implementation.queue import FifoFrameQueue, DropPolicy

from transmision.implementation.selector import AdapterSelector

from transmision.interfaces import INetworkAdapter



# ══════════════════════════════════════════════════════════════════════════════

# common/checksum

# ══════════════════════════════════════════════════════════════════════════════


class TestChecksum:

    def test_crc16_known_value(self):

        # CRC-16/CCITT-FALSE de b"123456789" = 0x29B1

        assert crc16(b"123456789") == 0x29B1


    def test_crc16_empty(self):

        assert isinstance(crc16(b""), int)


    def test_crc32_known_value(self):

        assert crc32(b"hello") == 0x3610A686


    def test_sha128_length(self):

        digest = sha128(b"qrnet")

        assert len(digest) == 16


    def test_sha128_deterministic(self):

        assert sha128(b"test") == sha128(b"test")


    def test_verify_crc16_ok(self):

        data = b"payload de prueba"

        assert verify_crc16(data, crc16(data))


    def test_verify_crc16_fail(self):

        assert not verify_crc16(b"data", 0xDEAD)



# ══════════════════════════════════════════════════════════════════════════════

# transmision/frames — HandshakeFrame

# ══════════════════════════════════════════════════════════════════════════════


class TestHandshakeFrame:

    def _make(self, **kwargs) -> HandshakeFrame:

        return HandshakeFrame(

            src_mac = b'\x01\x02\x03\x04\x05\x06',

            dst_mac = b'\xFF\xFF\xFF\xFF\xFF\xFF',

            **kwargs,
        )


    def test_roundtrip(self):

        hs = self._make(color_depth=3, interval_ms=200)

        recovered = HandshakeFrame.from_bytes(hs.to_bytes())

        assert recovered.color_depth == 3

        assert recovered.interval_ms == 200

        assert recovered.src_mac == b'\x01\x02\x03\x04\x05\x06'


    def test_checksum_detection(self):

        raw = bytearray(self._make().to_bytes())

        raw[-1] ^= 0xFF           # corromper último byte (checksum)

        with pytest.raises(ChecksumError):

            HandshakeFrame.from_bytes(bytes(raw))


    def test_magic_detection(self):

        raw = bytearray(self._make().to_bytes())

        raw[1] ^= 0xFF            # corromper MAGIC

        with pytest.raises((ChecksumError, ValueError)):

            HandshakeFrame.from_bytes(bytes(raw))


    def test_n_colors_property(self):

        for cd, expected in [(0, 2), (1, 4), (2, 8), (3, 16)]:

            assert self._make(color_depth=cd).n_colors == expected


    def test_negotiate_takes_minimum(self):

        local  = self._make(color_depth=3, interval_ms=100)

        remote = HandshakeFrame(color_depth=1, interval_ms=200, src_mac=b'\xAA'*6, dst_mac=b'\x01'*6)

        agreed = local.negotiate_with(remote)

        assert agreed.color_depth == 1        # mínimo

        assert agreed.interval_ms == 200      # máximo (más conservador)


    def test_verify(self):

        hs = self._make()

        assert hs.verify()


    def test_frame_type_syn(self):

        hs = self._make(frame_type=FrameType.SYN)

        recovered = HandshakeFrame.from_bytes(hs.to_bytes())

        assert recovered.frame_type == FrameType.SYN



# ══════════════════════════════════════════════════════════════════════════════

# transmision/frames — DataFrame

# ══════════════════════════════════════════════════════════════════════════════


class TestDataFrame:

    SRC = b'\xAA\xBB\xCC\xDD\xEE\xFF'

    DST = b'\x11\x22\x33\x44\x55\x66'


    def test_roundtrip_data(self):

        payload = b"hola QR-NET" * 5

        df = DataFrame(src_mac=self.SRC, dst_mac=self.DST, seq_num=42, payload=payload)

        recovered = DataFrame.from_bytes(df.to_bytes())

        assert recovered.payload == payload

        assert recovered.seq_num == 42


    def test_max_payload_accepted(self):

        df = DataFrame(payload=b'x' * MAX_PAYLOAD)

        assert df.verify()


    def test_payload_too_large(self):

        with pytest.raises(ValueError):

            DataFrame(payload=b'x' * (MAX_PAYLOAD + 1))


    def test_checksum_detection(self):

        raw = bytearray(DataFrame(payload=b"test").to_bytes())

        raw[-1] ^= 0xFF

        with pytest.raises(ChecksumError):

            DataFrame.from_bytes(bytes(raw))


    def test_empty_payload(self):

        df = DataFrame(src_mac=self.SRC, dst_mac=self.DST)

        recovered = DataFrame.from_bytes(df.to_bytes())

        assert recovered.payload == b''


    def test_nack_factory(self):

        nack = DataFrame.nack(self.SRC, self.DST, missing_seq=7)

        assert nack.frame_type == FrameType.NACK

        assert nack.seq_num == 7


    def test_fin_factory(self):

        fin = DataFrame.fin(self.SRC, self.DST, seq=99)

        assert fin.frame_type == FrameType.FIN



# ══════════════════════════════════════════════════════════════════════════════

# transmision/color_palette

# ══════════════════════════════════════════════════════════════════════════════


class TestColorPalette:

    def test_invalid_n_colors(self):

        with pytest.raises(ValueError):

            ColorPalette(n_colors=5)


    @pytest.mark.parametrize("n", [2, 4, 8, 16])

    def test_encode_decode_identity(self, n):

        """Para cada color depth, encode→decode debe recuperar el nibble."""

        palette = ColorPalette(n_colors=n)

        for nibble in range(n):

            color = palette.encode(nibble)
            decoded = palette.decode(color)

            assert decoded == nibble, f"n={n} nibble={nibble} color={color} decoded={decoded}"


    def test_encode_out_of_range(self):

        p = ColorPalette(16)

        with pytest.raises(ValueError):

            p.encode(16)


    def test_bits_per_cell(self):

        assert ColorPalette(2).bits_per_cell  == 1

        assert ColorPalette(4).bits_per_cell  == 2

        assert ColorPalette(8).bits_per_cell  == 3

        assert ColorPalette(16).bits_per_cell == 4


    def test_calibrate_updates_palette(self):

        p = ColorPalette(16)

        original = p.palette_rgb()[0]

        # Crear parche sintético con colores distintos

        patch = np.zeros((4, 4, 3), dtype=np.uint8)

        patch[0, 0] = (10, 20, 30)   # BGR

        p.calibrate(patch)

        # El primer color calibrado debe haber cambiado

        assert p.palette_rgb()[0] != original


    def test_reset_calibration(self):

        p = ColorPalette(16)

        original = p.palette_rgb()

        patch = np.random.randint(0, 255, (4, 4, 3), dtype=np.uint8)

        p.calibrate(patch)

        p.reset_calibration()

        assert p.palette_rgb() == original


    def test_hsv_distance_same_color(self):

        assert _hsv_distance(0.5, 0.8, 0.9, 0.5, 0.8, 0.9) == pytest.approx(0.0)


    def test_hsv_distance_circular_hue(self):

        # hue 0.0 y 1.0 son el mismo color → distancia 0

        d = _hsv_distance(0.0, 1.0, 1.0, 1.0, 1.0, 1.0)

        assert d == pytest.approx(0.0, abs=1e-9)



# ══════════════════════════════════════════════════════════════════════════════

# transmision/grid — conversión de símbolos

# ══════════════════════════════════════════════════════════════════════════════


class TestSymbolConversion:

    @pytest.mark.parametrize("bpc", [1, 2, 3, 4])

    def test_roundtrip(self, bpc):

        data = bytes(range(32))

        symbols = _bytes_to_symbols(data, bpc)

        recovered = _symbols_to_bytes(symbols, bpc, len(data))

        assert recovered == data


    def test_nibble_split(self):

        # 0xAB = 1010 1011 → nibbles 0xA, 0xB

        syms = _bytes_to_symbols(b'\xAB', 4)

        assert syms[0] == 0xA

        assert syms[1] == 0xB


    def test_bit_split(self):

        # 0b10110010 → bits [1,0,1,1,0,0,1,0]

        syms = _bytes_to_symbols(b'\xB2', 1)

        assert syms == [1, 0, 1, 1, 0, 0, 1, 0]



# ══════════════════════════════════════════════════════════════════════════════

# transmision/grid — Grid64Codec (sintético, sin cámara)

# ══════════════════════════════════════════════════════════════════════════════


class TestGrid64Codec:

    def setup_method(self):

        self.codec = Grid64Codec(module_px=4)   # módulos pequeños para test rápido

        self.palette = ColorPalette(n_colors=16)


    def test_payload_mask_shape(self):

        assert self.codec._payload_mask.shape == (GRID_MODULES, GRID_MODULES)


    def test_payload_positions_count(self):
        n = len(self.codec._payload_positions)

        # debe haber entre 2000 y 4000 módulos de payload

        assert 2000 < n < 4000


    def test_max_payload_bytes_positive(self):

        assert self.codec.max_payload_bytes > 0


    def test_encode_returns_correct_shape(self):

        payload = b"test payload 123"

        img = self.codec.encode_grid(payload, self.palette)

        assert img.dtype == np.uint8

        assert img.ndim == 3

        assert img.shape[2] == 3   # BGR


    def test_encode_payload_too_large(self):

        big = b'x' * (self.codec.max_payload_bytes + 1)

        with pytest.raises(ValueError):

            self.codec.encode_grid(big, self.palette)


    def test_encode_decode_synthetic(self):

        """

        Test de integración encode→decode sobre imagen perfecta (sin ruido).

        Usa la misma imagen generada por encode como entrada de decode

        para validar que el pipeline completo funciona.

        """

        payload = b"QR-NET test payload 2026"

        img = self.codec.encode_grid(payload, self.palette)


        # Para decode necesitamos que align_grid encuentre la grilla.

        # En tests usamos la imagen directamente (ya alineada).

        # Mockeamos _align_grid para que retorne la imagen tal cual.

        original_align = self.codec._align_grid

        self.codec._align_grid = lambda im: im


        try:

            recovered = self.codec.decode_grid(img, self.palette)

            assert recovered == payload

        finally:

            self.codec._align_grid = original_align



# ══════════════════════════════════════════════════════════════════════════════

# transmision/compression

# ══════════════════════════════════════════════════════════════════════════════


class TestZstdCompressor:

    def test_compress_decompress(self):

        comp = ZstdCompressor(level=3)

        data = b"datos de prueba " * 100
        assert comp.decompress(comp.compress(data)) == data


    def test_compress_reduces_size(self):

        comp = ZstdCompressor()

        data = b"aaaa" * 1000       # muy comprimible

        assert len(comp.compress(data)) < len(data)


    def test_algo_property(self):

        assert ZstdCompressor().compression_algorithm == CompressionAlgorithm.ZSTD


    def test_invalid_level(self):

        with pytest.raises(ValueError):

            ZstdCompressor(level=0)


    def test_compress_stream(self):

        comp = ZstdCompressor()

        data = b"streaming test " * 500

        chunks = [data[i:i+64] for i in range(0, len(data), 64)]

        compressed = b"".join(comp.compress_stream(iter(chunks)))
        assert comp.decompress(compressed) == data


    def test_large_data(self):

        comp = ZstdCompressor()

        data = bytes(range(256)) * 4000   # ~1 MB

        recovered = comp.decompress(comp.compress(data))

        assert recovered == data



# ══════════════════════════════════════════════════════════════════════════════

# transmision/fifo_queue

# ══════════════════════════════════════════════════════════════════════════════


class TestFifoFrameQueue:

    def _frame(self, val: int = 0) -> np.ndarray:

        return np.full((4, 4, 3), val, dtype=np.uint8)


    def test_put_get(self):

        q = FifoFrameQueue(maxsize=4, hash_dedup=False)

        f = self._frame(42)

        assert q.put(f)

        out = q.get(timeout=0.5)

        assert np.array_equal(out, f)


    def test_drop_oldest_when_full(self):

        q = FifoFrameQueue(maxsize=2, policy=DropPolicy.DROP_OLDEST, hash_dedup=False)

        q.put(self._frame(1))

        q.put(self._frame(2))

        q.put(self._frame(3))   # debería descartar frame 1

        assert q.qsize() == 2

        first = q.get(timeout=0.5)

        # El frame 1 fue descartado → el primero ahora es 2 o 3

        assert first is not None


    def test_discard_new_when_full(self):

        q = FifoFrameQueue(maxsize=2, policy=DropPolicy.DISCARD_NEW, hash_dedup=False)

        q.put(self._frame(1))

        q.put(self._frame(2))

        result = q.put(self._frame(3))   # debe ser descartado

        assert not result

        assert q.qsize() == 2


    def test_hash_dedup(self):

        q = FifoFrameQueue(maxsize=8, hash_dedup=True)

        f = self._frame(7)

        q.put(f)

        result = q.put(f)    # mismo frame → descartado

        assert not result

        assert q.qsize() == 1


    def test_get_timeout(self):

        q = FifoFrameQueue()

        with pytest.raises(queue.Empty):

            q.get(timeout=0.1)


    def test_clear(self):

        q = FifoFrameQueue(maxsize=4, hash_dedup=False)

        q.put(self._frame(1))

        q.put(self._frame(2))

        q.clear()

        assert q.qsize() == 0


    def test_producer_consumer_threads(self):

        """Test de integración: productor y consumidor en hilos separados."""

        q = FifoFrameQueue(maxsize=16, hash_dedup=False)

        received: list[np.ndarray] = []

        N = 10


        def producer():

            for i in range(N):

                q.put(self._frame(i))

                time.sleep(0.01)


        def consumer():

            for _ in range(N):

                try:

                    f = q.get(timeout=2.0)

                    received.append(f)

                except queue.Empty:

                    break


        t1 = threading.Thread(target=producer)

        t2 = threading.Thread(target=consumer)

        t1.start(); t2.start()

        t1.join(); t2.join()

        assert len(received) == N


    def test_drop_rate(self):

        q = FifoFrameQueue(maxsize=1, policy=DropPolicy.DISCARD_NEW, hash_dedup=False)

        q.put(self._frame(1))

        q.put(self._frame(2))   # descartado

        assert q.drop_rate > 0.0



# ══════════════════════════════════════════════════════════════════════════════

# transmision/selector

# ══════════════════════════════════════════════════════════════════════════════


def _mock_adapter(atype: AdapterType, cost: int, mac: bytes, available: bool = True) -> INetworkAdapter:

    a = MagicMock(spec=INetworkAdapter)

    a.get_type.return_value    = atype

    a.get_cost.return_value    = cost

    a.get_mac.return_value     = mac

    a.is_available.return_value = available

    return a



class TestAdapterSelector:

    QR_MAC  = b'\x01\x00\x00\x00\x00\x01'

    TCP_MAC = b'\x02\x00\x00\x00\x00\x02'

    DST     = b'\x03\x00\x00\x00\x00\x03'


    def _sel(self, policy=SelectionPolicy.PREFER_QR):

        s = AdapterSelector(policy=policy)

        s.register(_mock_adapter(AdapterType.QR_LIGHT, 100, self.QR_MAC))

        s.register(_mock_adapter(AdapterType.ETHERNET, 10,  self.TCP_MAC))

        return s


    def test_prefer_qr(self):

        sel = self._sel(SelectionPolicy.PREFER_QR)

        chosen = sel.select(self.DST)

        assert chosen.get_type() == AdapterType.QR_LIGHT


    def test_prefer_tcp(self):

        sel = self._sel(SelectionPolicy.PREFER_TCP)

        chosen = sel.select(self.DST)

        assert chosen.get_type() == AdapterType.ETHERNET


    def test_lowest_cost(self):

        sel = self._sel(SelectionPolicy.LOWEST_COST)

        chosen = sel.select(self.DST)

        assert chosen.get_cost() == 10   # Ethernet tiene costo 10


    def test_broadcast_forces_qr(self):

        sel = self._sel(SelectionPolicy.PREFER_TCP)   # aunque prefiera TCP

        chosen = sel.select(b'\xff\xff\xff\xff\xff\xff')

        assert chosen.get_type() == AdapterType.QR_LIGHT


    def test_gateway_forces_qr(self):

        sel = self._sel(SelectionPolicy.PREFER_TCP)

        gateway_mac = b'\xAA\xBB\xCC\xDD\xEE\xFF'

        sel.register_qr_gateway(gateway_mac)

        chosen = sel.select(gateway_mac)

        assert chosen.get_type() == AdapterType.QR_LIGHT


    def test_no_available_raises(self):

        sel = AdapterSelector()

        sel.register(_mock_adapter(AdapterType.QR_LIGHT, 100, self.QR_MAC, available=False))

        with pytest.raises(AdapterError):

            sel.select(self.DST)


    def test_register_replaces_same_mac(self):

        sel = AdapterSelector()

        a1 = _mock_adapter(AdapterType.QR_LIGHT, 100, self.QR_MAC)

        a2 = _mock_adapter(AdapterType.QR_LIGHT, 50,  self.QR_MAC)

        sel.register(a1)

        sel.register(a2)

        assert len(sel._adapters) == 1

        assert sel._adapters[0].get_cost() == 50


    def test_set_policy_runtime(self):

        sel = self._sel(SelectionPolicy.PREFER_QR)

        sel.set_policy(SelectionPolicy.PREFER_TCP)

        chosen = sel.select(self.DST)

        assert chosen.get_type() == AdapterType.ETHERNET


    def test_first_available(self):

        sel = AdapterSelector(SelectionPolicy.FIRST_AVAILABLE)

        sel.register(_mock_adapter(AdapterType.ETHERNET, 10, self.TCP_MAC))

        chosen = sel.select(self.DST)

        assert chosen is not None