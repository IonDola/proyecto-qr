"""
transmision/factory.py

Factories centralizadas para construir implementaciones concretas
sin exponerlas a capas superiores.
"""

from __future__ import annotations

from transmision.interfaces import (
    ICameraInterface,
    IColorCodec,
    ICompressor,
    IGridCodec,
    IFrameQueue,
    IAdapterSelector,
    INetworkAdapter,
)

from transmision.implementation import (
    OpenCVCamera,
    ColorPalette,
    ZstdCompressor,
    Grid64Codec,
    FifoFrameQueue,
    AdapterSelector,
    QRLightAdapter,
    OpenCVFrameRenderer,
)


def create_camera() -> ICameraInterface:
    return OpenCVCamera()


def create_color_codec() -> IColorCodec:
    return ColorPalette()


def create_grid_codec() -> IGridCodec:
    return Grid64Codec()


def create_compressor() -> ICompressor:
    return ZstdCompressor()


def create_frame_queue() -> IFrameQueue:
    return FifoFrameQueue()


def create_adapter_selector() -> IAdapterSelector:
    return AdapterSelector()


def create_qr_adapter(mac: bytes) -> INetworkAdapter:

    camera = create_camera()
    color = create_color_codec()
    grid = create_grid_codec()
    queue = create_frame_queue()
    renderer = OpenCVFrameRenderer()

    return QRLightAdapter(
        grid_codec=grid,
        color_codec=color,
        camera=camera,
        frame_queue=queue,
        mac=mac,
        renderer=renderer,
    )