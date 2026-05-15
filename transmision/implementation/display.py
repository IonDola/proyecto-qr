from __future__ import annotations

import cv2
import numpy as np

from transmision.interfaces import IFrameRenderer


class OpenCVFrameRenderer(IFrameRenderer):

    def __init__(self, window_name: str = "QR-NET") -> None:
        self._window_name = window_name

    def show(self, frame: np.ndarray) -> None:
        cv2.imshow(self._window_name, frame)
        cv2.waitKey(1)

    def close(self) -> None:
        cv2.destroyWindow(self._window_name)