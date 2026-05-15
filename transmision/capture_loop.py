from __future__ import annotations

import time

from transmision.interfaces import (
    ICameraInterface,
    IFrameQueue,
)


class CameraCaptureLoop:

    def __init__(
        self,
        camera: ICameraInterface,
        queue: IFrameQueue,
    ) -> None:

        self._camera = camera
        self._queue = queue

        self._running = False

    def run(self) -> None:

        self._camera.open()

        self._running = True

        prev = None

        while self._running:

            frame = self._camera.capture()

            if prev is None or self._camera.detect_change(prev, frame):
                self._queue.put(frame)

            prev = frame

            time.sleep(1.0 / self._camera.fps)

    def stop(self) -> None:

        self._running = False
        self._camera.close()