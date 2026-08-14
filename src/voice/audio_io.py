from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import sounddevice as sd


StreamFactory = Callable[..., Any]


class AudioHub:
    """Own microphone capture and queued speaker playback streams."""

    def __init__(
        self,
        sample_rate: int,
        frame_samples: int = 512,
        *,
        input_stream_factory: StreamFactory = sd.InputStream,
        output_stream_factory: StreamFactory = sd.OutputStream,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self._mic_frames: queue.Queue[np.ndarray] = queue.Queue(maxsize=32)
        self._playback: list[np.ndarray] = []
        self._current: np.ndarray | None = None
        self._current_offset = 0
        self._playback_lock = threading.Lock()
        self._input_stream = input_stream_factory(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=frame_samples,
            callback=self._on_input,
        )
        self._output_stream = output_stream_factory(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=frame_samples,
            callback=self._on_output,
        )

    def start(self) -> None:
        self._input_stream.start()
        self._output_stream.start()

    def read_frame(self, timeout: float | None = None) -> np.ndarray:
        return self._mic_frames.get(timeout=timeout)

    def mic_frames(self) -> Iterator[np.ndarray]:
        while True:
            yield self.read_frame()

    def play(
        self,
        audio: bytes | np.ndarray,
        *,
        sample_rate: int | None = None,
    ) -> None:
        if isinstance(audio, bytes):
            samples = np.frombuffer(audio, dtype="<i2").astype(np.float32) / 32768.0
        else:
            samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        source_rate = self.sample_rate if sample_rate is None else sample_rate
        if source_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if samples.size and source_rate != self.sample_rate:
            output_size = round(samples.size * self.sample_rate / source_rate)
            source_times = np.arange(samples.size) / source_rate
            output_times = np.arange(output_size) / self.sample_rate
            samples = np.interp(
                output_times,
                source_times,
                samples,
                right=float(samples[-1]),
            ).astype(np.float32)
        if samples.size:
            with self._playback_lock:
                self._playback.append(samples.copy())

    def stop_playback(self) -> None:
        with self._playback_lock:
            self._playback.clear()
            self._current = None
            self._current_offset = 0

    def close(self) -> None:
        self.stop_playback()
        self._input_stream.stop()
        self._output_stream.stop()
        self._input_stream.close()
        self._output_stream.close()

    def __enter__(self) -> AudioHub:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _on_input(
        self,
        indata: np.ndarray,
        frames: int,
        time: Any,
        status: sd.CallbackFlags,
    ) -> None:
        del frames, time, status
        frame = np.asarray(indata[:, 0], dtype=np.float32).copy()
        try:
            self._mic_frames.put_nowait(frame)
        except queue.Full:
            try:
                self._mic_frames.get_nowait()
            except queue.Empty:
                pass
            self._mic_frames.put_nowait(frame)

    def _on_output(
        self,
        outdata: np.ndarray,
        frames: int,
        time: Any,
        status: sd.CallbackFlags,
    ) -> None:
        del time, status
        outdata.fill(0)
        written = 0
        with self._playback_lock:
            while written < frames:
                if self._current is None:
                    if not self._playback:
                        break
                    self._current = self._playback.pop(0)
                    self._current_offset = 0

                available = self._current.size - self._current_offset
                count = min(frames - written, available)
                end = self._current_offset + count
                outdata[written : written + count, 0] = self._current[
                    self._current_offset : end
                ]
                written += count
                self._current_offset = end
                if self._current_offset == self._current.size:
                    self._current = None
                    self._current_offset = 0
