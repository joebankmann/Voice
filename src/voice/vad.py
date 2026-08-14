from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Protocol

import numpy as np
import torch


class VadBackend(Protocol):
    def __call__(self, frame: np.ndarray) -> float: ...


class VadEngine:
    def __init__(self, backend: VadBackend, threshold: float) -> None:
        self._backend = backend
        self._threshold = threshold

    def is_speech(self, frame: np.ndarray) -> bool:
        return float(self._backend(frame)) >= self._threshold


class _SileroBackend:
    def __init__(self, model: Callable[..., torch.Tensor], sample_rate: int) -> None:
        self._model = model
        self._sample_rate = sample_rate

    def __call__(self, frame: np.ndarray) -> float:
        audio = torch.from_numpy(np.asarray(frame, dtype=np.float32))
        with torch.inference_mode():
            probability = self._model(audio, self._sample_rate)
        return float(probability.item())


def create_default_vad(
    threshold: float = 0.5,
    sample_rate: int = 16_000,
) -> VadEngine:
    """Create a VAD backed by the locally installed ``silero-vad`` package.

    Model loading is intentionally deferred until this function is called.
    Install the package and pre-populate its model cache while provisioning the
    offline machine; importing :mod:`voice.vad` itself performs no model load.
    """
    try:
        silero_vad = importlib.import_module("silero_vad")
    except ImportError as exc:
        raise RuntimeError(
            "create_default_vad requires the optional 'silero-vad' package"
        ) from exc

    model = silero_vad.load_silero_vad()
    return VadEngine(
        backend=_SileroBackend(model=model, sample_rate=sample_rate),
        threshold=threshold,
    )
