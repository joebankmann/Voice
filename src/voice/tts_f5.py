from __future__ import annotations

import pkgutil
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np

from voice.speak_text import sanitize_for_speech
from voice.voices import VoiceInfo

GenerateFn = Callable[..., Any]


class F5CloneEngine:
    """Resident F5-TTS (MLX) engine with zero-shot voice cloning."""

    SAMPLE_RATE = 24_000

    def __init__(
        self,
        *,
        ref_audio_path: str | Path | None = None,
        ref_text: str | None = None,
        model_name: str = "lucasnewman/f5-tts-mlx",
        steps: int = 8,
        speed: float = 1.0,
        quantization_bits: int | None = 4,
        generate_fn: GenerateFn | None = None,
        model_loader: Callable[..., Any] | None = None,
    ) -> None:
        self.sample_rate = self.SAMPLE_RATE
        self.model_name = model_name
        self.steps = steps
        self.speed = speed
        self.quantization_bits = quantization_bits
        self._generate_fn = generate_fn
        self._model_loader = model_loader
        self._model: Any | None = None
        self._voice_path = ""
        self._ref_text = ref_text or ""
        if ref_audio_path is not None:
            self.set_clone(ref_audio_path, ref_text or "")

    @property
    def voice_path(self) -> str:
        return self._voice_path

    def set_clone(self, ref_audio_path: str | Path, ref_text: str) -> None:
        path = Path(ref_audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"Clone reference audio not found: {path}")
        if not ref_text.strip():
            raise ValueError("Clone profiles require non-empty ref.txt transcript")
        self._voice_path = str(path)
        self._ref_text = ref_text.strip()

    def apply_voice_info(self, voice: VoiceInfo, *, length_scale: float | None = None) -> None:
        del length_scale
        if voice.engine != "f5":
            raise ValueError(f"F5 engine cannot use voice profile engine={voice.engine}")
        self.set_clone(voice.model_path, voice.ref_text or "")

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        if self._model_loader is not None:
            self._model = self._model_loader(
                self.model_name,
                quantization_bits=self.quantization_bits,
            )
            return self._model
        from f5_tts_mlx.cfm import F5TTS

        self._model = F5TTS.from_pretrained(
            self.model_name,
            quantization_bits=self.quantization_bits,
        )
        return self._model

    def synthesize(self, text: str) -> bytes:
        spoken = sanitize_for_speech(text)
        if not spoken:
            return b""

        if self._generate_fn is not None:
            # Test/injectable path: return raw PCM float32 or int16 bytes.
            result = self._generate_fn(
                spoken,
                ref_audio_path=self._voice_path or None,
                ref_audio_text=self._ref_text or None,
            )
            if isinstance(result, bytes):
                return result
            samples = np.asarray(result, dtype=np.float32).reshape(-1)
            pcm = np.clip(samples, -1.0, 1.0)
            return (pcm * np.iinfo(np.int16).max).astype("<i2").tobytes()

        import mlx.core as mx
        import soundfile as sf
        from f5_tts_mlx.utils import convert_char_to_pinyin

        ref_path = self._voice_path
        ref_text = self._ref_text
        if not ref_path:
            data = pkgutil.get_data("f5_tts_mlx", "tests/test_en_1_ref_short.wav")
            if data is None:
                raise RuntimeError("No F5 reference audio configured and package default missing")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
                handle.write(data)
                ref_path = handle.name
            ref_text = "Some call me nature, others call me mother nature."

        audio, sr = sf.read(ref_path)
        if sr != self.SAMPLE_RATE:
            raise ValueError(
                f"Clone reference must be {self.SAMPLE_RATE} Hz mono WAV (got {sr})"
            )
        audio_mx = mx.array(np.asarray(audio, dtype=np.float32))
        model = self._ensure_model()
        prompt = convert_char_to_pinyin([f"{ref_text} {spoken}"])
        wave, _ = model.sample(
            mx.expand_dims(audio_mx, axis=0),
            text=prompt,
            duration=None,
            steps=self.steps,
            method="euler",
            speed=self.speed,
            cfg_strength=2.0,
            sway_sampling_coef=-1.0,
            seed=None,
        )
        wave = wave[audio_mx.shape[0] :]
        mx.eval(wave)
        samples = np.array(wave, dtype=np.float32).reshape(-1)
        pcm = np.clip(samples, -1.0, 1.0)
        return (pcm * np.iinfo(np.int16).max).astype("<i2").tobytes()
