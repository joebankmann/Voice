from __future__ import annotations

from pathlib import Path
from typing import Any

from voice.config import AppConfig, TtsConfig
from voice.tts import PiperTtsEngine
from voice.tts_f5 import F5CloneEngine
from voice.voices import VoiceInfo


class HybridTtsEngine:
    """Prefer F5 clone synthesis; fall back to Piper if F5 fails at runtime."""

    def __init__(self, primary: F5CloneEngine, fallback: PiperTtsEngine) -> None:
        self._primary = primary
        self._fallback = fallback
        self._use_fallback = False
        self.sample_rate = primary.sample_rate

    @property
    def voice_path(self) -> str:
        if self._use_fallback:
            return self._fallback.voice_path
        return self._primary.voice_path

    def apply_voice_info(self, voice: VoiceInfo, *, length_scale: float | None = None) -> None:
        if voice.engine == "f5":
            self._use_fallback = False
            self._primary.apply_voice_info(voice, length_scale=length_scale)
            self.sample_rate = self._primary.sample_rate
            return
        self._use_fallback = True
        self._fallback.apply_voice_info(voice, length_scale=length_scale)
        self.sample_rate = self._fallback.sample_rate

    def synthesize(self, text: str) -> bytes:
        if self._use_fallback:
            return self._fallback.synthesize(text)
        try:
            audio = self._primary.synthesize(text)
            self.sample_rate = self._primary.sample_rate
            return audio
        except Exception:
            self._use_fallback = True
            self.sample_rate = self._fallback.sample_rate
            return self._fallback.synthesize(text)

    def warmup(self) -> None:
        self.synthesize("Ready.")


def create_tts_engine(
    config: AppConfig,
    *,
    selected: VoiceInfo | None,
    config_dir: Path,
) -> Any:
    """Build the configured TTS backend, optionally seeded with a voice profile."""
    tts_cfg = config.tts
    piper = PiperTtsEngine(
        tts_cfg.piper_bin,
        str(
            selected.model_path
            if selected is not None and selected.engine == "piper"
            else _resolve(config_dir, tts_cfg.voice_path)
        ),
        sample_rate=(
            selected.sample_rate
            if selected is not None and selected.engine == "piper"
            else tts_cfg.sample_rate
        ),
        length_scale=tts_cfg.length_scale,
        sentence_silence=tts_cfg.sentence_silence,
        noise_scale=tts_cfg.noise_scale,
        noise_w_scale=tts_cfg.noise_w_scale,
    )

    backend = tts_cfg.backend.lower().strip()
    if backend == "piper":
        return piper

    ref_path = None
    ref_text = None
    if selected is not None and selected.engine == "f5":
        ref_path = selected.model_path
        ref_text = selected.ref_text
    elif tts_cfg.clone_ref_wav:
        ref_path = _resolve(config_dir, tts_cfg.clone_ref_wav)
        ref_text_path = _resolve(config_dir, tts_cfg.clone_ref_text) if tts_cfg.clone_ref_text else None
        ref_text = ref_text_path.read_text().strip() if ref_text_path and ref_text_path.is_file() else ""

    f5 = F5CloneEngine(
        ref_audio_path=ref_path,
        ref_text=ref_text,
        model_name=tts_cfg.f5_model,
        steps=tts_cfg.f5_steps,
        speed=tts_cfg.f5_speed,
        quantization_bits=tts_cfg.f5_quantization_bits,
    )

    if backend == "f5":
        return f5
    if backend == "hybrid":
        return HybridTtsEngine(primary=f5, fallback=piper)
    raise ValueError(f"Unknown tts.backend: {tts_cfg.backend}")


def _resolve(config_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else config_dir / path


def piper_from_config(tts_cfg: TtsConfig, voice_path: str | Path, sample_rate: int) -> PiperTtsEngine:
    return PiperTtsEngine(
        tts_cfg.piper_bin,
        str(voice_path),
        sample_rate=sample_rate,
        length_scale=tts_cfg.length_scale,
        sentence_silence=tts_cfg.sentence_silence,
        noise_scale=tts_cfg.noise_scale,
        noise_w_scale=tts_cfg.noise_w_scale,
    )
