import asyncio
import inspect
import io
import logging
import threading
import wave
from typing import Any

import numpy as np
import torch
from pydub import AudioSegment

from .config import (
    DEVICE,
    DTYPE,
    MODEL_NAME,
    SAMPLE_RATE,
    STREAM_DECODE_WINDOW_FRAMES,
    STREAM_EMIT_EVERY_FRAMES,
)

logger = logging.getLogger(__name__)


class TTSEngine:
    def __init__(self) -> None:
        self.model: Any | None = None
        self._model_lock = threading.Lock()
        self._generation_lock = asyncio.Lock()
        self._last_error: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def status(self) -> dict[str, Any]:
        return {
            "loaded": self.is_loaded,
            "device": DEVICE,
            "dtype": DTYPE,
            "model_name": MODEL_NAME,
            "last_error": self._last_error,
        }

    def _get_model_mode(self) -> str:
        name = MODEL_NAME.lower()
        if "customvoice" in name:
            return "custom_voice"
        if "voicedesign" in name:
            return "voice_design"
        if "base" in name:
            return "voice_clone"
        return "custom_voice"

    def load_model(self) -> None:
        if self.model is not None:
            return

        with self._model_lock:
            if self.model is not None:
                return

            try:
                from qwen_tts import Qwen3TTSModel

                dtype = torch.bfloat16 if DTYPE == "bfloat16" else torch.float16
                attn_impl = "sdpa" if torch.cuda.is_available() else "eager"

                logger.info(
                    "Loading model with dtype=%s, attn_implementation=%s",
                    dtype,
                    attn_impl,
                )

                self.model = Qwen3TTSModel.from_pretrained(
                    MODEL_NAME,
                    device_map=DEVICE,
                    dtype=dtype,
                    attn_implementation=attn_impl,
                )
                self._last_error = None
                logger.info("Qwen TTS model loaded")
            except Exception as exc:
                self._last_error = str(exc)
                logger.exception("Failed to load TTS model")
                raise

    def _normalize_pcm(self, pcm: Any) -> np.ndarray:
        arr = np.asarray(pcm).reshape(-1)

        if arr.dtype == np.int16:
            return arr

        if np.issubdtype(arr.dtype, np.floating):
            clipped = np.clip(arr, -1.0, 1.0)
            return (clipped * 32767).astype(np.int16)

        return arr.astype(np.int16)

    def _first_waveform(self, wavs: Any) -> np.ndarray:
        if isinstance(wavs, (list, tuple)):
            if not wavs:
                raise RuntimeError("Model returned empty waveform list")
            return np.asarray(wavs[0])

        arr = np.asarray(wavs)
        if arr.ndim > 1:
            return np.asarray(arr[0])
        return arr

    def _build_method_kwargs(
        self,
        method_name: str,
        text: str,
        voice: str,
        language: str,
        instruct: str | None,
    ) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("TTS model is not loaded")

        method = getattr(self.model, method_name)
        params = inspect.signature(method).parameters
        kwargs: dict[str, Any] = {}

        if "text" in params:
            kwargs["text"] = text

        if "language" in params:
            kwargs["language"] = language

        if method_name in {"generate_custom_voice", "stream_generate_pcm"}:
            if "speaker" in params:
                kwargs["speaker"] = voice
            elif "voice" in params:
                kwargs["voice"] = voice

            if instruct:
                for alias in ("instruct", "instruction", "style_prompt", "prompt"):
                    if alias in params:
                        kwargs[alias] = instruct
                        break

        elif method_name == "generate_voice_design":
            if instruct:
                for alias in ("instruct", "instruction", "style_prompt", "prompt"):
                    if alias in params:
                        kwargs[alias] = instruct
                        break

        elif method_name == "generate_voice_clone":
            raise NotImplementedError(
                "Текущий UI не поддерживает Base/Voice Clone модели: нужны ref_audio и ref_text."
            )

        if method_name == "stream_generate_pcm":
            if "emit_every_frames" in params:
                kwargs["emit_every_frames"] = STREAM_EMIT_EVERY_FRAMES
            if "decode_window_frames" in params:
                kwargs["decode_window_frames"] = STREAM_DECODE_WINDOW_FRAMES

        return kwargs

    def _get_nonstream_method_name(self) -> str:
        mode = self._get_model_mode()
        if mode == "custom_voice":
            return "generate_custom_voice"
        if mode == "voice_design":
            return "generate_voice_design"
        if mode == "voice_clone":
            return "generate_voice_clone"
        return "generate_custom_voice"

    def _pcm_to_mp3_bytes(self, pcm: np.ndarray, sample_rate: int) -> bytes:
        audio = AudioSegment(
            data=pcm.tobytes(),
            sample_width=2,
            frame_rate=sample_rate,
            channels=1,
        )
        buffer = io.BytesIO()
        audio.export(buffer, format="mp3", bitrate="192k")
        return buffer.getvalue()

    def _pcm_to_wav_bytes(self, pcm: np.ndarray, sample_rate: int) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm.tobytes())
        return buffer.getvalue()

    async def _generate_waveform_locked(
        self,
        text: str,
        voice: str,
        language: str,
        instruct: str | None,
    ) -> tuple[np.ndarray, int]:
        if self.model is None:
            raise RuntimeError("TTS model is not loaded")

        method_name = self._get_nonstream_method_name()
        method = getattr(self.model, method_name)
        kwargs = self._build_method_kwargs(method_name, text, voice, language, instruct)

        result = await asyncio.to_thread(method, **kwargs)

        if not isinstance(result, tuple) or len(result) < 2:
            raise RuntimeError(
                f"Unexpected return value from {method_name}: expected (wavs, sample_rate)"
            )

        wavs, sample_rate = result[0], int(result[1])
        waveform = self._first_waveform(wavs)
        normalized = self._normalize_pcm(waveform)

        if sample_rate != SAMPLE_RATE:
            logger.warning(
                "Model returned sample_rate=%s, expected=%s",
                sample_rate,
                SAMPLE_RATE,
            )

        return normalized, sample_rate

    async def generate_audio(
        self,
        text: str,
        voice: str,
        language: str,
        response_format: str = "mp3",
        instruct: str | None = None,
    ) -> bytes:
        if self.model is None:
            await asyncio.to_thread(self.load_model)

        async with self._generation_lock:
            normalized, sample_rate = await self._generate_waveform_locked(
                text,
                voice,
                language,
                instruct,
            )

        if response_format == "wav":
            return self._pcm_to_wav_bytes(normalized, sample_rate)
        if response_format == "mp3":
            return self._pcm_to_mp3_bytes(normalized, sample_rate)
        return normalized.tobytes()

    async def stream_generate(
        self,
        text: str,
        voice: str,
        language: str,
        instruct: str | None = None,
    ):
        if self.model is None:
            await asyncio.to_thread(self.load_model)

        if self.model is not None and hasattr(self.model, "stream_generate_pcm"):
            async with self._generation_lock:
                kwargs = self._build_method_kwargs(
                    "stream_generate_pcm",
                    text,
                    voice,
                    language,
                    instruct,
                )
                for chunk in self.model.stream_generate_pcm(**kwargs):
                    normalized = self._normalize_pcm(chunk)
                    yield normalized.tobytes()
                    await asyncio.sleep(0)
            return

        async with self._generation_lock:
            normalized, sample_rate = await self._generate_waveform_locked(
                text,
                voice,
                language,
                instruct,
            )

        chunk_samples = sample_rate // 4
        for index in range(0, len(normalized), chunk_samples):
            yield normalized[index:index + chunk_samples].tobytes()
            await asyncio.sleep(0)


engine = TTSEngine()