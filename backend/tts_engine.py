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
        self.model = None
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

    def load_model(self) -> None:
        if self.model is not None:
            return

        with self._model_lock:
            if self.model is not None:
                return
            try:
                from qwen_tts import Qwen3TTSModel

                dtype = torch.bfloat16 if DTYPE == "bfloat16" else torch.float16
                attn_impl = "flash_attention_2" if torch.cuda.is_available() else "eager"

                self.model = Qwen3TTSModel.from_pretrained(
                    MODEL_NAME,
                    device_map=DEVICE,
                    dtype=dtype,
                    attn_implementation=attn_impl,
                )
                self._last_error = None
                logger.info("Qwen TTS model loaded")
            except Exception as exc:  # pragma: no cover - depends on runtime package/model
                self._last_error = str(exc)
                logger.exception("Failed to load TTS model")
                raise

    def _normalize_pcm(self, pcm: Any) -> np.ndarray:
        arr = np.asarray(pcm)
        if arr.dtype == np.int16:
            return arr
        if np.issubdtype(arr.dtype, np.floating):
            clipped = np.clip(arr, -1.0, 1.0)
            return (clipped * 32767).astype(np.int16)
        return arr.astype(np.int16)

    def _build_generate_kwargs(self, method_name: str, text: str, voice: str, language: str, instruct: str | None) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("TTS model is not loaded")

        method = getattr(self.model, method_name)
        params = inspect.signature(method).parameters
        kwargs: dict[str, Any] = {}

        if "text" in params:
            kwargs["text"] = text
        if "speaker" in params:
            kwargs["speaker"] = voice
        elif "voice" in params:
            kwargs["voice"] = voice
        if "language" in params:
            kwargs["language"] = language
        if "return_numpy" in params:
            kwargs["return_numpy"] = True

        if instruct:
            for alias in ("instruction", "instruct", "style_prompt", "prompt"):
                if alias in params:
                    kwargs[alias] = instruct
                    break

        if method_name == "stream_generate_pcm":
            if "emit_every_frames" in params:
                kwargs["emit_every_frames"] = STREAM_EMIT_EVERY_FRAMES
            if "decode_window_frames" in params:
                kwargs["decode_window_frames"] = STREAM_DECODE_WINDOW_FRAMES

        return kwargs

    def _pcm_to_mp3_bytes(self, pcm: np.ndarray) -> bytes:
        audio = AudioSegment(
            data=pcm.tobytes(),
            sample_width=2,
            frame_rate=SAMPLE_RATE,
            channels=1,
        )
        buffer = io.BytesIO()
        audio.export(buffer, format="mp3", bitrate="192k")
        return buffer.getvalue()

    def _pcm_to_wav_bytes(self, pcm: np.ndarray) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(pcm.tobytes())
        return buffer.getvalue()

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
            kwargs = self._build_generate_kwargs("generate_pcm", text, voice, language, instruct)
            pcm = await asyncio.to_thread(self.model.generate_pcm, **kwargs)
            normalized = self._normalize_pcm(pcm)

            if response_format == "wav":
                return self._pcm_to_wav_bytes(normalized)
            if response_format == "mp3":
                return self._pcm_to_mp3_bytes(normalized)
            return normalized.tobytes()

    async def stream_generate(self, text: str, voice: str, language: str, instruct: str | None = None):
        if self.model is None:
            await asyncio.to_thread(self.load_model)

        async with self._generation_lock:
            if hasattr(self.model, "stream_generate_pcm"):
                kwargs = self._build_generate_kwargs("stream_generate_pcm", text, voice, language, instruct)
                for chunk in self.model.stream_generate_pcm(**kwargs):
                    normalized = self._normalize_pcm(chunk)
                    yield normalized.tobytes()
                    await asyncio.sleep(0)
                return

            pcm = await self.generate_audio(
                text=text,
                voice=voice,
                language=language,
                response_format="pcm",
                instruct=instruct,
            )
            chunk_size = SAMPLE_RATE // 4 * 2
            for index in range(0, len(pcm), chunk_size):
                yield pcm[index : index + chunk_size]
                await asyncio.sleep(0)


engine = TTSEngine()
