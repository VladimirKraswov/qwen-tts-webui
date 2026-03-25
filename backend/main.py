import asyncio
import json
import logging
import shutil
import uuid
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import aiofiles
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import (
    APP_NAME,
    APP_VERSION,
    BOOK_MAX_FILE_SIZE_MB,
    BOOK_MAX_PARAGRAPHS,
    BOOK_PREVIEW_SEGMENTS,
    BOOK_UPLOAD_DIR,
    DEFAULT_LANGUAGE,
    DEFAULT_VOICE,
    FRONTEND_DIR,
    MAX_TTS_TEXT_LENGTH,
    SAMPLE_RATE,
    SUPPORTED_LANGUAGES,
    VOICES,
)
from .tts_engine import engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TTS_TEXT_LENGTH)
    voice: str = DEFAULT_VOICE
    language: str = DEFAULT_LANGUAGE
    response_format: str = Field(default="mp3", pattern="^(mp3|wav|pcm)$")
    stream: bool = False
    instruct: str | None = Field(default=None, max_length=250)


class BookGenerationRequest(BaseModel):
    voice: str = DEFAULT_VOICE
    language: str = DEFAULT_LANGUAGE
    response_format: str = Field(default="mp3", pattern="^(mp3|wav)$")
    instruct: str | None = Field(default=None, max_length=250)


book_tasks: dict[str, asyncio.Task] = {}


def _split_paragraphs(text: str) -> list[str]:
    return [item.strip() for item in text.split("\n\n") if item.strip()]


async def _warm_model() -> None:
    try:
        await asyncio.to_thread(engine.load_model)
    except Exception:
        logger.warning("Model warmup failed during startup; the API will retry on demand.", exc_info=True)


@asynccontextmanager
async def lifespan(_: FastAPI):
    warmup_task = asyncio.create_task(_warm_model())
    try:
        yield
    finally:
        warmup_task.cancel()
        with suppress(asyncio.CancelledError):
            await warmup_task


app = FastAPI(title=APP_NAME, version=APP_VERSION, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


def _job_dir(job_id: str) -> Path:
    return BOOK_UPLOAD_DIR / job_id


def _job_meta_path(job_id: str) -> Path:
    return _job_dir(job_id) / "job.json"


def _job_output_dir(job_id: str) -> Path:
    return _job_dir(job_id) / "audio"


def _job_zip_path(job_id: str) -> Path:
    return _job_dir(job_id) / "audio_bundle.zip"


def _write_job_meta(job_id: str, payload: dict) -> None:
    _job_dir(job_id).mkdir(parents=True, exist_ok=True)
    _job_meta_path(job_id).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _read_job_meta(job_id: str) -> dict:
    path = _job_meta_path(job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Задача не найдена")
    return json.loads(path.read_text(encoding="utf-8"))


def _serialize_job(job_id: str) -> dict:
    meta = _read_job_meta(job_id)
    output_dir = _job_output_dir(job_id)
    zip_path = _job_zip_path(job_id)

    meta["download_url"] = f"/api/books/{job_id}/download" if zip_path.exists() else None
    meta["files"] = sorted(item.name for item in output_dir.glob("*")) if output_dir.exists() else []
    return meta


def _validate_voice_and_language(voice: str, language: str) -> None:
    if voice not in VOICES:
        raise HTTPException(status_code=400, detail="Некорректный голос")
    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail="Некорректный язык")


def _attach_task_cleanup(job_id: str, task: asyncio.Task) -> None:
    def _cleanup(_: asyncio.Task) -> None:
        current = book_tasks.get(job_id)
        if current is task:
            book_tasks.pop(job_id, None)

    task.add_done_callback(_cleanup)


@app.get("/")
async def root():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path, media_type="text/html; charset=utf-8")
    return JSONResponse({"message": f"{APP_NAME} backend is running"})


@app.get("/api/health")
@app.get("/health")
async def health() -> dict:
    return {"ok": True, "engine": engine.status()}


@app.get("/api/voices")
@app.get("/voices")
async def get_voices() -> dict:
    return {
        "voices": VOICES,
        "default_voice": DEFAULT_VOICE,
        "languages": SUPPORTED_LANGUAGES,
        "default_language": DEFAULT_LANGUAGE,
    }


@app.post("/api/audio/speech")
@app.post("/v1/audio/speech")
async def generate_speech(request: TTSRequest):
    _validate_voice_and_language(request.voice, request.language)

    if request.response_format not in {"mp3", "wav"}:
        raise HTTPException(status_code=400, detail="Поддерживаются только mp3 и wav")

    try:
        audio_bytes = await engine.generate_audio(
            text=request.text,
            voice=request.voice,
            language=request.language,
            response_format=request.response_format,
            instruct=request.instruct,
        )
    except Exception as exc:  # pragma: no cover
        logger.exception("Speech generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    media_type = "audio/mpeg" if request.response_format == "mp3" else "audio/wav"
    filename = f"speech.{request.response_format}"

    return StreamingResponse(
        iter([audio_bytes]),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/audio/stream")
@app.post("/v1/audio/stream")
async def stream_speech(request: TTSRequest):
    _validate_voice_and_language(request.voice, request.language)

    if not request.stream:
        return await generate_speech(request)

    async def generator():
        try:
            async for chunk in engine.stream_generate(
                text=request.text,
                voice=request.voice,
                language=request.language,
                instruct=request.instruct,
            ):
                yield chunk
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover
            logger.exception("Stream generation failed")
            return

    return StreamingResponse(
        generator(),
        media_type="audio/L16",
        headers={
            "Content-Type": "audio/L16",
            "X-Sample-Rate": str(SAMPLE_RATE),
            "X-Channels": "1",
        },
    )


@app.post("/api/books/upload")
@app.post("/upload_text")
async def upload_book_text(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="Поддерживаются только .txt файлы")

    content = await file.read()
    max_bytes = BOOK_MAX_FILE_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"Файл слишком большой. Лимит: {BOOK_MAX_FILE_SIZE_MB} MB",
        )

    text = content.decode("utf-8-sig", errors="replace")
    paragraphs = _split_paragraphs(text)

    if not paragraphs:
        raise HTTPException(status_code=400, detail="Файл не содержит текста")

    if len(paragraphs) > BOOK_MAX_PARAGRAPHS:
        raise HTTPException(
            status_code=400,
            detail=f"Слишком много абзацев. Лимит: {BOOK_MAX_PARAGRAPHS}",
        )

    job_id = str(uuid.uuid4())
    job_dir = _job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)

    async with aiofiles.open(job_dir / "original.txt", "wb") as out:
        await out.write(content)

    preview = [
        {"id": index, "text": paragraph[:500]}
        for index, paragraph in enumerate(paragraphs[:BOOK_PREVIEW_SEGMENTS])
    ]

    meta = {
        "job_id": job_id,
        "filename": file.filename,
        "status": "uploaded",
        "total": len(paragraphs),
        "processed": 0,
        "voice": DEFAULT_VOICE,
        "language": DEFAULT_LANGUAGE,
        "response_format": "mp3",
        "instruct": None,
        "error": None,
    }
    _write_job_meta(job_id, meta)

    return {**meta, "segments": preview}


async def _generate_book_worker(
    job_id: str,
    voice: str,
    language: str,
    response_format: str,
    instruct: str | None,
) -> None:
    output_dir = _job_output_dir(job_id)
    zip_path = _job_zip_path(job_id)

    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    with suppress(FileNotFoundError):
        zip_path.unlink()

    original_text = (_job_dir(job_id) / "original.txt").read_text(encoding="utf-8-sig")
    paragraphs = _split_paragraphs(original_text)

    meta = _read_job_meta(job_id)
    meta.update(
        {
            "status": "running",
            "processed": 0,
            "voice": voice,
            "language": language,
            "response_format": response_format,
            "instruct": instruct,
            "error": None,
        }
    )
    _write_job_meta(job_id, meta)

    try:
        for index, paragraph in enumerate(paragraphs, start=1):
            audio_bytes = await engine.generate_audio(
                text=paragraph,
                voice=voice,
                language=language,
                response_format=response_format,
                instruct=instruct,
            )
            out_path = output_dir / f"part_{index:04d}.{response_format}"
            async with aiofiles.open(out_path, "wb") as out:
                await out.write(audio_bytes)

            meta["processed"] = index
            _write_job_meta(job_id, meta)

        with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
            for item in sorted(output_dir.glob("*")):
                archive.write(item, arcname=item.name)

        meta["status"] = "completed"
        _write_job_meta(job_id, meta)

    except asyncio.CancelledError:
        meta["status"] = "cancelled"
        meta["error"] = "Задача отменена"
        _write_job_meta(job_id, meta)
        raise

    except Exception as exc:  # pragma: no cover
        logger.exception("Book generation failed")
        meta["status"] = "failed"
        meta["error"] = str(exc)
        _write_job_meta(job_id, meta)


@app.post("/api/books/{job_id}/start")
@app.post("/generate_book/{job_id}")
async def start_book_generation(job_id: str, request: BookGenerationRequest | None = None):
    if not _job_dir(job_id).exists():
        raise HTTPException(status_code=404, detail="Задача не найдена")

    payload = request or BookGenerationRequest()
    _validate_voice_and_language(payload.voice, payload.language)

    existing_task = book_tasks.get(job_id)
    if existing_task and not existing_task.done():
        return _serialize_job(job_id)

    task = asyncio.create_task(
        _generate_book_worker(
            job_id=job_id,
            voice=payload.voice,
            language=payload.language,
            response_format=payload.response_format,
            instruct=payload.instruct,
        )
    )
    book_tasks[job_id] = task
    _attach_task_cleanup(job_id, task)

    meta = _read_job_meta(job_id)
    meta.update(
        {
            "status": "queued",
            "voice": payload.voice,
            "language": payload.language,
            "response_format": payload.response_format,
            "instruct": payload.instruct,
            "error": None,
        }
    )
    _write_job_meta(job_id, meta)

    return _serialize_job(job_id)


@app.get("/api/books/{job_id}")
async def get_book_job(job_id: str):
    return _serialize_job(job_id)


@app.get("/api/books/{job_id}/download")
async def download_book_bundle(job_id: str):
    zip_path = _job_zip_path(job_id)
    if not zip_path.exists():
        raise HTTPException(status_code=404, detail="Архив ещё не готов")
    return FileResponse(zip_path, media_type="application/zip", filename=f"book-{job_id}.zip")


@app.delete("/api/books/{job_id}")
async def delete_book_job(job_id: str):
    job_dir = _job_dir(job_id)
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Задача не найдена")

    task = book_tasks.get(job_id)
    if task and not task.done():
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    shutil.rmtree(job_dir, ignore_errors=True)
    book_tasks.pop(job_id, None)
    return {"ok": True}


@app.post("/train_voice")
async def train_voice(
    voice_name: str = Form(...),
    audio_files: list[UploadFile] = File(...),
    transcriptions: list[str] = Form(...),
):
    raise HTTPException(
        status_code=501,
        detail=(
            "Обучение голоса пока не реализовано в веб-интерфейсе. "
            f"Получено: voice_name={voice_name}, audio_files={len(audio_files)}, transcriptions={len(transcriptions)}"
        ),
    )