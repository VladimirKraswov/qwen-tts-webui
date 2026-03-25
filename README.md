# Qwen TTS Web UI

Обновлённый веб-интерфейс для синтеза речи на базе Qwen3-TTS.

## Что исправлено

- исправлена сборка Docker: `torch` ставится из репозитория PyTorch отдельно, остальные зависимости — из PyPI;
- backend переведён на `lifespan`, добавлены health/status endpoints;
- корректно поддерживаются `mp3` и `wav`, а не только MP3 под видом любого формата;
- `instruct` теперь реально проходит из UI в движок;
- добавлен трекинг задач озвучки книги с прогрессом и скачиванием ZIP;
- frontend полностью обновлён: состояние модели, прогресс, предпросмотр, download actions, более чистый UI/UX.

## Почему у тебя падала сборка

В Dockerfile использовалось:

```bash
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu124
```

Это подменяло основной индекс пакетов, поэтому `fastapi` и другие зависимости искались только в PyTorch wheel index и не находились.

## Запуск

Важно: в твоей команде не было перевода строки между `git checkout -- backend/Dockerfile` и `docker compose ...`, поэтому shell склеил их в одну команду.

Правильно так:

```bash
git checkout -- backend/Dockerfile
docker compose down --volumes --rmi all
docker compose build --no-cache
docker compose up
```

## Структура API

- `GET /api/health`
- `GET /api/voices`
- `POST /api/audio/speech`
- `POST /api/audio/stream`
- `POST /api/books/upload`
- `POST /api/books/{job_id}/start`
- `GET /api/books/{job_id}`
- `GET /api/books/{job_id}/download`

## Замечание по GPU

Для Docker Compose добавлена современная GPU-конфигурация через `deploy.resources.reservations.devices`. Убедись, что на хосте установлен NVIDIA Container Toolkit.
