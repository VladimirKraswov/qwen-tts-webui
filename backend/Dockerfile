FROM qwen-tts-webui-tts-base:latest

WORKDIR /app
ENV PATH="/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

COPY backend/requirements.app.txt /tmp/requirements.app.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.app.txt

COPY backend /app/backend
COPY frontend /app/frontend

EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]