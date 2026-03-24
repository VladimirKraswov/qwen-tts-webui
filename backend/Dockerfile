FROM nvidia/cuda:12.1.1-base-ubuntu22.04

ENV PYTHONUNBUFFERED=1     PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends     python3 python3-pip python3-dev ffmpeg git curl     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip setuptools wheel &&     python3 -m pip install -r /app/requirements.txt

COPY . /app

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
