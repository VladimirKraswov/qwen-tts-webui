.PHONY: first-run base up down logs rebuild-base clean-images

first-run:
	docker compose build tts-base
	docker compose up --build tts

base:
	docker compose build tts-base

up:
	docker compose up --build tts

down:
	docker compose down

logs:
	docker compose logs -f tts

rebuild-base:
	docker compose build --no-cache tts-base
	docker compose build --no-cache tts

clean-images:
	docker image rm -f qwen-tts-webui-tts:latest qwen-tts-webui-tts-base:latest || true