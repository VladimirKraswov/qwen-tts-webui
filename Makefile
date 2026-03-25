.PHONY: base up down logs rebuild-base clean-images

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
	docker image rm -f qwen-tts-webui:dev qwen-tts-webui-ai-base:cu124 || true
