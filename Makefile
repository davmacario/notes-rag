.PHONY: up down restart logs ps build help

help:
	@echo "Targets:"
	@echo "  up       Build (if needed) and start the notes-rag container in the background"
	@echo "  down     Stop and remove the notes-rag container"
	@echo "  restart  Recreate the notes-rag container"
	@echo "  logs     Follow the notes-rag container logs (Ctrl-C to detach)"
	@echo "  ps       Show the status of the notes-rag container"
	@echo "  build    Build the notes-rag image from the local Dockerfile"
	@echo
	@echo "Configuration is read from docker-compose.yaml and .env (see .test.env for the expected keys)."

up:
	docker compose up --detach

down:
	docker compose down

restart:
	docker compose up --detach --force-recreate

logs:
	docker compose logs --follow

ps:
	docker compose ps

build:
	docker compose build
