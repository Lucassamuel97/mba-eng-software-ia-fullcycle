DOCKER_COMPOSE ?= docker compose
SERVICE ?= app
PY ?= python

.PHONY: build up down run-file run-hello-world run-role-prompting

build:
	$(DOCKER_COMPOSE) build $(SERVICE)

up:
	$(DOCKER_COMPOSE) up -d postgres

down:
	$(DOCKER_COMPOSE) down

run-file: build
	$(DOCKER_COMPOSE) run --rm $(SERVICE) $(PY) $(FILE)

run-hello-world: build
	$(DOCKER_COMPOSE) run --rm $(SERVICE) $(PY) langchain/exemplos/1-fundamentos/1-hello-world.py

run-role-prompting: build
	$(DOCKER_COMPOSE) run --rm $(SERVICE) $(PY) promp-engineering/1-tipos-de-prompts/0-Role-prompting.py
