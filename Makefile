DOCKER_COMPOSE ?= docker compose
SERVICE ?= app
PY ?= python

.PHONY: build up down run-file run-hello-world run-init-chat-model run-prompt-template \
	run-chat-prompt-template run-iniciando-com-chains run-chains-com-decorators \
	run-runnable-lambda run-pipeline-de-processamento run-sumarizacao \
	run-sumarizacao-com-map-reduce run-pipeline-de-sumarizacao \
	run-agente-react-e-tools run-agente-react-usando-prompt-hub \
	run-armazenamento-de-historico run-historico-baseado-em-sliding-window \
	run-carregamento-usando-webbaseloader run-carregamento-de-pdf \
	run-ingestion-pgvector run-search-vector run-role-prompting

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

run-init-chat-model: build
	$(DOCKER_COMPOSE) run --rm $(SERVICE) $(PY) langchain/exemplos/1-fundamentos/2-init-chat-model.py

run-prompt-template: build
	$(DOCKER_COMPOSE) run --rm $(SERVICE) $(PY) langchain/exemplos/1-fundamentos/3-prompt-template.py

run-chat-prompt-template: build
	$(DOCKER_COMPOSE) run --rm $(SERVICE) $(PY) langchain/exemplos/1-fundamentos/4-chat-prompt-template.py

run-iniciando-com-chains: build
	$(DOCKER_COMPOSE) run --rm $(SERVICE) $(PY) langchain/exemplos/2-chains-e-processamento/1-iniciando-com-chains.py

run-chains-com-decorators: build
	$(DOCKER_COMPOSE) run --rm $(SERVICE) $(PY) langchain/exemplos/2-chains-e-processamento/2-chains-com-decorators.py

run-runnable-lambda: build
	$(DOCKER_COMPOSE) run --rm $(SERVICE) $(PY) langchain/exemplos/2-chains-e-processamento/3-runnable-lambda.py

run-pipeline-de-processamento: build
	$(DOCKER_COMPOSE) run --rm $(SERVICE) $(PY) langchain/exemplos/2-chains-e-processamento/4-pipeline-de-processamento.py

run-sumarizacao: build
	$(DOCKER_COMPOSE) run --rm $(SERVICE) $(PY) langchain/exemplos/2-chains-e-processamento/5-sumarizacao.py

run-sumarizacao-com-map-reduce: build
	$(DOCKER_COMPOSE) run --rm $(SERVICE) $(PY) langchain/exemplos/2-chains-e-processamento/6-sumarizacao-com-map-reduce.py

run-pipeline-de-sumarizacao: build
	$(DOCKER_COMPOSE) run --rm $(SERVICE) $(PY) langchain/exemplos/2-chains-e-processamento/7-pipeline-de-sumarizacao.py

run-agente-react-e-tools: build
	$(DOCKER_COMPOSE) run --rm $(SERVICE) $(PY) langchain/exemplos/3-agentes-e-tools/1-agente-react-e-tools.py

run-agente-react-usando-prompt-hub: build
	$(DOCKER_COMPOSE) run --rm $(SERVICE) $(PY) langchain/exemplos/3-agentes-e-tools/2-agente-react-usando-prompt-hub.py

run-armazenamento-de-historico: build
	$(DOCKER_COMPOSE) run --rm $(SERVICE) $(PY) langchain/exemplos/4-gerenciamento-de-memoria/1-armazenamento-de-historico.py

run-historico-baseado-em-sliding-window: build
	$(DOCKER_COMPOSE) run --rm $(SERVICE) $(PY) langchain/exemplos/4-gerenciamento-de-memoria/2-historico-baseado-em-sliding-window.py

run-carregamento-usando-webbaseloader: build
	$(DOCKER_COMPOSE) run --rm $(SERVICE) $(PY) langchain/exemplos/5-loaders-e-banco-de-dados-vetoriais/1-carregamento-usando-WebBaseLoader.py

run-carregamento-de-pdf: build
	$(DOCKER_COMPOSE) run --rm $(SERVICE) $(PY) langchain/exemplos/5-loaders-e-banco-de-dados-vetoriais/2-carregamento-de-pdf.py

run-ingestion-pgvector: build
	$(DOCKER_COMPOSE) run --rm $(SERVICE) $(PY) langchain/exemplos/5-loaders-e-banco-de-dados-vetoriais/3-ingestion-pgvector.py

run-search-vector: build
	$(DOCKER_COMPOSE) run --rm $(SERVICE) $(PY) langchain/exemplos/5-loaders-e-banco-de-dados-vetoriais/4-search-vector.py

run-role-prompting: build
	$(DOCKER_COMPOSE) run --rm $(SERVICE) $(PY) prompt-engineering/1-tipos-de-prompts/0-Role-prompting.py

run-zero-shot: build
	$(DOCKER_COMPOSE) run --rm $(SERVICE) $(PY) prompt-engineering/1-tipos-de-prompts/1-zero-shot.py

run-one-few-shot: build
	$(DOCKER_COMPOSE) run --rm $(SERVICE) $(PY) prompt-engineering/1-tipos-de-prompts/2-one-few-shot.py


run: build
	@FILE=$$(whiptail \
		--title "LangChain Exemplos" \
		--menu "Escolha um exemplo para executar:" \
		25 100 15 \
		"1" "1-fundamentos/1-hello-world.py" \
		"2" "1-fundamentos/2-init-chat-model.py" \
		"3" "1-fundamentos/3-prompt-template.py" \
		"4" "1-fundamentos/4-chat-prompt-template.py" \
		"5" "2-chains-e-processamento/1-iniciando-com-chains.py" \
		"6" "2-chains-e-processamento/2-chains-com-decorators.py" \
		"7" "2-chains-e-processamento/3-runnable-lambda.py" \
		"8" "2-chains-e-processamento/4-pipeline-de-processamento.py" \
		"9" "2-chains-e-processamento/5-sumarizacao.py" \
		"10" "2-chains-e-processamento/6-sumarizacao-com-map-reduce.py" \
		"11" "2-chains-e-processamento/7-pipeline-de-sumarizacao.py" \
		"12" "3-agentes-e-tools/1-agente-react-e-tools.py" \
		"13" "3-agentes-e-tools/2-agente-react-usando-prompt-hub.py" \
		"14" "4-gerenciamento-de-memoria/1-armazenamento-de-historico.py" \
		"15" "4-gerenciamento-de-memoria/2-historico-baseado-em-sliding-window.py" \
		"16" "5-loaders-e-banco-de-dados-vetoriais/1-carregamento-usando-WebBaseLoader.py" \
		"17" "5-loaders-e-banco-de-dados-vetoriais/2-carregamento-de-pdf.py" \
		"18" "5-loaders-e-banco-de-dados-vetoriais/3-ingestion-pgvector.py" \
		"19" "5-loaders-e-banco-de-dados-vetoriais/4-search-vector.py" \
		"20" "prompt-engineering/1-tipos-de-prompts/0-Role-prompting.py" \
		"21" "prompt-engineering/1-tipos-de-prompts/1-zero-shot.py" \
		"22" "prompt-engineering/1-tipos-de-prompts/2-one-few-shot.py" \
		"23" "prompt-engineering/1-tipos-de-prompts/3-CoT.py" \
		3>&1 1>&2 2>&3); \
	STATUS=$$?; \
	if [ $$STATUS -eq 0 ]; then \
		case $$FILE in \
			1) FILE_PATH="langchain/exemplos/1-fundamentos/1-hello-world.py" ;; \
			2) FILE_PATH="langchain/exemplos/1-fundamentos/2-init-chat-model.py" ;; \
			3) FILE_PATH="langchain/exemplos/1-fundamentos/3-prompt-template.py" ;; \
			4) FILE_PATH="langchain/exemplos/1-fundamentos/4-chat-prompt-template.py" ;; \
			5) FILE_PATH="langchain/exemplos/2-chains-e-processamento/1-iniciando-com-chains.py" ;; \
			6) FILE_PATH="langchain/exemplos/2-chains-e-processamento/2-chains-com-decorators.py" ;; \
			7) FILE_PATH="langchain/exemplos/2-chains-e-processamento/3-runnable-lambda.py" ;; \
			8) FILE_PATH="langchain/exemplos/2-chains-e-processamento/4-pipeline-de-processamento.py" ;; \
			9) FILE_PATH="langchain/exemplos/2-chains-e-processamento/5-sumarizacao.py" ;; \
			10) FILE_PATH="langchain/exemplos/2-chains-e-processamento/6-sumarizacao-com-map-reduce.py" ;; \
			11) FILE_PATH="langchain/exemplos/2-chains-e-processamento/7-pipeline-de-sumarizacao.py" ;; \
			12) FILE_PATH="langchain/exemplos/3-agentes-e-tools/1-agente-react-e-tools.py" ;; \
			13) FILE_PATH="langchain/exemplos/3-agentes-e-tools/2-agente-react-usando-prompt-hub.py" ;; \
			14) FILE_PATH="langchain/exemplos/4-gerenciamento-de-memoria/1-armazenamento-de-historico.py" ;; \
			15) FILE_PATH="langchain/exemplos/4-gerenciamento-de-memoria/2-historico-baseado-em-sliding-window.py" ;; \
			16) FILE_PATH="langchain/exemplos/5-loaders-e-banco-de-dados-vetoriais/1-carregamento-usando-WebBaseLoader.py" ;; \
			17) FILE_PATH="langchain/exemplos/5-loaders-e-banco-de-dados-vetoriais/2-carregamento-de-pdf.py" ;; \
			18) FILE_PATH="langchain/exemplos/5-loaders-e-banco-de-dados-vetoriais/3-ingestion-pgvector.py" ;; \
			19) FILE_PATH="langchain/exemplos/5-loaders-e-banco-de-dados-vetoriais/4-search-vector.py" ;; \
			20) FILE_PATH="prompt-engineering/1-tipos-de-prompts/0-Role-prompting.py" ;; \
			21) FILE_PATH="prompt-engineering/1-tipos-de-prompts/1-zero-shot.py" ;; \
			22) FILE_PATH="prompt-engineering/1-tipos-de-prompts/2-one-few-shot.py" ;; \
			23) FILE_PATH="prompt-engineering/1-tipos-de-prompts/3-CoT.py" ;; \
		esac; \
		clear; \
		echo "Executando $$FILE_PATH"; \
		$(DOCKER_COMPOSE) run --rm $(SERVICE) $(PY) $$FILE_PATH; \
	else \
		echo "Execução cancelada."; \
	fi