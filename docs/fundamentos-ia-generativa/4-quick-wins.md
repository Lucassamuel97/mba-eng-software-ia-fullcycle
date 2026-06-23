# Resumo da aula: Quick Wins com IA 

## Visão geral

A aula mostra como obter ganhos rápidos de produtividade com IA no desenvolvimento de software, focando em **Context Engineering** (mais do que apenas prompt), organização de documentação de contexto e uso de ferramentas como MCP para reduzir alucinações e melhorar a qualidade das entregas.

## Principais aprendizados

### 1. Ferramentas de desenvolvimento com IA

- IDEs com IA: Cursor, Windsurf, VS Code + GitHub Copilot, JetBrains.
- Agentes CLI: Claude Code, Gemini CLI, Codex CLI.
- Mensagem principal: ferramenta ajuda, mas **processo e contexto** são o diferencial.

### 2. Documentos de contexto são ativos do projeto

- Documentação de contexto deve ser tratada como ativo de longo prazo, assim como testes automatizados.
- Regras e memórias (ex.: arquivos de rules, CLAUDE.md etc.) orientam comportamento do agente.
- Quanto melhor o material de referência do projeto, menor a chance de respostas genéricas ou incorretas.

### 3. Context Engineering vs Prompt Engineering

- **Prompt Engineering**: foco em instruções textuais pontuais.
- **Context Engineering**: inclui prompt + curadoria de documentos + ferramentas + workflow.
- Ideia central: muitas falhas não são do modelo, mas de contexto insuficiente, mal estruturado ou mal selecionado.

### 4. Tipos de documentos essenciais

- Contexto geral do projeto (produto e visão técnica).
- Stack e documentação de libs/frameworks de terceiros.
- Guidelines de desenvolvimento (padrões de código, testes, documentação).
- README com execução e índice de documentação.
- ADRs (Architecture Decision Records) para registrar decisões e limites de arquitetura.
- Plano de ação (visão de alto nível do que será feito).
- Tarefas detalhadas (passo a passo + TODO).
- `state.local.md` para registrar estado atual do projeto e ponto de continuidade.

### 5. Janelas de contexto e consumo de tokens

- Todo modelo tem limite de contexto; ao ultrapassar, conteúdo antigo sai da memória ativa (sliding window).
- Isso aumenta risco de inconsistência/alucinação em conversas longas.
- Recomendação prática: separar por tarefa/subtarefa em chats distintos.
- Trade-off: mais documentos = mais tokens consumidos, mas potencialmente mais precisão.
- Mitigação: instruir o agente a carregar apenas documentos relevantes para a tarefa atual.

### 6. Context 7 e MCP Server

- Context7 centraliza documentação técnica e expõe acesso via MCP.
- Estratégia prática: baixar/gerar docs de libs usadas no projeto para consulta local da IA.
- Exemplo da aula: usar MCP no Claude Code para mapear dependências e criar markdowns de referência em pasta de docs.

### 7. Workflow com PRPs

- PRP (Product Requirement Prompt) combina requisito + contexto curado + runbook do agente.
- Objetivo: aumentar chance de gerar código mais pronto para produção já na primeira tentativa.
- Base citada: projeto `PRPs-agentic-eng`.

## Quick wins aplicáveis imediatamente

1. Criar uma pasta de documentação de contexto no projeto (`/docs` ou similar).
2. Definir um arquivo de regras/memória do agente com padrões do time.
3. Registrar ADRs para decisões importantes e evitar mudanças fora de escopo.
4. Manter um arquivo de estado atual (`state.local.md`) para retomada rápida entre sessões.
5. Quebrar demandas grandes em subtarefas com chats separados.
6. Usar MCP (ex.: Context7 e Firecrawl) para enriquecer contexto com documentação confiável.

## Síntese final

O maior ganho não vem apenas de "pedir melhor" para a IA, mas de **projetar melhor o contexto**: documentos certos, regras claras, decisões registradas e workflow disciplinado. Com isso, a IA deixa de ser só assistente de texto e passa a atuar como parceira mais consistente de engenharia.

## Referências citadas na aula

- https://github.com/Wirasm/PRPs-agentic-eng
- https://context7.com/
- https://www.firecrawl.dev/
- https://blog.langchain.com/the-rise-of-context-engineering/
- https://www.anthropic.com/news/model-context-protocol
- https://devfullcycle.notion.site/Quick-Wins-com-IA-29-07-2025-2411423c0388801d8369f34cfe403e6c
