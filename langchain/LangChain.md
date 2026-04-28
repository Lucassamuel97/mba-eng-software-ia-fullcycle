# Resumo: Visão Geral e Ecossistema LangChain

## 1. Histórico e Evolução (Timeline)
Apesar de ser amplamente conhecida como uma biblioteca open-source, a LangChain possui uma estrutura corporativa e financeira robusta por trás de seu desenvolvimento, o que a torna uma opção confiável para produção (embora não seja uma "bala de prata").

* **Outubro/2022:** Criação da LangChain por Harrison Chase.
* **Abril/2023:** Transformou-se em uma startup, recebendo investimentos de Série A (totalizando ~US$ 35 milhões).
* **Fevereiro/2024:** Lançamento do **LangSmith**.
* **2025:** Recebeu investimento de Série B de US$ 100 milhões, alcançando um *valuation* estimado de **US$ 1.1 Bilhão**.
* **Maio/2025:** Lançamento da **LangGraph Platform**.

---

## 2. O Ecossistema de Serviços LangChain
A LangChain oferece uma suíte de ferramentas (algumas proprietárias/pagas, outras open-source) para cobrir todo o ciclo de vida de aplicações baseadas em LLMs:

### LangSmith
* **O que é:** Plataforma (com tier gratuito e pago) focada em **observabilidade e debugging**.
* **Para que serve:** Monitoramento de aplicações em produção, avaliação de performance dos modelos, e gerenciamento de custos e latência.

### LangServe
* **O que é:** Biblioteca para expor os agentes como serviços.
* **Para que serve:** Criação de **APIs padronizadas** para os agentes de forma simplificada. Possui serviços de *hosting* próprios para deploy com infraestrutura gerenciada e escalável.

### LangGraph & LangGraph Platform
* **LangGraph:** Framework consolidado no mercado para a **orquestração e gerenciamento de agentes autônomos**. Permite a criação de fluxos de decisão determinísticos baseados em grafos.
* **LangGraph Platform:** Infraestrutura que permite rodar *Stateful Agents* (agentes de longa duração com gerenciamento de estado, memória e fluxos de trabalho completos).
* **LangGraph Studio:** Interface web (semelhante a uma IDE) para o gerenciamento visual de projetos LangGraph, podendo ser executada localmente ou na nuvem.

### LangChain Hub
* **O que é:** Um repositório/catálogo central.
* **Para que serve:** Permite **publicar, versionar, testar e baixar prompts** e outros artefatos. Possui integração nativa com o LangSmith e disponibiliza SDKs para interações automatizadas (via *push* ou *pull*).


# Resumo: Principais Recursos e Arquitetura da LangChain

## 1. Objetivo Principal
A LangChain atua como uma poderosa camada de abstração projetada para **simplificar a integração com LLMs** e serviços auxiliares. Ela evita que o desenvolvedor precise lidar com a complexidade das APIs específicas de cada provedor, facilitando a criação de fluxos de dados, armazenamento de contexto e tomada de decisões.

## 2. Modularização dos Pacotes
Com a evolução do framework, a biblioteca principal foi desmembrada para facilitar a manutenção e extensibilidade:
* **`langchain-core`**: Contém as implementações de alto nível e interfaces base, padronizando o ecossistema.
* **`langchain`**: A biblioteca principal que utiliza as implementações do *core*.
* **`langchain-community`**: Destinada a integrações de terceiros.
* **Pacotes Específicos**: Integrações isoladas por provedor (ex: `langchain-openai`).

## 3. Principais Recursos (Core Capabilities)

### Chains e LCEL
* **Chains**: São fluxos de execução (pipelines) compostos por etapas. A saída de um LLM ou processo pode ser diretamente direcionada como entrada para a próxima etapa.
* **LCEL (LangChain Expression Language)**: Introduzido nas versões mais recentes, permite criar esses fluxos de forma muito mais simples e limpa, utilizando o operador de *pipe* (`|`), semelhante ao comportamento em sistemas Unix (ex: `prompt | model | output_parser`).

### Ingestão de Dados (Loaders e Splitters)
* **Document Loaders**: Abstrações de uma linha de comando para carregar diversos tipos de arquivos (PDF, CSV, HTML, JSON, Markdown) ou fazer Web Scraping, sem precisar de bibliotecas nativas complexas para cada formato.
* **Document Splitters**: Ferramentas para dividir documentos grandes em pedaços menores (*chunks*), permitindo que sejam inseridos no limite de contexto (janela de tokens) das LLMs.

### Embeddings e Bancos de Dados Vetoriais
* A LangChain abstrai a geração de **Embeddings** (representações vetoriais matemáticas de textos) usando provedores como OpenAI e Hugging Face.
* Facilita a persistência e **busca semântica** (por similaridade) abstraindo as complexidades de Bancos de Dados Vetoriais consolidados, como Pinecone, PG Vector, Weaviate e FAISS.

### Agentes e Ferramentas (Tools)
* Permite a criação de **Agentes** que tomam decisões autônomas e escolhem quais ferramentas (Tools) acionar para cumprir uma tarefa.
* *Nota:* Para orquestração determinística e fluxos complexos de multiagentes, a abordagem recomendada é migrar para o **LangGraph** (ou alternativas como CrewAI).

### Manipulação de Prompts e Saídas Estruturadas
* **Prompt Templates**: Criação de prompts dinâmicos utilizando variáveis (*placeholders*).
* **Integração com Pydantic**: Capacidade nativa de forçar (fazer o *parse*) a saída da LLM para que ela retorne dados estruturados baseados em modelos/classes fortemente tipados definidos via Pydantic (ex: garantir que a saída seja um objeto com `nome` (string) e `preco` (inteiro)).

### Memória e Otimização
* **Memória**: Gerenciamento automático do histórico de conversas para manter o contexto.
* **Sumarização / MapReduce**: Técnicas integradas para resumir conversas longas ou processar grandes volumes de dados sem estourar os limites de contexto do modelo.



# Resumo: Exemplos Práticos e Aplicações da LangChain
## Fundamentos
* **[1-hello-world.py](exemplos/1-fundamentos/1-hello-world.py)**: Invoca o modelo Gemini diretamente com `ChatGoogleGenerativeAI` e imprime o texto retornado.
* **[2-init-chat-model.py](exemplos/1-fundamentos/2-init-chat-model.py)**: Inicializa um chat model via `init_chat_model`, abstraindo o provedor, e faz uma chamada simples.
* **[3-prompt-template.py](exemplos/1-fundamentos/3-prompt-template.py)**: Cria um `PromptTemplate` com variavel dinamica e faz a formatacao do texto.
* **[4-chat-prompt-template.py](exemplos/1-fundamentos/4-chat-prompt-template.py)**: Monta mensagens `system` e `user` com `ChatPromptTemplate`, inspeciona as mensagens e invoca o modelo.

## Chains e Processamento

* **[1-iniciando-com-chains.py](exemplos/2-chains-e-processamento/1-iniciando-com-chains.py)**: Compoe um pipeline LCEL com `prompt | model` e executa via `invoke`.
* **[2-chains-com-decorators.py](exemplos/2-chains-e-processamento/2-chains-com-decorators.py)**: Usa `@chain` para criar uma etapa customizada (quadrado) e encadeia com prompt e LLM.
* **[3-runnable-lambda.py](exemplos/2-chains-e-processamento/3-runnable-lambda.py)**: Demonstra `RunnableLambda` transformando string em inteiro e executando com `invoke`.
* **[4-pipeline-de-processamento.py](exemplos/2-chains-e-processamento/4-pipeline-de-processamento.py)**: Pipeline de traducao + resumo com `PromptTemplate` e `StrOutputParser`.
* **[5-sumarizacao.py](exemplos/2-chains-e-processamento/5-sumarizacao.py)**: Faz chunking de texto longo e resume o conteudo em uma unica chamada.
* **[6-sumarizacao-com-map-reduce.py](exemplos/2-chains-e-processamento/6-sumarizacao-com-map-reduce.py)**: Implementa sumarizacao em duas fases (map e reduce) com LLM.
* **[7-pipeline-de-sumarizacao.py](exemplos/2-chains-e-processamento/7-pipeline-de-sumarizacao.py)**: Constroi pipeline LCEL completo de map-reduce com `RunnableLambda` e `StrOutputParser`.

## Agentes e Ferramentas (Tools)

* **[1-agente-react-e-tools.py](exemplos/3-agentes-e-tools/1-agente-react-e-tools.py)**: Cria um agente ReAct com tools locais (calculadora e busca mock) e prompt customizado.
* **[2-agente-react-usando-prompt-hub.py](exemplos/3-agentes-e-tools/2-agente-react-usando-prompt-hub.py)**: Usa prompt do LangChain Hub para ReAct e executa agente com tool mock.

## Gerenciamento de Memória e Contexto

* **[1-armazenamento-de-historico.py](exemplos/4-gerenciamento-de-memoria/1-armazenamento-de-historico.py)**: Adiciona memoria de conversa por sessao com `RunnableWithMessageHistory`.
* **[2-historico-baseado-em-sliding-window.py](exemplos/4-gerenciamento-de-memoria/2-historico-baseado-em-sliding-window.py)**: Limita o historico com sliding window usando `trim_messages` antes do prompt.

## Ingestão de Dados, Embeddings e Bancos de Dados Vetoriais

* **[1-carregamento-usando-WebBaseLoader.py](exemplos/5-loaders-e-banco-de-dados-vetoriais/1-carregamento-usando-WebBaseLoader.py)**: Carrega uma pagina web com `WebBaseLoader`, aplica chunking e imprime os trechos.
* **[2-carregamento-de-pdf.py](exemplos/5-loaders-e-banco-de-dados-vetoriais/2-carregamento-de-pdf.py)**: Carrega um PDF com `PyPDFLoader`, aplica chunking com `RecursiveCharacterTextSplitter` e mostra a quantidade de chunks.
* **[3-ingestion-pgvector.py](exemplos/5-loaders-e-banco-de-dados-vetoriais/3-ingestion-pgvector.py)**: Ingestao de chunks do PDF no PGVector com embeddings do Gemini e ids estaveis.
* **[4-search-vector.py](exemplos/5-loaders-e-banco-de-dados-vetoriais/4-search-vector.py)**: Faz busca por similaridade no PGVector e imprime texto e metadados dos resultados.
