# Módulo Prompt Engineering

## Sumário

- [Aula 1: Introdução e overview da disciplina](#aula-1-introdução-e-overview-da-disciplina)

- [Aula 2: O que realmente é Prompt Engineering](#aula-2-o-que-realmente-é-prompt-engineering)

- [Aula 3: Outras Aplicabilidades do Prompt Engineering](#aula-3-outras-aplicabilidades-do-prompt-engineering)

- [Aula 4: Técnicas e Tipos de Prompt](#aula-4-técnicas-e-tipos-de-prompt)

- [Aula 5: Role Prompting (Prompt de Persona)](#aula-5-role-prompting-prompt-de-persona)

- [Aula 6: Exemplo Prático de Role Prompting (LangChain)](#aula-6-exemplo-prático-de-role-prompting-langchain)

- [Aula 7: Zero-shot](#aula-7-zero-shot)

- [Aula 8: Few-shot](#aula-8-few-shot)

- [Aula 9: Chain of Thought](#aula-9-chain-of-thought)

- [Aula 10: Chain of Thought com Self-consistency](#aula-10-chain-of-thought-com-self-consistency)

- [Aula 11: Tree of Thought](#aula-11-tree-of-thought)

- [Aula 12: Skeleton of Thought](#aula-12-skeleton-of-thought)

### Conceitos Importantes

- [Revisitando conceitos](#revisitando-conceitos)

- [Context Window para Prompt Engineering](#context-window-para-prompt-engineering)

- [Context Window vs memória, custo e latência](#context-window-vs-memória-custo-e-latência)

- [Janela de contexto vs parâmetros](#janela-de-contexto-vs-parâmetros)

- [Truncamento](#truncamento)

- [Sumarização](#sumarização)

- [Sliding window](#sliding-window)

- [Prompt Caching](#prompt-caching)

- [Batch Prompting](#batch-prompting)

- [Demonstração prática para batch prompting](#demonstração-prática-para-batch-prompting)

## Aula 1: Introdução e overview da disciplina

Esta aula inaugural estabelece as bases de funcionamento do MBA, destacando a flexibilidade da ementa. O objetivo é permitir que os encontros ao vivo acompanhem as constantes inovações da área, como *prompt engineering* e novos workflows, sem ficar preso estritamente ao cronograma das aulas gravadas.

A dinâmica dos encontros ao vivo, com duração aproximada de 1h30, foca na aplicação prática e na interação. O professor utiliza uma analogia com séries de TV para explicar que as aulas são abrangentes, permitindo que o aluno acompanhe o conteúdo mesmo que não tenha assistido a todos os episódios anteriores, embora temas complexos exijam o estudo dos pré-requisitos disponíveis na plataforma.


## Aula 2: O que realmente é Prompt Engineering

Esta aula aborda a transição do *Prompt Engineering* de um termo de "hype" para uma competência técnica essencial no dia a dia do desenvolvedor.

---

### 1. Definição e Conceito
* **Não é apenas uma palavra da moda:** É a disciplina de estruturar instruções de forma clara e muitas vezes criativa para obter resultados específicos e de alta qualidade.
* **O "Salto" de Qualidade:** A diferença entre um prompt ruim e um bem estruturado é comparada à diferença entre interagir com um "adolescente" ou com um "pós-doutor".
* **Mecanismo de Resposta:** A capacidade de um modelo de IA compreender instruções complexas e técnicas depende diretamente da construção do prompt.

### 2. Contexto de Software (Domínio Específico)
O *Prompt Engineering* deve ser aplicado dentro do domínio de especialidade (Engenharia de Software) para que a linguagem e a complexidade sejam adequadas. 

**Onde ele impacta o fluxo de trabalho:**
* **Produtividade:** Atua como um catalisador em praticamente todas as etapas do ciclo de vida do software.
* **Desenvolvimento e Manutenção:** Escrita de código novo, refatoração e manutenção de sistemas legados.
* **Documentação Técnica:** Criação de *Design Docs* e documentações de sistemas.
* **Qualidade e Governança:** Condução de *Code Reviews* e auxílio no brainstorming de arquitetura.
* **Automação de Rotina:** Geração de mensagens de commit, automação de tarefas repetitivas e orientação de agentes de IA especializados (agentes de codificação).

### 3. Conclusão da Aula
O foco central é entender que o Prompt Engineering não é uma tarefa isolada, mas um recurso que garante a confiabilidade do software, desde a sua concepção inicial até a entrega em produção, aumentando a precisão das ferramentas de IA generativa.

## Aula 3: Outras Aplicabilidades do Prompt Engineering

Esta aula explora a dualidade do uso de prompts: como ferramenta de produtividade para o desenvolvedor e como componente estrutural em aplicações que integram IA.

---

### 1. IA como Ferramenta vs. IA como Produto
Existem dois cenários distintos para o uso de prompts:
* **Apoio ao Desenvolvimento:** Uso de prompts extensos e contextos ricos (muitos documentos) para gerar código e ganhar produtividade.
* **Integração em Aplicações:** Desenvolvimento de agentes ou sistemas multiagentes. Aqui, o foco muda para **prompts enxutos** visando reduzir custos de tokens e latência das requisições.

### 2. Definição de Escopo e Proatividade
O prompt é o que "doma" o agente de IA, definindo:
* **Comportamento e Transações:** Regras para tarefas críticas (ex: cálculo de juros baseado em histórico).
* **Nível de Autonomia:** Define se a IA deve ser proativa (tomar ações) ou apenas esclarecer dúvidas, evitando execuções indesejadas.
* **Acesso a Dados Externos:** Como a IA deve interagir com bancos de dados, APIs ou conectores (HubSpot, CRM, etc.).

### 3. Segurança e Modelos Especializados
* **Segurança e Privacidade:** Prompts mal estruturados podem permitir vazamento de dados sensíveis ou exploração de vulnerabilidades no sistema.
* **Modelos Menores (Small LLMs):** Em modelos com poucos parâmetros, a sensibilidade é maior; uma pequena alteração no prompt pode alterar drasticamente o comportamento do agente.

### 4. A Mudança de Paradigma: Programação Probabilística
* **A Nova Linguagem:** O Prompt Engineering é apresentado como a nova "linguagem de programação".
* **Determinístico vs. Probabilístico:** No desenvolvimento tradicional, as regras são definidas por `if/else` (determinístico). Na IA, o fluxo de decisão é baseado em probabilidades. O prompt substitui as regras rígidas de código para orientar o comportamento do sistema.

### 5. Ecossistema e Uso como Usuário Final
Aplicações para produtividade diária em plataformas como ChatGPT, Gemini e Claude:
* **Conectores e Ferramentas:** Uso de protocolos como MCP para ler planilhas, acessar calendários, Google Drive e gerar apresentações.
* **Intencionalidade:** Dominar as técnicas de prompt permite reduzir tarefas que levariam dias para poucos minutos, tornando o uso da IA mais assertivo e menos acidental.

## Aula 4: Técnicas e Tipos de Prompt

Esta seção da aula introduz a necessidade de entender que o *Prompt Engineering* é uma disciplina baseada em estudos científicos e benchmarks, e não apenas em "tentativa e erro" intuitiva.

---

### 1. O Mito do Detalhamento Excessivo
* **Qualidade vs. Quantidade:** Escrever um prompt não significa fornecer o máximo de detalhes possível. 
* **Ruído de Informação:** Fornecer mais dados do que o necessário pode confundir o modelo e comprometer a qualidade da resposta final.

### 2. Fundamentação Teórica e Benchmarks
* **Base Científica:** As técnicas de prompt (como as que serão vistas a seguir) são baseadas em *papers*, estatísticas e estudos de comportamento de modelos.
* **Impacto nos Resultados:** Os benchmarks dos modelos mostram claramente como diferentes formatos de instrução alteram a eficácia da IA em tarefas específicas.

### 3. Metodologia de Aprendizado
* **Foco na Prática:** O aprendizado não será apenas teórico. Envolverá:
    * Análise de código e execução de prompts em tempo real.
    * Uso de ferramentas como ChatGPT e ambientes de desenvolvimento.
    * Entendimento do que acontece "por baixo do capô" dos modelos.
* **Objetivo:** Visualizar como as variações nos prompts influenciam diretamente o comportamento e a geração de código.

### 4. Próximos Passos
* A introdução às técnicas específicas começará pelo **Role Prompting** (Prompt de Persona), uma das técnicas mais comuns, mas que exige entendimento técnico para ser usada com máxima eficiência.

## Aula 5: Role Prompting (Prompt de Persona)

Esta aula explora a técnica de definir um papel ou função específica para o modelo, analisando quando essa estratégia é realmente eficaz e quais são suas limitações técnicas.

---

### 1. O que é Role Prompting?
* **Definição:** É a atribuição explícita de um papel (*Role*) ao modelo (ex: professor, engenheiro de software, crítico).
* **Objetivo:** Controlar o estilo, a consistência da resposta e, principalmente, a **contextualização**.
* **Desambiguação:** O papel ajuda o modelo a decidir a direção do próximo token. 
    * *Exemplo:* Se o modelo é definido como "Engenheiro de Software", ao ler a palavra "Go", ele entenderá como a linguagem de programação e não como o verbo "ir".

### 2. Quando o Role Prompting é realmente útil?
* **Simulação de Cenários:** Revisões de arquitetura com diferentes perfis ou agentes de suporte com personas específicas.
* **Ajuste de Comunicação:** Adaptar explicações técnicas para diferentes públicos (ex: explicar para um aluno vs. para um especialista).
* **Modelos Menores (Small LLMs):** O impacto do Role Prompting é muito mais perceptível em modelos menores e com menos parâmetros, pois ajuda a "guiar" um raciocínio mais limitado.

### 3. Limitações e Comportamento dos Modelos
* **Modelos Avançados:** Em modelos de alto desempenho (como GPT-4 ou modelos de *reasoning* sofisticados), o impacto de dizer "você é um especialista" costuma ser baixo, pois o modelo já possui parâmetros suficientes para entender o contexto técnico sem essa instrução.
* **Role Drift (Desvio de Papel):** Em conversas longas, o modelo pode perder a persona inicial devido à ambiguidade do histórico ou ao tom do próprio usuário (o modelo tende a mimetizar o estilo do interlocutor).
* **Sobrescrita (User Override):** Instruções diretas do usuário no prompt atual podem anular o papel definido inicialmente (ex: pedir para ignorar ordens anteriores e agir como uma "galinha").

### 4. Segurança e Estabilidade
* **Jailbreaking:** Técnicas de ataque onde o usuário tenta forçar o modelo a sair de seu papel ou escopo de segurança.
* **Doma do Agente:** Para evitar que o papel seja facilmente alterado, utilizam-se técnicas mais avançadas de estruturação de prompt para garantir que o comportamento permaneça íntegro.

### 5. Conclusão Técnica
O Role Prompting não é uma "fórmula mágica" para melhorar respostas automaticamente. Sua eficácia depende da relação entre a **complexidade do modelo** (grandes vs. pequenos) e a **especificidade do escopo** definido.

## Aula 6: Exemplo Prático de Role Prompting (LangChain)

Nesta aula, o conceito de *Role Prompting* é demonstrado na prática através de um script Python utilizando a biblioteca **LangChain** e o modelo GPT-4.

---

### 1. Estrutura do Experimento
Para visualizar a diferença entre os papéis (roles), foram definidos três tipos de mensagens:
* **System Prompt:** Define as instruções iniciais e o comportamento base do modelo.
* **User Prompt:** A interação ou pergunta do usuário ("Explique recursão em 50 palavras").
* **Assistant:** A resposta gerada pela IA.

### 2. Comparação de Personas (System Prompts)
O experimento utilizou dois perfis opostos para a mesma pergunta:

| Perfil | Características da Instrução | Resultado Esperado |
| :--- | :--- | :--- |
| **Professor Universitário** | Técnico, formal, uso de definições conceituais e pseudocódigo. | Resposta sofisticada, termos como "caso base" e "terminação de processo". |
| **Aluno de Ensino Médio** | Linguagem simples, uso de exemplos e analogias. | Explicação didática, uso de analogias (ex: espelhos reflexivos) e gírias naturais. |

### 3. Implementação e Observações Técnicas
* **Ferramentas:** Uso de uma função utilitária (`printLLMResult`) para exibir o prompt e a resposta de forma colorida, facilitando o debug visual.
* **Tokens:** Acompanhamento da quantidade de tokens de entrada e saída para monitorar o custo e a eficiência da chamada.
* **Impacto no Modelo:** A demonstração prova que o *System Prompt* altera drasticamente não apenas o conteúdo, mas o **tom** e a **didática** da resposta.

### 4. Conclusões Chave
* **Especificidade é fundamental:** Um papel genérico (ex: "Cozinheiro") é menos eficiente que um papel detalhado (ex: "Cozinheiro especialista em steaks americanos apimentados").
* **Flexibilidade de Posicionamento:** Embora o *Role Prompting* seja comumente definido no *System Prompt*, ele também pode ser inserido ou redefinido no *User Prompt* durante a conversa.
* **Contextualização Semântica:** Definir a persona ajuda a IA a desambiguar termos (ex: identificar que "Go" se refere à linguagem de programação e não ao verbo, caso o papel seja de um Engenheiro de Software).

```execute o exemplo com "make run" e selecione a opção 20-prompt-engineering/1-tipos-de-prompts/0-Role-prompting.py ```

## Aula 7: Zero-shot:

Esta aula mostra como formular instruções diretas sem fornecer exemplos: o modelo responde apenas com base na instrução.

- **Descrição:** O modelo recebe a tarefa sem exemplos prévios e tem que generalizar apenas pela instrução.
- **Exemplo de prompt:** "Resuma o texto abaixo em uma frase clara e objetiva. Texto: \"...\""
- **Arquivo de exemplo:** `prompt-engineering/1-tipos-de-prompts/1-zero-shot.py`
- **Como executar:**
```
make run-zero-shot
# ou
python prompt-engineering/1-tipos-de-prompts/1-zero-shot.py
```
## Aula 8: Few-shot:

Nesta aula introduzimos exemplos no prompt para guiar o comportamento do modelo (poucos exemplos — few-shot).

- **Descrição:** Fornece 1–5 exemplos de entrada→saída no prompt para demonstrar o formato desejado antes da nova tarefa.
- **Exemplo de prompt:** "Exemplo1: Entrada: \"2+2\" → Saída: \"4\"; Exemplo2: Entrada: \"3*3\" → Saída: \"9\"; Agora: Entrada: \"5-2\" →"
- **Arquivo de exemplo:** `prompt-engineering/1-tipos-de-prompts/2-one-few-shot.py`
- **Como executar:**
```
make run-one-few-shot
# ou
python prompt-engineering/1-tipos-de-prompts/2-one-few-shot.py
```
## Aula 9: Chain of Thought:

Aqui pedimos explicitamente que o modelo exponha seu raciocínio (cadeia de pensamento) passo a passo para tarefas complexas.

- **Descrição:** Solicita que o modelo descreva os passos intermediários antes de dar a conclusão, útil em problemas de raciocínio ou matemática.
- **Exemplo de prompt:** "Pense passo a passo: qual a soma de 47 + 58? Mostre o raciocínio e o resultado."
- **Arquivo de exemplo:** `prompt-engineering/1-tipos-de-prompts/3-CoT.py`
- **Como executar:**
```
python prompt-engineering/1-tipos-de-prompts/3-CoT.py
# ou via menu
make run  # escolha a opção correspondente (menu 23)
```
## Aula 10: Chain of Thought com Self-consistency:

Combina CoT com múltiplas amostras de raciocínio independentes e escolhe a resposta mais consistente entre elas.

- **Descrição:** Gera várias trajetórias de raciocínio (amostras) para a mesma pergunta e agrega as respostas, reduzindo erros aleatórios.
- **Exemplo de prompt:** "Gere 5 raciocínios independentes passo a passo para resolver: qual é 23×17? Agrupe respostas e escolha a mais frequente."
- **Arquivo de exemplo:** `prompt-engineering/1-tipos-de-prompts/3.1-CoT-Self-consistency.py`
- **Como executar:**
```
make run-cot-self-consistency
# ou
python prompt-engineering/1-tipos-de-prompts/3.1-CoT-Self-consistency.py
```
## Aula 11: Tree of Thought:

Explora múltiplos ramos de pensamento (árvore), avaliando e podando caminhos promissores antes de escolher a solução.

- **Descrição:** Em vez de uma única cadeia, expande alternativas em cada passo (nós da árvore), avalia e seleciona os melhores ramos.
- **Exemplo de prompt:** "Explore caminhos alternativos para resolver: planejar rota mínima que passe por A,B,C — expanda duas ações por passo e escolha o melhor ramo."
- **Arquivo de exemplo:** `prompt-engineering/1-tipos-de-prompts/4-ToT.py`
- **Como executar:**
```
make run-tot
# ou
python prompt-engineering/1-tipos-de-prompts/4-ToT.py
```
## Aula 12: Skeleton of Thought:

Primeiro pede um esqueleto (sumário dos passos), depois solicita a expansão detalhada de cada etapa — combina rapidez com clareza.

- **Descrição:** Solicita uma estrutura concisa dos passos (skeleton) e em seguida pede a expansão detalhada de cada item.
- **Exemplo de prompt:** "1) Liste os passos principais para resolver X (esqueleto). 2) Expanda cada passo com detalhes práticos."
- **Arquivo de exemplo:** `prompt-engineering/1-tipos-de-prompts/5-SoT.py`
- **Como executar:**
```
make run-sot
# ou
python prompt-engineering/1-tipos-de-prompts/5-SoT.py
```

## Aula 12: ReAct:
Nesta aula, o modelo é instruído a agir (Action) e raciocinar (Reasoning) de forma intercalada, permitindo que ele execute ações intermediárias (ex: buscar dados) antes de concluir a resposta.

- **Descrição:** O modelo alterna entre raciocinar e agir, podendo realizar ações como consultas a APIs ou bancos de dados durante o processo de resposta.
- **Exemplo de prompt:** "Raciocine e aja: para responder 'Qual é a capital da França?', primeiro raciocine sobre o país, depois aja buscando a informação e finalmente conclua com a resposta."
- **Arquivo de exemplo:** `prompt-engineering/1-tipos-de-prompts/6-ReAct.py`
- **Como executar:**
```make run-react
# ou
python prompt-engineering/1-tipos-de-prompts/6-ReAct.py
```

## Aula 13: Prompt Chaining:
Aqui encadeamos múltiplos prompts, onde a saída de um é a entrada do próximo, permitindo construir fluxos de trabalho complexos e modulares.

- **Descrição:** Cria uma sequência de prompts onde cada etapa depende da resposta anterior, facilitando a construção de processos mais complexos.
- **Exemplo de prompt:** "Prompt 1: Resuma o artigo X. Prompt 2: Com base no resumo, gere perguntas de compreensão. Prompt 3: Responda às perguntas geradas."
- **Arquivo de exemplo:** `prompt-engineering/1-tipos-de-prompts/7-Prompt-chaining.py`
- **Como executar:**
```make run-prompt-chaining
# ou
python prompt-engineering/1-tipos-de-prompts/7-Prompt-chaining.py
```

# Conceitos Importantes

Este bloco aprofunda os fundamentos que sustentam o Prompt Engineering na prática: como o modelo "enxerga" o texto (janela de contexto), como isso afeta custo e latência, e quais técnicas usamos para caber e otimizar informação dentro desses limites.

## Revisitando conceitos

Antes de avançar para os fundamentos de contexto, esta aula recapitula os pilares já vistos para garantir uma base sólida.

---

### 1. Por que revisitar?
* **Consolidação:** As técnicas de prompt (Role, Zero/Few-shot, CoT, ToT etc.) só fazem sentido quando combinadas com o entendimento de **como o modelo processa o texto**.
* **Ponte para o próximo bloco:** Os conceitos a seguir (janela de contexto, custo, latência) são **transversais** — afetam todas as técnicas anteriores.

### 2. Ideias-chave retomadas
* **O prompt é a "linguagem de programação" probabilística:** orienta o comportamento sem regras rígidas de `if/else`.
* **Qualidade > quantidade:** mais texto nem sempre é melhor; ruído degrada a resposta.
* **Tudo tem custo:** cada palavra enviada e gerada consome **tokens**, que se traduzem em **dinheiro e tempo**.

## Context Window para Prompt Engineering

Esta aula define o conceito central que limita e molda toda interação com um LLM: a **janela de contexto** (*context window*).

---

### 1. O que é a Janela de Contexto?
* **Definição:** É a quantidade máxima de **tokens** (entrada + saída) que o modelo consegue processar em uma única requisição.
* **Token:** Unidade básica de texto (≈ ¾ de uma palavra em inglês; em português costuma ser um pouco mais "caro"). Inclui palavras, partes de palavras, pontuação e espaços.
* **Memória de curto prazo:** A janela funciona como a "memória de trabalho" do modelo. Tudo o que não cabe nela, simplesmente **não existe** para o modelo naquela chamada.

### 2. Por que importa para o Prompt Engineering?
* **Orçamento de tokens:** Todo prompt deve ser pensado como um orçamento limitado — System Prompt, histórico, exemplos (few-shot), documentos e a resposta competem pelo mesmo espaço.
* **Priorização:** Informação relevante deve estar dentro da janela; o engenheiro de prompt decide o que entra, o que resume e o que descarta.
* **Posição importa:** Modelos tendem a dar mais atenção ao início e ao fim do contexto (*lost in the middle*); informação crítica enterrada no meio pode ser ignorada.

## Context Window vs memória, custo e latência

Aqui conectamos o tamanho do contexto às três variáveis práticas que mais impactam aplicações reais: **memória, custo e latência**.

---

### 1. Janela de Contexto ≠ Memória Persistente
* **Sem estado (stateless):** O modelo **não lembra** de conversas anteriores por si só. A "memória" de um chat é uma ilusão criada ao **reenviar o histórico** a cada requisição.
* **Consequência:** Conversas longas crescem o prompt continuamente, consumindo mais janela a cada turno.

### 2. Impacto no Custo
* **Cobrança por token:** A maioria das APIs cobra por **tokens de entrada** e **tokens de saída** (geralmente a saída é mais cara).
* **Efeito cumulativo:** Reenviar todo o histórico em cada turno faz o custo crescer de forma quase **quadrática** ao longo de uma conversa longa.
* **Otimização:** Prompts enxutos e técnicas de compressão (resumo, sliding window) reduzem diretamente a conta.

### 3. Impacto na Latência
* **Mais tokens = mais lento:** Quanto maior o contexto, mais tempo o modelo leva para processar a entrada (*prefill*) e gerar a resposta.
* **Trade-off central:** Há um equilíbrio entre **fornecer contexto suficiente** (qualidade) e **manter o prompt curto** (velocidade e custo).

## Janela de contexto vs parâmetros

Esta aula desfaz uma confusão comum: **tamanho da janela de contexto** e **número de parâmetros** são coisas diferentes.

---

### 1. Definindo cada um
* **Parâmetros:** São os "pesos" aprendidos durante o treinamento (ex: 7B, 70B). Representam o **conhecimento e a capacidade de raciocínio** internalizados do modelo — fixos após o treino.
* **Janela de contexto:** É quanto texto o modelo consegue **ler de uma vez** em tempo de uso (inferência) — sua memória de trabalho temporária.

### 2. A Analogia
* **Parâmetros = "inteligência/educação"** da pessoa (o que ela já sabe).
* **Janela de contexto = "mesa de trabalho"** (quantos documentos ela consegue ter abertos à frente ao mesmo tempo).
* Uma pessoa muito inteligente com uma mesa minúscula tem dificuldade com tarefas longas; uma mesa enorme não compensa falta de conhecimento.

### 3. Por que não confundir?
* **Janela grande ≠ modelo melhor:** Um modelo pode ter contexto enorme e ainda raciocinar mal (poucos parâmetros).
* **Decisão de engenharia:** A escolha do modelo deve pesar **ambos** — capacidade (parâmetros) e capacidade de "ler tudo" (contexto) — conforme a tarefa.

## Truncamento

Primeira das técnicas para lidar com o limite da janela: simplesmente **cortar** o que não cabe.

---

### 1. O que é?
* **Definição:** Quando o conteúdo excede a janela de contexto, partes do texto são **removidas** (normalmente as mais antigas) para caber no limite.
* **Onde ocorre:** Pode ser feito automaticamente pela aplicação/SDK ou manualmente pelo desenvolvedor.

### 2. Vantagens e Riscos
* **Vantagem:** Simples e barato de implementar; garante que a requisição não estoure o limite.
* **Risco:** **Perda de informação** — instruções iniciais ou detalhes importantes podem ser descartados, causando respostas inconsistentes ou "esquecimento" de regras.

### 3. Boas práticas
* **Proteja o essencial:** Mantenha o System Prompt e instruções críticas fora da zona de corte.
* **Use como último recurso:** Quando possível, prefira **sumarização** ou **sliding window** para preservar a semântica.

## Sumarização

Em vez de cortar, **condensar**: substituir trechos longos por um resumo que preserva o essencial.

---

### 1. O que é?
* **Definição:** Técnica em que partes do contexto (ex: histórico antigo da conversa ou documentos) são **resumidas** por um modelo antes de seguir o fluxo.
* **Objetivo:** Reduzir tokens **mantendo o significado**, ao contrário do truncamento que simplesmente descarta.

### 2. Como é aplicada
* **Resumo do histórico:** Conversas longas têm seus turnos antigos comprimidos em um resumo curto, liberando espaço na janela.
* **Resumo de documentos:** Textos extensos (RAG, manuais) são condensados antes de entrar no prompt.
* **Recursivo:** Em fluxos longos, pode-se resumir resumos progressivamente.

### 3. Trade-offs
* **Ganho:** Mantém o "fio da meada" gastando menos tokens.
* **Custo extra:** A sumarização em si é **uma chamada adicional** ao modelo (custo/latência).
* **Risco:** Detalhes finos podem se perder no resumo; é preciso calibrar o quão agressivo ele é.

## Sliding window

Técnica de "janela deslizante": manter apenas uma **janela móvel** das mensagens mais recentes.

---

### 1. O que é?
* **Definição:** Mantém-se no contexto apenas as **N interações mais recentes** (ou os últimos N tokens), descartando as antigas conforme a conversa avança.
* **Analogia:** Como uma janela que desliza sobre o texto, sempre mostrando a parte mais atual.

### 2. Quando usar
* **Conversas contínuas:** Chatbots e assistentes onde o contexto **recente** é o mais relevante.
* **Custo previsível:** O tamanho do prompt permanece **estável** ao longo do tempo, evitando crescimento ilimitado.

### 3. Limitações
* **Esquecimento do início:** Informações antigas (ex: o nome do usuário dado no começo) saem da janela.
* **Combinação comum:** Frequentemente usada **junto com sumarização** — resume-se o que sai da janela para não perder tudo, e mantém-se o recente em detalhe.

## Prompt Caching

Mecanismo de otimização que **reaproveita** partes repetidas do prompt entre requisições para economizar custo e tempo.

---

### 1. O que é?
* **Definição:** Permite **armazenar em cache** trechos estáticos e reutilizados do prompt (ex: System Prompt longo, instruções, documentos de referência, exemplos few-shot).
* **Como funciona:** Em vez de reprocessar o mesmo prefixo a cada chamada, o provedor reutiliza o processamento já feito (o *prefill* daquele trecho).

### 2. Benefícios
* **Redução de custo:** Tokens em cache costumam ser **significativamente mais baratos** que tokens normais de entrada.
* **Redução de latência:** O modelo pula o reprocessamento da parte cacheada, respondendo mais rápido.
* **Ideal para:** Aplicações que repetem o mesmo contexto-base em muitas chamadas (agentes, RAG com instruções fixas, atendimento).

### 3. Boas práticas
* **Estável no início:** Coloque o conteúdo **fixo e reutilizável no começo** do prompt; o que muda (a pergunta do usuário) vai no final.
* **Atenção ao TTL:** O cache tem **tempo de vida** limitado; consultas espaçadas podem perder o benefício.
* **Varia por provedor:** Cada API (OpenAI, Gemini, Anthropic/Claude) implementa o caching com regras e preços próprios — consultar a documentação específica.

## Batch Prompting

Técnica para processar **vários itens de uma só vez**, reduzindo o número de chamadas e o overhead.

---

### 1. O que é?
* **Definição:** Em vez de uma requisição por item, agrupam-se **múltiplas tarefas/perguntas em um único prompt** (ou em um lote enviado de uma vez).
* **Objetivo:** Diluir o **custo fixo** (instruções, System Prompt) entre vários itens e reduzir o número de round-trips.

### 2. Vantagens
* **Economia:** Instruções comuns são enviadas **uma vez** para todo o lote, não repetidas por item.
* **Throughput:** Processa grandes volumes (ex: classificar 100 frases) de forma mais eficiente.
* **Menos latência agregada:** Menos chamadas de rede para a mesma quantidade de trabalho.

### 3. Cuidados
* **Formatação rígida:** É essencial pedir saída estruturada (ex: JSON com índices) para **mapear cada resposta ao seu item**.
* **Limite da janela:** O lote inteiro precisa caber no context window (entrada + saída esperada).
* **Contaminação entre itens:** Itens no mesmo prompt podem se influenciar; tarefas muito sensíveis podem exigir isolamento.

