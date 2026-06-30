# Módulo 4 - Architecture Decision Record (ADR)

## Sumário

- [Aula 1: Introdução a ADRs](#aula-1-introdução-a-adrs)

- [Aula 2: Estrutura clássica](#aula-2-estrutura-clássica)


## Aula 1: Introdução a ADRs

Esta aula apresenta o **ADR (Architecture Decision Record)** como o documento que registra **decisões arquiteturais e, sobretudo, o porquê** delas. O código mostra o "o quê"; o ADR preserva restrições, trade-offs e pressões que motivaram a escolha — virando **memória técnica explícita** que sobrevive ao tempo e à rotatividade do time. Em ambientes com IA, esse registro deixa o modelo trabalhar sobre **contexto declarado** em vez de inferir motivos a partir do código.

---

### 1. ADR como registro do porquê arquitetural
* **Registra decisão + motivo:** Documenta decisões arquiteturais relevantes e, principalmente, **por que** foram tomadas.
* **Código não basta:** Ele mostra o que foi implementado, mas raramente preserva restrições, trade-offs e pressões que levaram à escolha.
* **Memória técnica explícita:** Transforma o raciocínio em registro acessível mesmo anos depois.
* **Evita tradição oral:** Reduz a perda de contexto e impede que decisões importantes virem só lembrança do time.

### 2. Memória técnica explícita e continuidade ao longo do tempo
* **Sistemas duram, pessoas rotacionam:** O projeto evolui por anos enquanto gente entra e sai.
* **Risco da memória individual:** Depender da lembrança de poucos gera interpretações incompletas, retrabalho e reversões mal informadas.
* **Continuidade do raciocínio:** Registrar preserva a escolha final **e** o porquê de ela ter feito sentido naquele momento.
* **Valor crescente:** Aumenta conforme a vida útil do sistema cresce.

### 3. Valor histórico da documentação arquitetural
* **Timeline arquitetural:** O histórico de decisões mostra a arquitetura como uma **sequência de escolhas**, não uma fotografia estática.
* **O que ajuda a entender:** Por que partes do sistema mudaram, quais direções foram priorizadas e quais consequências já eram conhecidas.
* **Manutenção mais segura:** Documentar o percurso torna evolução e manutenção menos arriscadas.

### 4. Quando ADR se torna especialmente útil
* **Impacto estrutural:** Mais valioso quando a decisão afeta múltiplas partes do sistema ou tende a gerar dúvidas no futuro.
* **Casos típicos:** Mudanças arquiteturais relevantes, escolhas com trade-offs importantes e decisões cujo motivo não fica evidente no código.
* **Nem tudo vira ADR:** Decisões operacionais corriqueiras não precisam — o foco é o que influencia a arquitetura.
* **Critério central:** Se **esquecer o porquê** prejudica a evolução do sistema, vale registrar.

### 5. Relação entre ADR e IA
* **IA opera melhor com contexto explícito:** Modelos rendem mais com registro claro do que com artefatos dispersos.
* **O que a IA faz com ADRs:** Recupera intenção arquitetural, relaciona mudanças ao longo do tempo e apoia análise, geração e revisão.
* **Sem registro:** A IA infere motivos a partir do código e de sinais indiretos — mais frágil.
* **Com memória documentada:** A IA deixa de **adivinhar** e passa a trabalhar sobre **contexto declarado**.

## Aula 2: Estrutura clássica

Esta aula detalha o **template clássico de Michael Nygard**: campos simples e suficientes para preservar o raciocínio arquitetural — **identificador + título, contexto, decisão, alternativas/trade-offs, consequências e referências**. Posiciona o ADR no ecossistema documental (não substitui PRD/HLD/LLD) e usa o caso **RabbitMQ × Kafka** para mostrar como o template explica *por que* a arquitetura seguiu um caminho.

---

### 1. ADR no ecossistema documental
* **Registra o porquê:** O ADR documenta por que uma decisão técnica relevante foi tomada.
* **Não substitui os demais:** PRD e HLD descrevem **o que** a solução é e **como** se estrutura em alto nível; o LLD detalha **como** implementar no código.
* **Papel próprio:** O ADR preserva a **justificativa** da escolha arquitetural — cada documento responde a uma pergunta diferente.

### 2. Quando criar um ADR
* **Critério:** Decisão que foi **debatida**, envolveu **trade-offs** e afeta o design/arquitetura do sistema.
* **Evita o trivial:** Concentra esforço nas decisões cujo motivo tende a se perder com o tempo.
* **Baixo custo:** Formato curto e simples — manutenção barata frente ao valor histórico.
* **Timeline arquitetural:** Cada ADR adiciona um ponto explícito na evolução do sistema.

### 3. Template clássico de Michael Nygard
* **Campos simples e suficientes:** Organiza o ADR para preservar o raciocínio sem burocracia.
* **Começo padronizado:** Identificador numérico + título descritivo, ex: "ADR-001 — Usar Redis como cache distribuído".
* **Facilita o uso:** Referência cruzada, ordenação histórica e leitura rápida em repositórios com muitos registros.
* **Baseline:** Serve de fundação conceitual para templates posteriores mais elaborados.

### 4. Identificador numérico e título
* **Número = identidade estável:** Independe de mudanças futuras no texto.
* **Título específico:** Nomeia a decisão de forma precisa — vira ponto de entrada para buscas, links e discussões.
* **Bom × vago:** "Usar Redis como cache distribuído" é melhor que "Decisão de cache".
* **Resultado:** Transforma o ADR numa **unidade documental** fácil de citar e relacionar.

### 5. Contexto
* **O cenário da decisão:** Descreve problema, motivação, restrições e forças em jogo no momento.
* **O que entra:** Limitações técnicas, conhecimento do time, custos, infraestrutura e condições do negócio.
* **Objetivo:** Mostrar em que **cenário concreto** aquela decisão fazia sentido — não defender uma tecnologia em abstrato.
* **Sem ele:** A decisão vira conclusão solta, sem as condições que a justificaram.

### 6. Decisão
* **Registra o caminho adotado:** A partir do contexto apresentado.
* **Direta:** O valor não está em esconder a escolha, mas em **conectá-la** às condições que a tornaram aceitável.
* **Sem ambiguidade:** Deixa claro o compromisso técnico assumido pelo time.
* **Núcleo do documento:** Mas só faz sentido completo lido junto com contexto, alternativas e consequências.

### 7. Alternativas e trade-offs
* **O que foi considerado:** Quais opções existiam e por que não foram escolhidas.
* **Por que é essencial:** Arquitetura quase nunca tem solução perfeita — envolve troca entre benefícios, custos, risco, prazo e capacidade operacional.
* **No exemplo:** RabbitMQ pode ser inferior em algum aspecto isolado, mas ser a melhor decisão sob restrições reais de custo, infraestrutura e conhecimento interno.
* **Ganho:** Torna os trade-offs explícitos para que o futuro interprete a escolha com **justiça técnica**.

### 8. Consequências
* **Efeitos e implicações:** Ganhos esperados, limitações aceitas, impactos operacionais e custos futuros.
* **Decisão não é gratuita:** Mostra o que o time aceitou **em troca** do benefício principal.
* **Apoia revisões:** Mudanças arquiteturais costumam surgir quando essas consequências **deixam de ser aceitáveis**.

### 9. Referências entre documentos
* **Conecta artefatos:** Liga o ADR a outras ADRs, documentos de produto, HLDs e RFCs.
* **Rede navegável:** Transforma documentos isolados em uma teia de raciocínio técnico e histórico.
* **Reduz ambiguidade:** Quando uma decisão depende de outra ou altera uma anterior, o vínculo explícito acelera a investigação.
* **Em sistemas longevos:** Essas conexões são tão importantes quanto o próprio texto do ADR.

### 10. Exemplo aplicado da estrutura
* **Caso RabbitMQ × Kafka:** O template clássico organiza a decisão em camadas claras.
* **Contexto:** Restrições de custo, infraestrutura disponível e conhecimento do time.
* **Decisão / Alternativas:** Adoção do RabbitMQ; Kafka aparece como opção considerada e descartada.
* **Consequências:** Benefícios operacionais e limitações aceitas — um documento curto, mas **suficiente** para explicar o caminho da arquitetura.
