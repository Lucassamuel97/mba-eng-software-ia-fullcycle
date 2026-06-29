# Módulo 3 - Diagramas (Design e Arquitetura)

## Sumário

- [Aula 1: Introdução a Diagramas](#aula-1-introdução-a-diagramas)

- [Aula 2: Introdução aos diagramas C4](#aula-2-introdução-aos-diagramas-c4)

- [Aula 3: C1 - System Context](#aula-3-c1---system-context)

- [Aula 4: C2 - Containers](#aula-4-c2---containers)

- [Aula 5: C3 - Components](#aula-5-c3---components)


## Aula 1: Introdução a Diagramas

Esta aula apresenta os **diagramas** como forma de **reduzir o custo cognitivo** de entender um sistema descrito em texto. O ponto-chave na era da IA é parar de desenhar do zero e passar a **derivar o diagrama da especificação** (HLD, FDD, LLD, ADRs, RFCs) que já é a fonte de verdade. Apresenta dois eixos complementares — **C4** (granularidade arquitetural) e **Mermaid** (linguagem textual leve) — sempre com **revisão humana** sobre o que a IA gera.

---

### 1. Diagramas como redução de custo cognitivo
* **Para que existem:** Reduzir o esforço de entender um sistema descrito em texto.
* **O problema do texto:** Documentos acumulam regras, componentes, decisões e exceções em muitas linhas, dificultando ver estrutura, dependências e fluxo.
* **A compressão visual:** A representação espacial é mais fácil de inspecionar e comunicar.
* **Ganho duplo:** Melhora a análise técnica e o alinhamento com time, liderança e demais áreas.

### 2. O atrito histórico para criar e manter diagramas
* **Valiosos, mas caros:** Sempre ajudaram, mas eram custosos de produzir e manter.
* **Causas do atrito:** Ferramentas específicas, metodologias pesadas e esforço manual de atualização — muitos times desistiam ou deixavam o diagrama desatualizado.
* **Paradoxo comum:** Onde a arquitetura era mais complexa, a documentação visual era a que mais faltava.
* **Sem manutenção:** O diagrama deixava de esclarecer e passava a **competir com a realidade**.

### 3. A especificação como fonte de verdade
* **Derivar, não desenhar do zero:** O diagrama nasce da especificação que já existe.
* **As fontes:** HLD, Feature Design Docs, LLD, ADRs e RFCs concentram as decisões e servem de base para visões visuais consistentes.
* **Novo papel do diagrama:** Deixa de ser artefato isolado e vira **projeção da documentação textual**.
* **Dependência:** Quanto melhor a especificação, maior a chance de diagramas úteis e coerentes.

### 4. IA para geração e atualização de diagramas
* **Reduz o esforço:** Transformar texto técnico em representação visual fica mais barato.
* **Como opera:** Pedir que a IA extraia atores, componentes, relações e nível de detalhe direto da especificação.
* **Ganho na manutenção:** Quando o projeto evolui, atualizar o documento-fonte é mais barato do que redesenhar tudo.
* **Não elimina validação:** A automação acelera um **rascunho revisável**, não uma verdade pronta.

### 5. Revisão humana obrigatória
* **Não é verdade automática:** O modelo pode inferir relações inexistentes, omitir restrições ou exagerar a granularidade.
* **O que a revisão garante:** Aderência à arquitetura real, ao vocabulário do domínio e ao objetivo do diagrama.
* **Uso correto:** Assistivo — a IA produz e atualiza mais rápido, mas a **responsabilidade técnica** fica com quem conhece o sistema.

### 6. C4 como estrutura de granularidade
* **O que é:** Uma forma de representar sistemas em **níveis progressivos** de detalhe.
* **Seu valor:** Organiza a arquitetura por camadas, do panorama amplo ao recorte específico.
* **Escolher o nível certo:** Evita misturar contexto executivo com detalhe de implementação.
* **Cuidado:** Granularidade excessiva gera custo e ruído — nem todo nível precisa ser usado.

### 7. Mermaid como linguagem textual leve
* **Outra lógica:** Em vez de priorizar estrutura arquitetural como o C4, oferece **sintaxe textual** para descrever diagramas direto em texto.
* **Documentação versionada:** O diagrama vive no mesmo repositório e arquivo da especificação.
* **Barreira menor:** Facilita a adoção no dia a dia.
* **Quando brilha:** Quando rapidez, portabilidade e edição simples importam mais que formalismo arquitetural.

### 8. Renderização automática em Markdown e GitHub
* **Vantagem prática:** Ambientes com Markdown enriquecido (como o GitHub) renderizam o bloco Mermaid sem ferramenta externa.
* **Menos atrito:** Facilita revisão em pull requests e mantém código, documentação e diagrama próximos.
* **Consequência:** Atualizar o diagrama passa a se parecer com **editar texto**, não operar software gráfico.

### 9. Dois eixos complementares para representar sistemas
* **Não competem:** C4 e Mermaid resolvem problemas diferentes e podem coexistir.
* **C4:** Estrutura forte para pensar **níveis de abstração** arquitetural.
* **Mermaid:** Forma leve e textual de **materializar** diagramas em documentação contínua.
* **Com IA:** A mesma especificação origina visões distintas com menos esforço manual.

## Aula 2: Introdução aos diagramas C4

Esta aula apresenta o **modelo C4** como documentação arquitetural em **camadas progressivas** de detalhe: contexto (C1), containers (C2), componentes (C3) e código (C4-Code). A regra central é a **leitura progressiva** — cada nível aprofunda o anterior sem substituí-lo —, escolhendo o nível certo para a pergunta certa. Cita as referências do ecossistema (c4model.com, Structurizr, PlantUML) e mantém o **Rate Limiter** como exemplo contínuo, agora observado em camadas.

---

### 1. Modelo C4 como documentação arquitetural em camadas
* **Mesmo sistema, níveis de detalhe:** Organiza a arquitetura em profundidades progressivas, em vez de um único diagrama.
* **Quatro visões:** Separa contexto, containers, componentes e código — cada uma responde a uma pergunta diferente.
* **Reduz ambiguidade:** Permite aprofundar a análise sem perder coerência entre camadas.
* **Redução sistemática:** Retomando diagramas como redução de custo cognitivo, o C4 estrutura essa redução.

### 2. Os quatro níveis: C1, C2, C3 e C4-Code
* **C1 (Contexto):** O sistema no seu ambiente, incluindo usuários e sistemas externos.
* **C2 (Containers):** Os containers que compõem a solução.
* **C3 (Componentes):** Os componentes internos de um container.
* **C4-Code:** Quando necessário, aproxima a documentação da organização do código.
* **Utilidade:** A **progressão** — sair do macro ao detalhe estrutural sem trocar de domínio nem refazer a explicação.

### 3. Leitura progressiva do mesmo sistema
* **Regra central:** Cada nível **aprofunda** o anterior, não o substitui.
* **Evita sobrecarga:** Impede diagramas superlotados e adapta a explicação ao público e à decisão técnica.
* **Nível por contexto:** Conversa executiva → contexto basta; revisão de implementação → componentes ou código.
* **O valor:** Não é produzir todos os níveis sempre, mas escolher o **nível certo para a pergunta certa**.

### 4. c4model.com como referência oficial
* **Fonte primária:** Referência oficial para terminologia, proposta e tipos de diagrama esperados.
* **Autoria:** Escrito por Simon Brown, criador do C4 — alinha nomenclatura e intenção arquitetural.
* **Por que importa:** Times usam "C4" de forma vaga; a referência reduz interpretações divergentes e melhora a consistência.

### 5. Structurizr no ecossistema C4
* **O que é:** Ferramenta do ecossistema C4 para modelar uma base arquitetural e visualizar diagramas derivados dela.
* **Ideia central:** Manter uma representação estruturada do sistema e explorar diferentes visões a partir dela, em vez de desenhar cada diagrama isolado.
* **Benefício:** Favorece consistência entre níveis e facilita a evolução da documentação.
* **Para quem:** Equipes que querem operacionalizar C4 com menos esforço manual.

### 6. PlantUML como meio textual de geração
* **Diagrama em texto:** Meio comum para gerar diagramas C4 sem edição gráfica manual.
* **Ganho de processo:** Permite versionamento, revisão e automação com o mesmo fluxo do desenvolvimento.
* **Combina com o módulo:** Alinha-se à proposta de gerar e manter diagramas a partir de documentos-fonte.
* **Quando usar:** Para integrar arquitetura ao repositório e ao processo de revisão, com menos atrito.

### 7. IA como apoio à interpretação arquitetural
* **O que faz:** Lê descrições do sistema e infere contexto, fronteiras, relações e níveis de detalhe compatíveis com o C4.
* **Mais que caixas e setas:** Ajuda a transformar base textual em visões úteis para análise, refatoração e geração de artefatos.
* **Depende de moldura:** Precisa de uma estrutura clara de leitura — e o C4 oferece exatamente essa moldura.
* **Validação humana:** Continua necessária para confirmar se a interpretação corresponde ao sistema real.

### 8. Rate Limiter como exemplo contínuo
* **Mesmo domínio, nova lente:** O avanço não é o problema, mas observá-lo em **camadas**.
* **Sem trocar de exemplo:** O mesmo sistema é lido em contexto, containers, componentes e código.
* **O que permite ver:** Como uma única solução muda de escala conforme a pergunta fica mais ampla ou mais detalhada.
* **Próximo passo:** O **C1**, que mostra o Rate Limiter no seu ambiente externo.

## Aula 3: C1 - System Context

Esta aula detalha o **C1 (System Context)**: a visão de **fora do sistema**, mostrando quem o usa, com quais sistemas ele se integra e qual papel cumpre no ecossistema — sem abrir a estrutura interna. Usa o caso menos óbvio do **Rate Limiter como SDK embutido**, aplicando System Boundary, atores, dependências externas (storage e observabilidade) e anotações de propósito para que a biblioteca não "desapareça" dentro dos serviços consumidores.

![C1 - System Context do Rate Limiter na Plataforma de Microsserviços](/docs/design-docs/assets/c1-system-context-rate-limiter.png)

---

### 1. C1 como visão externa do sistema
* **Visão de contexto:** Mostra o sistema **a partir de fora**, no ambiente em que opera.
* **O que responde:** Quem usa, com quais sistemas se integra e qual papel cumpre — não a estrutura interna.
* **Quando é útil:** Para vários times alinharem rapidamente fronteiras, responsabilidades e dependências sem detalhes de implementação.

### 2. Granularidade do nível de contexto
* **Deliberadamente alta:** O C1 abstrai o detalhe de propósito.
* **Em ecossistemas grandes:** Reduz ruído e destaca só relações arquiteturalmente relevantes — consumidores, atores e integrações externas.
* **Foco do nível:** Ainda não abre containers ou componentes; fixa o **recorte correto** do que será aprofundado depois.

### 3. Biblioteca ou SDK como sistema central embutido
* **Caso menos óbvio:** O sistema central não é um serviço isolado, mas um **SDK embutido** em outros serviços.
* **Tratado como o sistema:** Quando a intenção é documentar seu contexto operacional — quem usa, quem configura e de quais dependências precisa.
* **Por quê:** Evita que a biblioteca **desapareça** visualmente dentro dos microserviços consumidores e explicita seu papel arquitetural.

### 4. System Boundary
* **O que delimita:** Separa o que pertence ao sistema documentado do que está fora dele.
* **No diagrama:** Distingue o que é desenvolvido e mantido como Rate Limiter das pessoas e sistemas que interagem com ele.
* **Sem essa marcação:** O leitor pode confundir o SDK com a plataforma inteira ou interpretar dependências externas como partes internas.

### 5. Atores do contexto
* **Usuário final:** Aparece porque interage com os Endpoints HTTP protegidos pelo middleware.
* **Desenvolvedor:** Entra como ator porque integra, mantém e configura o SDK nos serviços consumidores.
* **Por que faz sentido:** Em uma biblioteca embutida, parte importante do uso acontece por **integração técnica**, não por interface de negócio.

### 6. Microserviços consumidores do Rate Limiter
* **Consumidores diretos:** Os microserviços usam o SDK para decisões de **Allow/Deny** nas requisições HTTP.
* **Padronização:** Também padronizam a observabilidade associada ao controle.
* **No C1:** Aparecem como sistemas **ao redor** do Rate Limiter, não como detalhamento interno dele.

### 7. Storage compartilhado como dependência externa
* **Estado por chave:** Rate limiting distribuído precisa manter contadores por IP, usuário ou outro identificador.
* **Por que não memória local:** Com várias instâncias rodando, cada uma veria um estado diferente.
* **No diagrama:** Um armazenamento compartilhado de estado aparece como **dependência externa opcional** para sustentar a limitação de forma consistente.

### 8. Observabilidade como sistema externo
* **Não opera isolado:** O Rate Limiter precisa alimentar a camada de observabilidade do ecossistema.
* **O que permite acompanhar:** Volume de requisições, decisões de bloqueio e comportamento operacional.
* **No C1:** Importa menos pelo detalhe da instrumentação e mais por deixar explícito que observabilidade faz parte do **contrato arquitetural**.

### 9. Anotações de propósito em C1 para bibliotecas embutidas
* **Nome nem sempre basta:** Em sistemas tradicionais o nome do serviço comunica a função; numa biblioteca embutida, não.
* **Anotação curta ajuda:** Esclarece o que ela faz — limitar por chave/IP/plano, decidir localmente, usar estado compartilhado opcional e padronizar observabilidade.
* **Não é obrigatória:** Mas evita que o elemento central pareça **genérico demais** quando não é um processo independente.

### 10. Aplicação ao exemplo do Rate Limiter
* **Recorte final do C1:** No centro, a biblioteca/SDK; ao redor, usuário final, desenvolvedor, microserviços consumidores, storage compartilhado e observabilidade.
* **O que o desenho responde:** Onde o Rate Limiter vive, quem o usa, quem o configura e de quais sistemas depende.
* **O que fica fechado:** A estrutura interna — essa abertura fica para o **C2**, no próximo nível.

## Aula 4: C2 - Containers

Esta aula aprofunda o C1 e mostra a **organização executável** do sistema no **C2 (Containers)**. Aqui "container" **não é Docker**: é um bloco implantável/executável com responsabilidade arquitetural (serviço, banco, fila, API). O nível revela quais partes existem, como se comunicam (verbo + protocolo) e quais dependências externas sustentam a operação — servindo de **ponte para deploy, integração e infraestrutura**. Reforça também a **fidelidade à especificação**: o diagrama não inventa fluxos que o documento-fonte não declara.

![C2 - Containers do Rate Limiter (Serviço HTTP da Plataforma)](/docs/design-docs/assets/c2-containers-rate-limiter.png)

---

### 1. Mudança de granularidade do C1 para o C2
* **Do externo ao executável:** O C2 aprofunda a visão do C1 e mostra a organização executável do sistema.
* **"Container" ≠ Docker:** É um bloco implantável/executável relevante para entender a operação.
* **Novo objetivo:** Mostrar quais partes existem, como se comunicam e quais dependências tornam o sistema viável.
* **Ponte:** Liga a arquitetura lógica a decisões de deploy, integração e infraestrutura.

### 2. O que é container no modelo C4
* **Unidade com responsabilidade:** Execução ou armazenamento com papel claro — serviço, aplicação, banco, fila ou API.
* **Nome ≠ empacotamento:** Escolhido para organizar o sistema em blocos operacionais, não para indicar tecnologia.
* **Erro comum:** Confundir o diagrama com a topologia de Docker ou Kubernetes.
* **O que importa:** O **papel executável** do bloco dentro do sistema documentado.

### 3. Container principal do Rate Limiter
* **Serviço HTTP em Go:** O container central é um serviço com a biblioteca de rate limit embutida **in-process**.
* **Como se apresenta:** Expõe endpoints e aplica middleware HTTP para checagem de limites — ponto central de execução da lógica.
* **Avanço sobre o C1:** Em vez de só descrever uma biblioteca usada por outros, explicita **onde a lógica roda**.
* **Efeito:** Transforma a visão conceitual em algo utilizável por quem implanta, integra e opera.

### 4. Anotações técnicas do container
* **Condensam decisões:** Ajudam a interpretar o bloco sem abrir um nível mais profundo.
* **No exemplo:** Modos de armazenamento (Redis com script Lua para atomicidade, InMemory com locks por chave), estratégias (janela fixa, token bucket) e headers padronizados.
* **Por que existem:** Comunicam restrições e capacidades sem exigir um C3 imediato.
* **Bem usadas:** Tornam o container semanticamente mais rico **sem poluir** a leitura.

### 5. Relações com verbos e protocolos
* **Setas comunicam natureza:** Não servem só para ligar caixas.
* **Verbo + protocolo:** "Leitura e atualização", "exporta métricas" ou "usa" deixam clara a responsabilidade; HTTP refina o meio.
* **Forma flexível:** Pode-se priorizar o verbo e pôr o protocolo entre parênteses/chaves, desde que a leitura siga clara.
* **Critério semântico:** Quem lê deve entender **o que flui** entre os blocos e **por qual meio**.

### 6. Observabilidade com Prometheus em modelo pull
* **HTTP pull:** O Prometheus **busca** as métricas no serviço, em vez de recebê-las por envio ativo.
* **Direção correta:** Esse detalhe muda a direção da relação no diagrama e evita representação enganosa.
* **Na prática:** O serviço **expõe** métricas para scrape; o Prometheus faz a leitura periódica do endpoint.
* **Por que no C2:** Esse comportamento afeta configuração, rede e operação.

### 7. Redis como dependência operacional
* **Estado distribuído:** Fica em Redis quando o Rate Limiter precisa operar entre múltiplas instâncias.
* **O que sustenta:** Contadores e buckets por chave compartilhados, preservando o limite mesmo com escala horizontal.
* **No diagrama:** Não é detalhe interno, mas **dependência operacional indispensável**.
* **Para quem importa:** É a informação que times de infraestrutura e plataforma precisam enxergar.

### 8. OpenTelemetry Collector e fidelidade à especificação
* **Sistema externo:** Recebe traces e potencialmente métricas e logs.
* **Seguir a fonte:** Se o documento não diz que o Prometheus lê métricas do Collector, o diagrama **não inventa** esse fluxo só por elegância.
* **Princípio do uso de IA:** **Fidelidade ao texto de origem** vale mais que completar lacunas com suposições plausíveis.
* **Ordem:** Quando a especificação mudar, o diagrama muda junto — antes disso, precisão documental vem primeiro.

### 9. Limite do sistema e containers externos
* **System Boundary continua:** Separa o que pertence ao sistema do que apenas se integra a ele.
* **Consequência prática:** Um container pode ter a mesma aparência estrutural de outro e ainda estar **fora** do sistema.
* **No exemplo:** Redis, Prometheus e OpenTelemetry Collector são relevantes, mas **externos** ao limite do Rate Limiter.
* **Evita confusão:** Impede que dependências sejam lidas como partes internas.

### 10. C2 como insumo para deploy e infraestrutura
* **Revela o necessário:** Mostra quais blocos precisam existir para o sistema funcionar fora do papel.
* **Informa operação:** Redis como requisito, Prometheus em pull e Collector como destino de telemetria orientam rede, provisionamento e observabilidade.
* **Discute escala e custo:** Inclusive sampling, para não enviar 100% dos sinais de telemetria.
* **Mais que visão intermediária:** É um artefato para transformar arquitetura em **ambiente executável**.

### 11. Aplicação prática de leitura do diagrama
* **Comece pelo centro:** O container principal dentro da fronteira e sua responsabilidade executável.
* **Percorra as setas:** Verbo e protocolo — chamadas de clientes, leitura/atualização em Redis, scrape do Prometheus e exportação ao Collector.
* **Use as anotações:** Para inferir capacidades e restrições (estratégias de limitação, modos de armazenamento).
* **Conecta desenho e operação:** O que roda, do que depende e como cada integração acontece.

## Aula 5: C3 - Components

Esta aula abre a "caixa preta" do container do C2 e mostra o **C3 (Components)**: os componentes internos que colaboram para decidir **allow/deny**. Apresenta o pipeline interno — **HTTP middleware → Key Extractor → API interna → Strategy Engine → Storage Adapter** — com cada peça em um contrato distinto. Reforça o **grounding no FDD**: cada componente e anotação é rastreável ao documento-fonte, reduzindo alucinação na geração assistida por IA.

![C3 - Components do Serviço HTTP com SDK embutido (Rate Limiter)](/docs/design-docs/assets/c3-component-rate-limiter.png)

---

### 1. C3 como decomposição interna de um container
* **Abre o container do C2:** Detalha a estrutura interna do serviço HTTP em vez de tratá-lo como caixa preta.
* **O que expõe:** Os componentes que colaboram para a decisão de allow/deny.
* **Objetivo:** Tornar visíveis responsabilidades, contratos internos e pontos de acoplamento — não redesenhar o sistema.
* **Ganho:** Preserva coesão entre módulos e reduz dependências desnecessárias.

### 2. Fronteira visual do C3
* **Só o container aberto é decomposto:** Dependências externas continuam como containers externos.
* **No exemplo:** Redis, Prometheus e OpenTelemetry permanecem fora — não pertencem ao interior lógico do Rate Limiter.
* **Convenção visual:** Evita misturar "o que está dentro do container" com "o que o container usa".
* **Resultado:** A fronteira do container segue explícita mesmo com mais detalhe.

### 3. HTTP middleware como ponto de entrada
* **Toda requisição passa por ele:** Antes de seguir no pipeline do servidor.
* **O que faz:** Recebe a chamada, extrai a identidade, aciona a API interna, interpreta a decisão e escreve os headers.
* **Papel de orquestração:** É o ponto de coordenação do fluxo, mesmo sem concentrar toda a lógica de negócio.
* **Vantagem:** Separa entrada HTTP, decisão de limitação e persistência em contratos distintos.

### 4. Key Extractor
* **Transforma requisição em chave:** Deriva a chave de rate limiting da requisição.
* **O que a chave representa:** IP, API key, API token ou tenant, conforme a política.
* **Por que existe:** O limite nunca é calculado "para a requisição em si", mas para uma **identidade** derivada dela.
* **Multi-tenant:** Define corretamente quem consome a cota e evita aplicar o mesmo contador a entidades diferentes.

### 5. API pública interna do Rate Limiter
* **Contrato interno:** Exposto para outros componentes do mesmo container, especialmente o middleware.
* **"Pública" ≠ externa:** Significa interface acessível e estável **dentro do processo** para solicitar a checagem.
* **Por que importa:** O middleware depende do **contrato**, não da implementação concreta de estratégia ou storage.
* **Resultado:** Composição interna mais testável e fácil de evoluir.

### 6. Strategy Engine
* **Decide o algoritmo:** Seleciona qual estratégia de limitação aplicar à requisição corrente.
* **No exemplo:** Entre fixed window e token bucket, preservando paridade comportamental entre os modos.
* **Por que existe:** Cada estratégia tem regras próprias de consumo, reposição ou contagem.
* **Centraliza a escolha:** Evita espalhar condicionais de algoritmo pelo middleware e pelo acesso ao estado.

### 7. Fixed window e token bucket
* **Fixed window:** Conta requisições em janelas discretas ("N por minuto") e reinicia ao trocar de janela.
* **Token bucket:** Modela capacidade por tokens consumidos a cada requisição e repostos no tempo, permitindo bursts controlados.
* **Impacto no cliente:** A escolha altera o comportamento percebido, sobretudo em bordas de janela e picos curtos.
* **No C3:** Por isso o diagrama mostra **onde** essa decisão algorítmica é tomada.

### 8. Storage adapter
* **Encapsula estado:** Abstrai leitura e atualização do estado usado pelas estratégias.
* **O que esconde:** Se o estado está em memória local com locks por chave ou em Redis com script Lua (distribuído e atômico).
* **Isolamento:** A lógica de estratégia não conhece detalhes de persistência e sincronização.
* **Benefício:** Mantém o mesmo contrato interno mesmo trocando entre execução local e distribuída.

### 9. Fluxo interno da requisição
* **Início no middleware:** Intercepta a requisição e delega a extração ao Key Extractor.
* **Decisão:** Chama a API interna, que usa o Strategy Engine para escolher o algoritmo e consulta/atualiza o estado via storage adapter.
* **Resposta:** Com a decisão pronta, o middleware devolve allow/deny e escreve os headers.
* **Efeito:** Transforma o container antes visto como bloco único em um **mecanismo interno compreensível**.

### 10. Grounding no FDD
* **Rastreabilidade:** Cada componente e anotação pode ser ligado a seções do Feature Design Doc.
* **No exemplo:** Headers e comportamentos de storage são justificados por seções específicas do documento-fonte, não inferidos livremente.
* **Reduz alucinação:** Mantém o diagrama fiel ao design aprovado na geração assistida por IA.
* **Próximo passo:** Materializar esses contratos internos em artefatos mais próximos do código.
