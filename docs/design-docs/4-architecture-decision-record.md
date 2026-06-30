# Módulo 4 - Architecture Decision Record (ADR)

## Sumário

- [Aula 1: Introdução a ADRs](#aula-1-introdução-a-adrs)

- [Aula 2: Estrutura clássica](#aula-2-estrutura-clássica)

- [Aula 3: MADRs](#aula-3-madrs)

- [Aula 4: Status, Metadados e Fluxos](#aula-4-status-metadados-e-fluxos)

- [Aula 5: Boas práticas e dicas importantes](#aula-5-boas-práticas-e-dicas-importantes)


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

## Aula 3: MADRs

Esta aula apresenta o **MADR (Markdown ADR)**: um formato padronizado em Markdown, **legível por pessoas e parseável por máquina**. Ele não muda o conteúdo lógico da decisão — muda a **disciplina documental**: metadados estruturados (`status`, `date`, `supersedes`, `superseded by`, `amends`) e seções fixas que viabilizam **busca, validação, automação em CI/CD** e uso por IA. O critério final é **consistência suficiente**, adaptável ao contexto da organização.

---

### 1. MADR como operacionalização do ADR
* **Formato padronizado em Markdown:** Registra decisões de forma legível para pessoas e parseável por máquina.
* **Não substitui o conceito:** Torna o registro mais previsível, reduzindo variação de estrutura entre autores.
* **A diferença central:** Está menos no conteúdo lógico da decisão e mais na **disciplina documental** aplicada ao arquivo.

### 2. Padronização e parseabilidade por máquina
* **Problema da variação:** Campos, nomes e seções diferentes a cada autor fragilizam busca, validação e automação.
* **A solução:** Campos recorrentes e estrutura fixa, interpretáveis por ferramentas sem leitura humana caso a caso.
* **O que habilita:** Documento processável em pipelines, CLIs, indexadores e fluxos assistidos por IA.

### 3. Metadados estruturados
* **Semântica explícita:** `status`, `date`, `tags`, `supersedes`, `amends` e `superseded by`.
* **Não são ornamento:** Permitem saber se a decisão está ativa, quando foi registrada, quais temas cobre e como se relaciona com outras.
* **Sem eles:** A timeline arquitetural existe, mas fica **implícita** e difícil de consultar automaticamente.

### 4. Relações entre ADRs
* **Vínculos explícitos:** O MADR materializa relações em vez de deixá-las no texto corrido.
* **Os campos:** `supersedes` (substitui outra), `superseded by` (é substituída), `amends` (complementa/ajusta parcialmente).
* **Resultado:** Transforma arquivos soltos em uma **rede navegável** de decisões.

### 5. Estrutura padronizada do conteúdo
* **Seções típicas:** `Context and Problem Statement`, `Decision Drivers`, `Considered Options`, `Decision Outcome`, `Pros and Cons of the Options`, `Consequences` e `References`.
* **Refina o clássico:** Separa melhor motivadores, opções avaliadas e efeitos da escolha.
* **O ganho:** Não é burocracia — é consistência para leitura rápida, comparação entre ADRs e extração automatizada.

### 6. Exemplo de substituição explícita de decisão
* **Cenário:** Uma ADR adota gRPC no lugar de REST e declara `supersedes ADR 02`.
* **Sem ambiguidade:** Informa que a decisão anterior deixou de ser a referência principal para aquele contexto.
* **Ciclo completo:** Se no futuro outra abordagem substituir o gRPC, o `superseded by` fecha o ciclo **sem apagar o histórico**.

### 7. Integração com ferramentas
* **Ecossistema:** ADR Tools, ADR Log, scripts CLI e extensões de editor.
* **O que um CLI faz:** Cria nova ADR com a próxima numeração, aplica template consistente e prepara os campos de relacionamento/rastreabilidade.
* **Dependência:** Essas ferramentas exigem **convenções estáveis** — sem elas, automatizar vira uma coleção de exceções.

### 8. Validação estrutural e automação em pipeline
* **Formato previsível = validável:** O repositório pode rodar lint e checagens estruturais em CI/CD.
* **Exemplo (GitHub Actions):** Verificar campos obrigatórios, nomenclatura correta e integridade dos links entre documentos.
* **Benefício:** Reduz **deriva documental** e mantém a consistência do acervo conforme ele cresce.

### 9. Uso por IA
* **Contexto explícito + estruturado:** O ganho aumenta quando o contexto também é estruturado, não só presente.
* **O que um conjunto de MADRs permite:** Identificar quais decisões substituíram outras, quais temas se repetem e quando certas escolhas foram feitas.
* **Efeito:** A IA opera sobre uma **memória arquitetural** com relações e metadados claros, em vez de inferir tudo do código.

### 10. Adaptação ao contexto da organização
* **Padrão, não ritual:** MADR é útil, mas não imutável.
* **Variações possíveis:** Templates, campos adicionais e papéis como `Informed` e `Consulted` — fazem sentido em alguns times, são excesso em outros.
* **Critério:** Preservar **consistência suficiente** para leitura, governança e automação, ajustando ao contexto real da empresa.

## Aula 4: Status, Metadados e Fluxos

Esta aula detalha o **ciclo de vida** de um ADR pelos seus **status** (`Draft`, `Proposal`, `Accepted`, `Rejected`, `Withdrawn`, `Deprecated`, `Superseded`) e pelos **metadados de relação** (`supersedes`/`superseded by`, `amends`, `relatesTo`, `dependsOn`). A regra de ouro: **status reflete o estado real** da decisão e **links refletem a relação real** entre documentos — usar o vínculo errado distorce a leitura histórica do acervo.

---

### 1. Proposal, Draft e o início formal da decisão
* **Não são sinônimos:** `Draft` é documento ainda em elaboração (pode circular de forma restrita); `Proposal` indica que a decisão já entrou em discussão/aprovação.
* **Diferença de governança:** Um rascunho pode existir **fora** do processo decisório; uma proposta já entra no **fluxo institucional** da arquitetura.

### 2. Accepted e a vigência da decisão
* **Referência oficial:** `Accepted` marca o ponto em que a decisão passa a valer.
* **Não é "implementado":** Significa que o direcionamento foi aprovado e deve orientar novas mudanças, não que tudo já foi construído.
* **Na timeline:** É o marco que torna a decisão **vigente** no acervo.

### 3. Rejected versus Withdrawn
* **Rejected:** A proposta foi analisada e **descartada** após avaliação — negativa deliberada.
* **Withdrawn:** A decisão foi **retirada de pauta** antes de seguir/ser aprovada, por mudança de contexto, prioridade ou interesse.
* **Por que distinguir:** Um registra discordância técnica; o outro registra **interrupção do processo**, não necessariamente discordância.

### 4. Deprecated como legado ainda válido
* **Não apaga nem invalida:** A decisão deprecada não invalida automaticamente o que já foi construído com ela.
* **O que comunica:** A escolha **não deve mais ser adotada** em novos contextos, mas segue relevante para interpretar legados.
* **No exemplo:** Pode deixar de ser recomendada hoje **sem exigir migração imediata** de tudo que a usou.

### 5. Superseded e substituição explícita
* **Substituição por outra:** `Superseded` indica que uma decisão anterior foi trocada por uma mais recente.
* **Pareamento de vínculos:** Vem com `superseded by` (aponta o sucessor); o novo documento declara `supersedes` (indica quem substituiu).
* **Benefício:** Evita ambiguidade sobre **qual ADR está vigente** e preserva a leitura histórica.

### 6. Metadados como estrutura da evolução histórica
* **Status × metadados:** O status diz a situação atual; os metadados dizem como o documento **se conecta** ao acervo.
* **Os campos-chave:** `supersedes`, `superseded by` e `amends` transformam ADRs isolados em sequência interpretável.
* **O que permitem:** Reconstruir a evolução arquitetural **sem depender da memória** do time.

### 7. Amends e alteração parcial
* **Quando usar:** A decisão antiga **continua existindo**, mas recebe um ajuste parcial.
* **Não é substituição:** Registra correção, extensão ou refinamento que altera só **parte** do conteúdo ou escopo.
* **Evita falsa ruptura:** Comunica **continuidade com modificação localizada**, não troca total.

### 8. RelatesTo versus DependsOn
* **relatesTo:** Relação técnica **sem dependência obrigatória** — ex: duas ADRs sobre gateways de pagamento que compartilham contexto, mas existem independentes.
* **dependsOn:** Mais forte — a decisão atual **só faz sentido/funciona** porque uma anterior já foi tomada.
* **A escolha do vínculo:** Define se há acoplamento real entre as decisões.

### 9. Fluxo de aprovação e governança do documento
* **Início:** `Draft`, nem sempre público; às vezes ocupa o espaço de uma RFC.
* **Submissão:** Ao time, tech lead ou diretoria, vira **proposta**; se aprovado, torna-se `Accepted`.
* **Depois:** A governança continua — permanecer vigente, virar `Deprecated` para novos usos ou ser `Superseded` por uma decisão posterior.

### 10. Regras de aplicação dos status e links
* **Reflitam a realidade:** Status = estado real da decisão; links = relação real entre documentos.
* **Deprecated × Superseded:** Não use `Deprecated` quando houve **substituição explícita** — aí `Superseded` com referência cruzada comunica melhor.
* **relatesTo × dependsOn:** Não use `relatesTo` para esconder dependência estrutural — se uma decisão **sustenta** a outra, o vínculo correto é `dependsOn`.

## Aula 5: Boas práticas e dicas importantes

Esta aula consolida o que mantém um acervo de ADRs **útil ao longo do tempo**: uma decisão por documento, escrita objetiva, nomenclatura rastreável, **governança por pull request** e **links bidirecionais** com HLD/FDD. Reforça a regra de **não reescrever o passado** (marcar, não apagar) e o alerta central da era da IA: **documentação errada pode ser pior que ausência**, porque o agente a consome como contexto válido e propaga o erro.

---

### 1. Uma ADR por decisão
* **Granularidade certa:** Registra **uma única** decisão arquitetural, não o sistema inteiro.
* **Evita:** Documentos longos, vagos e difíceis de manter — e torna a timeline mais precisa.
* **Decisão recorrente:** Consolidar em um ADR próprio e **referenciá-lo**, em vez de duplicar o raciocínio em vários lugares.

### 2. Escrita objetiva e técnica
* **Registro, não narrativa:** É o raciocínio que levou à escolha, não uma história extensa.
* **Contexto a serviço da decisão:** Indispensável, mas existe para **sustentar** a escolha, não para deixar o texto prolixo.
* **Quanto mais direto:** Mais fácil revisar, comparar alternativas e usar em busca, automação e IA.

### 3. Nomenclatura consistente e rastreabilidade
* **Identificadores estáveis:** ADR001, ADR002 e variações por domínio/módulo/componente — só funcionam com **convenção previsível**.
* **Objetivo operacional:** Localizar, ordenar, citar e relacionar decisões sem ambiguidade (não é estética).
* **Sem isso:** Se o time não acha o ADR rápido, o repositório vira um **diretório de arquivos**, não memória técnica.

### 4. Revisão e governança por pull request
* **Mesmo rigor do código:** Pull request, comentários, aprovação e histórico de mudanças.
* **O que gera:** Artefato governado, menos decisões mal formuladas e **responsabilização** sobre o que entra no acervo.
* **Mais crítico com IA:** Prompts, instruções e documentos alteram diretamente o comportamento do agente sobre o sistema.

### 5. Links bidirecionais entre ADR, HLD e FDD
* **Apontar nos dois sentidos:** O ADR aponta para os artefatos de requisito/desenho, e eles apontam **de volta** para a decisão.
* **O que cria:** Navegação confiável entre motivo, impacto estrutural e origem funcional da escolha.
* **Sem o encadeamento:** O leitor encontra peças soltas, mas **não reconstrói** o raciocínio completo.

### 6. Encadeamento documental e contexto de decisão
* **Decisão não nasce isolada:** Pode depender de um requisito em FDD, de uma restrição em HLD ou de outra decisão anterior aceita.
* **Encadear = explicitar:** Dependências e relações de leitura, formando uma **malha coerente** em vez de arquivos independentes.
* **Vale para humanos e IA:** Agentes inferem melhor quando as conexões **já estão declaradas**.

### 7. Não reescrever o passado
* **Histórico preservado:** Decisões antigas permanecem como foram tomadas, mesmo quando deixam de valer.
* **Ao mudar:** Criar um **novo ADR** e ajustar o status/relação do anterior — não reescrever o documento antigo.
* **Benefício:** Mantém a evolução **auditável** e não apaga o contexto que justificou decisões passadas.

### 8. ADR desatualizada deve ser marcada, não apagada
* **Sinalizar explicitamente:** Decisão que deixa de valer precisa ser marcada como substituída, não ficar ambígua.
* **Papel operacional do `superseded`:** Impedir que leitores e ferramentas tratem uma decisão antiga como atual.
* **Em acervos com IA:** Essa marcação reduz respostas incorretas baseadas em documentação obsoleta.

### 9. Documentação errada pode ser pior que ausência
* **Falsa confiança:** Documentação incompleta, contraditória ou mal conectada engana.
* **Humano × agente:** Pessoas podem desconfiar e investigar; a IA tende a usar o texto como **contexto válido** e propagar o erro em código, sugestões e análises.
* **Conclusão:** Qualidade documental é **requisito operacional**, não luxo editorial.

### 10. IA exige padronização e revisão disciplinada
* **Não elimina entendimento:** Gerar com IA não dispensa conhecer o tipo de documento nem a governança.
* **Sem prompt claro, convenção e revisão:** O resultado tende a ser prolixo, inconsistente e contraditório com o acervo.
* **Dependência direta:** A utilidade da IA depende da **qualidade estrutural** dos ADRs e dos vínculos com os demais documentos.
