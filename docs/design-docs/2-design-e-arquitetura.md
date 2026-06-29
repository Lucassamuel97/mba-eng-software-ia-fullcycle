# Módulo 2 - Design e Arquitetura

## Sumário

- [Aula 1: Documentos de Design e Arquitetura](#aula-1-documentos-de-design-e-arquitetura)

- [Aula 2: Documentos comuns](#aula-2-documentos-comuns)

- [Aula 3: High Level Design](#aula-3-high-level-design)

- [Aula 4: Exemplo de um High Level Design Document](#aula-4-exemplo-de-um-high-level-design-document)

- [Aula 5: Prompt para High Level Design](#aula-5-prompt-para-high-level-design)

- [Aula 6: Feature Design Doc](#aula-6-feature-design-doc)

- [Aula 7: Exemplo de um FDD](#aula-7-exemplo-de-um-fdd)

- [Aula 8: Prompt para FDD](#aula-8-prompt-para-fdd)

- [Aula 9: Realizando Deep Research](#aula-9-realizando-deep-research)

- [Aula 10: Adaptando Deep Research a um novo formato](#aula-10-adaptando-deep-research-a-um-novo-formato)

- [Aula 11: Gerando um FDD a partir da Deep Research](#aula-11-gerando-um-fdd-a-partir-da-deep-research)


## Aula 1: Documentos de Design e Arquitetura

Esta aula posiciona os **design docs** como o artefato que converte uma necessidade ainda difusa em uma **direção técnica compartilhada**. Eles reduzem ambiguidade ao iniciar projetos, implementar features ou evoluir produtos — mas só agregam valor quando o ganho de alinhamento supera o custo de produzir e manter o documento. A aula também reforça que a IA depende de **contexto confiável**: ela acelera a redação, mas não inventa o domínio.

---

### 1. Design docs como orientação de implementação
* **Transformam o difuso em direção:** Convertem uma necessidade ainda vaga em uma direção técnica compartilhada.
* **Reduzem ambiguidade:** Importam ao iniciar um projeto, implementar uma feature ou evoluir um produto existente.
* **Entendimento comum:** Em vez de cada pessoa assumir uma interpretação diferente, o documento explicita **o que** fazer e em **qual nível de detalhe**.
* **Crítico em arquitetura:** Decisões mal alinhadas geram retrabalho e inconsistência entre times.

### 2. Quando documentar e quando evitar excesso
* **Documentar tudo não é o objetivo:** Documentação útil não significa registrar tudo o tempo todo.
* **Depende do contexto:** Em alguns casos poucos documentos bastam; em outros, artefatos demais só adicionam redundância e custo de manutenção.
* **Variáveis de decisão:** Contexto organizacional, tipo de mudança e quantidade de pessoas e áreas envolvidas.
* **Critério:** Documentar faz sentido quando o **ganho de alinhamento supera o esforço** de produzir e manter.

### 3. Níveis de abstração e adequação ao público
* **Públicos diferentes, respostas diferentes:** Documentos existem em níveis distintos de abstração.
* **Alto nível × operacional:** Áreas estratégicas precisam de visão ampla; quem implementa precisa de decisões concretas.
* **Empresas maiores:** O desdobramento é mais forte — mais pontos de decisão, camadas organizacionais e necessidade de coordenação.
* **Qualidade real:** Depende menos do formato e mais da **adequação entre profundidade, audiência e objetivo**.

### 4. Documentação como ponte entre negócio e técnica
* **A necessidade chega fragmentada:** Raramente nasce pronta para implementação — vem implícita e distribuída entre pessoas.
* **O design doc é a ponte:** Converte esse contexto em uma decisão técnica compartilhável por produto, engenharia e liderança.
* **Mais que registrar solução:** Consolida **premissas, restrições e direção**.
* **Evita suposições isoladas:** Impede que a arquitetura seja definida pelo entendimento parcial de cada participante.

### 5. IA depende de contexto confiável
* **Não inventa o domínio:** A IA não recria corretamente o contexto específico da empresa, do domínio e da feature.
* **Apoia, mas depende de insumos:** Mesmo ajudando a estruturar texto e acelerar a redação, segue dependente de informações reais sobre necessidades e decisões.
* **Limitação principal:** Não é a capacidade de escrever, mas a **ausência de contexto confiável** para orientar o conteúdo.
* **Risco:** Sem contexto, o resultado pode parecer plausível e ainda estar desalinhado com o problema real.

### 6. Estrutura, prompts e entrevistas como apoio
* **Saem da página em branco:** Templates, prompts e entrevistas reduzem o esforço inicial e organizam a coleta de informação.
* **Não substituem o domínio:** Ajudam a transformar conhecimento implícito em perguntas, seções e decisões registráveis.
* **Orientam stakeholders:** Bem usados, tornam a documentação mais consistente entre projetos.
* **Onde está o ganho:** Na **estruturação do processo**, não na promessa de geração automática perfeita.

### 7. Documentação como ativo cumulativo reutilizável
* **Além do momento inicial:** Depois de produzidos, os documentos compõem o **contexto reutilizável** do ciclo de desenvolvimento.
* **Sustentam o futuro:** Ajudam novas pessoas a entender o que se constrói e embasam revisões, extensões e refinamentos arquiteturais.
* **A IA se beneficia:** Com documentos anteriores como contexto, ela apoia melhor as tarefas posteriores.
* **Ativo cumulativo:** Preserva raciocínio e **reduz perda de contexto** ao longo do tempo.

## Aula 2: Documentos comuns

Esta aula mapeia os principais artefatos de design e arquitetura como uma **cadeia de abstração** — do problema de produto até o registro das decisões. Cada documento (PRD, HLD, Feature Design Doc/FRD, LLD, RFC, ADR) responde a uma **pergunta diferente** e atua em um momento distinto. A chave não é produzir todos, mas escolher os adequados conforme porte da mudança, risco técnico e necessidade de alinhamento.

---

### 1. Encadeamento entre documentos
* **Cadeia de abstração:** Vai do problema de produto até a implementação e o registro das decisões.
* **Cada artefato, uma pergunta:** PRD descreve o problema e o valor; HLD organiza a solução em alto nível; Feature Design Doc detalha uma feature; LLD aproxima do código; RFC delibera alternativas; ADR registra a decisão tomada.
* **Não é obrigatório usar todos:** A adequação depende do porte da mudança, do risco técnico e da necessidade de alinhamento entre áreas.

### 2. HLD (High Level Design)
* **O que é:** Descreve como o sistema é estruturado sem entrar em detalhes finos de implementação.
* **Para que serve:** Transforma contexto de produto em visão técnica compartilhada — componentes, responsabilidades, integrações e limites.
* **Momento:** Costuma vir **depois do PRD**, quando o problema já está claro e é preciso discutir a forma geral da solução.
* **Mudança pequena:** Esse nível pode ser condensado ou absorvido por um documento mais específico.

### 3. Feature Design Doc e FRD
* **O que é:** Especifica como uma feature ou módulo será implementado — foco maior que o HLD e menor que o LLD.
* **O que responde:** Escopo técnico da feature, fluxo principal, impactos no sistema e escolhas necessárias para entregá-la.
* **FRD:** Nomenclatura legada em muitas organizações, próxima desse papel, com herança de uma abordagem centrada em requisitos funcionais.
* **O nome importa menos:** A função é tornar explícito **como** uma parte específica do sistema será construída.

### 4. Quando o Feature Design Doc pode bastar
* **Feature pequena ou isolada:** Nem sempre exige um HLD completo antes do detalhamento.
* **Concentra a decisão:** O próprio Feature Design Doc pode bastar para alinhar implementação, dependências e impacto técnico.
* **Quando subir um nível:** Se a mudança afeta múltiplos módulos, envolve arquitetura mais ampla ou exige coordenação entre times, vale produzir antes um artefato mais high level.
* **Escolha contextual:** A decisão do documento é situacional, **não ritualística**.

### 5. LLD (Low Level Design)
* **O que é:** Aproxima o design da implementação concreta.
* **O que detalha:** Endpoints, contratos de comunicação, campos de API, patterns adotados e outras decisões que orientam diretamente o código.
* **Diferença para o Feature Design Doc:** É o grau de precisão — o de feature pode dizer que *haverá* endpoints; o LLD define **quais** existirão e como seus contratos serão estruturados.
* **Quando usar:** Quando a equipe precisa reduzir ambiguidade técnica antes de implementar.

### 6. RFC (Request for Comments)
* **O que é:** Documento de **deliberação** técnica.
* **Como funciona:** Circula uma proposta para receber comentários, objeções e alternativas antes da decisão final.
* **Onde é comum:** Projetos open source e mudanças de impacto relevante, como novas features ou breaking changes.
* **Valor:** Expõe o raciocínio enquanto a decisão ainda está aberta, permitindo revisão coletiva. **Não registra a decisão definitiva** — organiza a discussão que a antecede.

### 7. ADR (Architecture Decision Record)
* **O que é:** Registra uma decisão arquitetural **já tomada**, com justificativa e contexto.
* **Diferente da RFC:** A RFC é deliberativa; o ADR é a **memória técnica** — responde por que uma tecnologia, stack, banco ou abordagem foi escolhida e em que condições isso faz sentido.
* **Ciclo de vida:** Pode ficar ativo, ser substituído ou tornar-se inativo quando a decisão deixa de valer.
* **Benefício:** Evita que o time **redescubra** continuamente o motivo das escolhas anteriores.

### 8. RFC e ADR na linha do tempo
* **Diferença temporal e funcional:** A RFC vem **antes** (ainda há debate); o ADR vem **depois** (registra a decisão vencedora).
* **Fluxo maduro:** A RFC captura argumentos e contrapontos; o ADR consolida a decisão de forma clara e sucinta.
* **Nem sempre os dois:** Nem toda decisão exige ambos, mas **confundir discussão com registro** costuma gerar perda de contexto.

### 9. Tabela comparativa

| Documento | Pergunta principal | Nível de detalhe | Momento de uso |
|---|---|---|---|
| PRD | Qual problema de produto precisa ser resolvido? | Baixo detalhe técnico | Antes do desenho técnico |
| HLD | Como a solução se organiza em alto nível? | Alto nível técnico | Após clareza de produto e antes do detalhamento da feature |
| Feature Design Doc / FRD | Como uma feature ou módulo será implementado? | Detalhe intermediário | Quando o escopo da feature já está definido |
| LLD | Como a implementação será estruturada concretamente? | Alto detalhe técnico | Próximo da codificação |
| RFC | Quais alternativas estão em discussão antes da decisão? | Variável, orientado a debate | Antes da decisão técnica |
| ADR | Qual decisão arquitetural foi tomada e por quê? | Registro objetivo da decisão | Depois da decisão técnica |

### 10. Aplicação no mesmo problema ao longo da cadeia
* **PRD:** Define o objetivo e as restrições do recurso.
* **HLD:** Transforma isso em visão estrutural da solução.
* **Feature Design Doc:** Desce para a implementação daquela feature específica.
* **LLD:** Fixa contratos, endpoints e detalhes executáveis.
* **RFC → ADR:** Se houver dúvida relevante entre alternativas, a discussão passa por uma RFC; quando a escolha é feita, registra-se em ADR.
* **Valor do ecossistema:** Está no **encadeamento** — cada artefato responde uma pergunta diferente sem competir com os demais.

## Aula 3: High Level Design

Esta aula detalha o **HLD (High Level Design)** como o desenho da solução no nível em que a **arquitetura faz sentido como sistema, não como código**. Ele responde perguntas estruturais — componentes, comunicação, tecnologias, padrões — e funciona como o "terreno arquitetural onde as features precisam caber". Seu valor está em **remover ambiguidade estrutural** sem virar especificação detalhada nem ficar genérico demais.

---

### 1. High Level Design como desenho arquitetural
* **Nível de sistema, não de código:** Organiza a solução onde a arquitetura precisa fazer sentido como um todo.
* **O que aborda:** Estrutura, componentes, módulos, comunicação, tecnologias e padrões principais.
* **Utilidade prática:** Alinhar a visão técnica ampla entre quem decide, revisa ou decompõe o trabalho.
* **Metáfora:** É o **terreno arquitetural** onde as features precisam caber.

### 2. Perguntas que o HLD precisa responder
* **Não é linha por linha:** Não diz como construir em detalhe, e sim **como o sistema é organizado** e como as partes se conectam.
* **O que inclui:** Identificar componentes relevantes, descrever os fluxos principais e explicitar como os módulos se comunicam.
* **Tecnologias e padrões:** Entram porque moldam restrições e possibilidades para os documentos mais detalhados.
* **Foco:** Menos exaustão, mais **remoção de ambiguidade estrutural**.

### 3. Uso de C4 no nível de containers
* **Representação adequada:** O diagrama C4 no nível de **containers** mostra os principais serviços, APIs e relações entre blocos executáveis.
* **O que não detalha:** Classes nem contratos completos — mas já revela responsabilidades, fronteiras e fluxos de comunicação.
* **Ganho:** Transforma uma descrição abstrata em arquitetura legível, suficiente para discutir composição e dependências.
* **Ponto de virada:** A solução deixa de ser intenção e passa a ter **forma arquitetural**.

### 4. Seções típicas do documento
* **Objetivo com recorte técnico:** Costuma abrir pelo objetivo do documento/feature, que já vem do PRD e aqui ganha foco técnico.
* **Seções centrais:** Arquitetura geral, principais componentes, responsabilidades, fluxo das requisições, modelo de dados em alto nível e interfaces públicas.
* **Para que servem:** Responder **o que** construir do ponto de vista estrutural, sem cair em especificação de implementação.
* **Nomes variam:** O importante é cobrir as **perguntas arquiteturais certas**.

### 5. Interfaces públicas em alto nível
* **O que cada parte expõe:** Deixam claro o que cada módulo ou serviço oferece ao restante do sistema, sem detalhar todos os contratos.
* **Por que importam:** A arquitetura depende das **fronteiras** entre partes — e elas aparecem nas interfaces.
* **Em alto nível basta:** Identificar quais pontos de integração existem e qual papel cumprem.
* **Detalhe fino depois:** Fica para documentos mais próximos da implementação.

### 6. Preocupações transversais: segurança, escalabilidade, disponibilidade e observabilidade
* **Afetam desde o início:** Influenciam a arquitetura, não apenas a fase de implementação.
* **Como moldam a solução:** Crescer, resistir a falhas, proteger acesso ou ser monitorável muda componentes, fluxos e padrões de comunicação.
* **Registrar como direcionadores:** Não exigem mecanismos e configurações completos ainda, mas precisam constar como **direcionadores arquiteturais**.
* **Risco de ignorar:** Empurra problemas estruturais para tarde demais.

### 7. Riscos e leitura do documento
* **Conectados à forma da solução:** No HLD, os riscos aparecem ligados à arquitetura proposta.
* **Para quem serve:** Ajuda líderes e arquitetos a entender o que será desenvolvido, com detalhe suficiente para orientar decisões e a decomposição em um Feature Design Doc.
* **Não é executável:** Não entrega instruções diretas de implementação, mas dá **base** para refinar a solução em nível mais baixo.
* **Limite intencional:** Clareza arquitetural sem confundir visão geral com especificação detalhada.

### 8. Fronteira entre HLD e documentos mais detalhados
* **Não substitui o de feature:** Quando a solução exige detalhamento operacional, o HLD não basta.
* **Exemplo:** Dizer apenas que "haverá autenticação" pode bastar em cenário simples (framework resolve quase tudo); em cenário complexo, o HLD sinaliza a preocupação, mas o detalhe **desce para outro artefato**.
* **Fronteira contextual:** Depende da complexidade e do contexto da mudança.
* **Erro comum:** Tentar transformar o HLD em especificação completa — ou, no extremo oposto, deixá-lo **genérico demais** para orientar qualquer decisão.

### 9. Adaptação ao contexto
* **Sem conjunto universal de seções:** Não existe template obrigatório para todo HLD.
* **Guideline adaptável:** Muitos artefatos compartilham uma espinha dorsal parecida, mas cada contexto pede seções extras, ênfases ou nomenclaturas diferentes.
* **Flexibilidade é parte do papel:** A arquitetura precisa refletir o **problema real**, não obedecer rigidamente a um modelo.
* **Bom documento:** Preserva a cobertura das **decisões essenciais** e ajusta a forma ao contexto do sistema.

## Aula 4: Exemplo de um High Level Design Document

Esta aula aterrissa o HLD em um **exemplo concreto** — o mesmo **rate limiter** já visto como PRD, agora no nível arquitetural. O documento fixa objetivo, metas não funcionais (P95 < 5 ms), topologia (SDK in-process + Redis), estratégias, fluxo, modelo de chaves, segurança, observabilidade e riscos — tudo no nível que **orienta a solução sem virar Low Level Design**.

> 📄 **Exemplo de referência:** o HLD completo está em [HLD — Rate Limiter](/docs/design-docs/templates-design-arquitetura/ex_HLD_Rate_Limiter.md). Compare com o [PRD de Feature — Rate Limiter](/docs/design-docs/templates-prd/ex_PRD_Feature_Rate_Limiter.md) para ver a mesma feature em níveis de abstração diferentes.

---

### 1. Objetivo arquitetural do rate limiter
* **Problema conhecido, desenho nem tanto:** É um caso recorrente porque o conceito é familiar, mas a implementação e o desenho arquitetural nem sempre.
* **O que o HLD fixa:** Um SDK embutido nos microserviços, escrito em Go, que limita acesso por **API key, IP e plano do cliente**.
* **Papel do documento:** Transformar uma necessidade operacional em **responsabilidades, metas e fronteiras técnicas** sem descer ao código.

### 2. Metas não funcionais e P95 abaixo de 5 ms
* **Moldam a arquitetura:** As metas não funcionais vêm antes de qualquer detalhe de implementação.
* **P95 < 5 ms (Redis):** 95% das verificações precisam terminar abaixo desse tempo, restringindo topologia, armazenamento e número de hops.
* **Por que tão rígido:** O rate limiter fica no **caminho crítico** da chamada HTTP — se for lento, toda a plataforma herda essa latência.

### 3. SDK in-process como middleware HTTP
* **Integração in-process:** Funciona como middleware HTTP antes da lógica de negócio do serviço hospedeiro.
* **Vantagem:** Reduz acoplamento com um serviço remoto dedicado e evita uma chamada de rede extra por requisição.
* **Efeito prático:** A requisição entra → o middleware consulta identidade e política → decide permitir/negar → injeta headers → a lógica de negócio continua.

### 4. Redis como estado compartilhado
* **Por que Redis:** O limitador precisa de leitura e atualização rápidas, com compartilhamento entre múltiplas instâncias.
* **Processo stateless:** O microserviço fica sem estado; contadores e janelas ficam em Redis (ou memória local em cenários específicos).
* **Ganho:** Escalar horizontalmente os serviços sem perder a consistência básica do controle distribuído.

### 5. Estratégias de limitação no nível do HLD
* **Direção, não algoritmo:** Fixed window e token bucket aparecem para registrar a direção arquitetural, não para detalhar o algoritmo.
* **Perfis distintos:** Fixed window serve a limites por janela temporal fixa; token bucket lida melhor com rajadas e recomposição gradual.
* **O que o HLD explicita:** Estratégias diferentes atendem perfis diferentes de tráfego e influenciam a modelagem de estado e chaves.

### 6. Topologia e componentes principais
* **No nível necessário:** Componentes e interfaces aparecem apenas o suficiente para orientar a solução.
* **Desenho:** Microserviço hospedeiro, SDK de rate limiting, Redis, telemetria e, em produção, eventual orquestração com Kubernetes.
* **Coração do SDK:** Um **check síncrono** que recebe identidade, rota, método e plano, resolve a política e devolve decisão com metadados (`Retry-After`, headers de limite).

### 7. Fluxo principal da requisição
* **Entrada pelo middleware:** A requisição atinge o serviço e passa primeiro pelo SDK.
* **Sequência:** compõe identidade (API key, tenant, IP, rota, método) → resolve política → consulta estado (Redis/memória) → decide.
* **Saídas:** Limite excedido → `429 Too Many Requests`; aceito → segue, e a telemetria registra métricas, logs e tracing.

### 8. PII na composição das chaves
* **Risco no chaveamento:** Identificadores como IP e dados de cliente podem vazar informação sensível se gravados de forma ingênua.
* **O que o HLD registra:** A **exigência** de não expor dados identificáveis nas estruturas de estado e observabilidade — sem definir a implementação exata.
* **Por que importa:** A chave técnica usada para contagem também é um ponto potencial de **risco regulatório e operacional**.

### 9. Hot keys em Redis
* **O que são:** Muitas requisições concentrando leitura/escrita na mesma chave, gerando contenção e degradando latência.
* **Quando surgem:** Clientes compartilhando um escopo global ou chaveamento que concentra tráfego em poucos identificadores.
* **Por que registrar:** Afeta diretamente a meta de **P95** e pode exigir particionamento, melhor granularidade de chave ou revisão de política.

### 10. Fallback open como decisão arquitetural
* **O que é:** Liberar a requisição quando o mecanismo de limitação falha (ex: Redis indisponível).
* **Decisão, não detalhe:** Troca rigor de proteção por **continuidade de acesso** em situações degradadas.
* **Trade-off explícito:** Reduz risco de indisponibilidade para usuários legítimos, mas aceita exposição temporária a sobrecarga.

### 11. Observabilidade e riscos no nível certo
* **Já esperados no HLD:** Segurança, observabilidade e riscos arquiteturais; o avanço é vê-los **concretizados** no exemplo.
* **Observabilidade:** Métricas, logs, tracing e integração com Prometheus e OpenTelemetry para acompanhar latência, falhas e decisões.
* **Riscos principais:** Redis indisponível ou particionado, configuração incorreta bloqueando tráfego legítimo, hot keys e exposição de PII.

### 12. Limite entre arquitetura e implementação
* **Interfaces, não contratos finais:** Mostra `check`, contexto, identidade, escopo, decisão, `next`, `render` — sem detalhar contratos completos, campos finais ou código.
* **Fronteira correta:** O leitor entende **como a solução se organiza**, quais decisões foram tomadas e onde estão os riscos, sem confundir HLD com Low Level Design.
* **Próximo passo:** Quando for preciso definir estruturas exatas, contratos e regras operacionais, segue-se para um documento de **nível mais baixo**.

## Aula 5: Prompt para High Level Design

Esta aula mostra como um **prompt de entrevista guiada** acelera a produção do HLD: em vez de pedir texto livre, ele coleta contexto por perguntas estruturadas, usa PRD e documentos como insumo, gera um draft em um **template reutilizável** e roda checagem de consistência. A IA é **aceleradora, não autora única** — o raciocínio arquitetural continua humano.

> 📄 **Exemplo de referência:** o prompt completo está em [Prompt para geração de um HLD](/docs/design-docs/templates-design-arquitetura/ex_prompt_gerar_HLD.md), com papel, princípios de entrevista, processo em 11 etapas, estrutura JSON, defaults e o esqueleto de saída. Veja o resultado aplicado no [HLD — Rate Limiter](/docs/design-docs/templates-design-arquitetura/ex_HLD_Rate_Limiter.md).

---

### 1. Entrevista guiada para gerar um HLD
* **Conduz a coleta:** Em vez de pedir texto livre, faz perguntas estruturadas e transforma as respostas em um draft inicial.
* **Cobre o esquecido:** Força seções que costumam ser puladas — riscos, observabilidade e interfaces públicas.
* **Onde está o valor:** Não em automatizar a arquitetura inteira, mas em **organizar a extração de contexto** técnico de forma repetível.

### 2. PRD e documentos complementares como insumo
* **PRD como ponto de partida:** Evita que o HLD nasça só de respostas improvisadas durante a entrevista.
* **Mais contexto, menos preenchimento:** Documentos técnicos, anotações de reunião e pesquisa prévia aproximam o draft da realidade do problema.
* **O que pesa:** A qualidade depende menos da eloquência do prompt e mais da **densidade do contexto anexado**.

### 3. Template reutilizável
* **Fixa o formato antes da geração:** Contexto, arquitetura geral, componentes, fluxo de requisição, modelo de dados, interfaces públicas, escalabilidade, segurança, observabilidade e riscos.
* **Evita inconsistência:** Impede texto solto ou reorganização do HLD a cada execução.
* **Facilita o resto:** Revisão humana, comparação entre documentos e adaptação ao padrão interno do time.

### 4. Defaults inteligentes no prompt
* **O que são:** Valores ou decisões assumidas quando a informação ainda não foi fornecida, de forma controlada.
* **Para que servem:** Reduzir fricção na geração inicial e evitar que o processo pare por falta de detalhes menores.
* **Exigem revisão:** Aceleram o draft, mas uma suposição plausível ainda pode estar **errada** para a empresa ou a feature.

### 5. JSON como formato operacional
* **Estrutura manipulável:** Transforma seções do HLD em algo previsível e processável por ferramentas.
* **O que habilita:** Validar campos, reaproveitar blocos, versionar estruturas e alimentar outros fluxos automatizados.
* **Objetivo:** Não substituir a leitura humana, mas tornar a geração mais **consistente e integrável**.

### 6. Checagem de consistência entre seções
* **Verifica contradições:** Confere se as partes do HLD não se contradizem antes de finalizar o draft.
* **Exemplo:** Se a arquitetura sugere um componente centralizado, o fluxo principal, os riscos e a escalabilidade precisam refletir essa escolha.
* **Evita:** Documentos formalmente bonitos, mas **internamente incoerentes**.

### 7. IA como aceleradora, não autora única
* **No que acelera:** Estruturar perguntas, preencher o esqueleto e expandir partes técnicas recorrentes.
* **Limite:** Não gera sozinha um documento confiável para contextos específicos de empresa, domínio e funcionalidade.
* **Papel humano:** Decidir, corrigir, complementar e **remover generalizações inadequadas**.

### 8. Adaptação ao fluxo de trabalho do time
* **Não é peça fixa:** O prompt muda conforme o workflow, o tipo de projeto e a documentação já existente.
* **Pontos de partida diferentes:** Em alguns casos a entrevista inicia o processo; em outros, começa-se com PRD, pesquisa técnica e um template parcialmente preenchido.
* **Ganho:** Adaptar ao processo real reduz atrito e aumenta a utilidade do draft.

### 9. Aplicação prática no exemplo de rate limiting
* **Não redesenha a solução:** Reconstrói o HLD de rate limiting a partir de insumos, perguntas e um esqueleto.
* **O que o prompt pede:** Objetivos técnicos, componentes, fluxo principal, riscos e preocupações transversais, usando o documento como referência estrutural.
* **Mudança de modo:** O time passa da escrita manual integral para a **geração assistida** de um primeiro rascunho revisável.

### 10. Workflow operacional de geração
* **Fluxo recomendado:** reunir insumos → executar a entrevista guiada → gerar o draft no template → rodar checagem de consistência → revisar manualmente.
* **Equilíbrio:** Reduz a fricção de começar do zero **sem terceirizar** o raciocínio arquitetural.
* **Resultado esperado:** Não um documento final perfeito, mas um **draft inicial** mais mastigado e mais barato de evoluir.

## Aula 6: Feature Design Doc

Esta aula posiciona o **Feature Design Doc (FDD)** como o documento que desce do desenho arquitetural (HLD) para a **especificação operacional** da feature: comportamento em runtime, contratos reais, erros, fallbacks, configuração e critérios de aceite. Ele é a **ponte entre arquitetura e código** — detalha o suficiente para orientar a implementação sem virar prescrição de código linha por linha.

---

### 1. Posição do FDD na hierarquia
* **Desce do HLD:** Vai do desenho arquitetural para a especificação operacional da feature.
* **Transforma contexto em detalhe:** O HLD definiu o terreno; o FDD vira comportamento detalhado, contratos reais e condições de implementação.
* **Nível intermediário:** Não prescreve código linha por linha, mas também não fica só na organização macro do sistema.
* **Papel central:** Reduzir a ambiguidade entre **arquitetura e execução**.

### 2. Quando usar um FDD
* **Gatilhos:** A feature expõe API, altera contratos, adiciona configuração ou afeta segurança, performance e compatibilidade.
* **Por que aqui:** A implementação depende de definições que não cabem mais no nível arquitetural, mas precisam ser compartilhadas antes do código.
* **Deixa de ser opcional:** Quando a mudança afeta integração entre partes, comportamento em runtime ou expectativas externas, vira **instrumento de alinhamento técnico**.

### 3. Perguntas que o documento responde
* **Comportamento e contratos:** Como a feature se comporta em runtime, quais interfaces expõe, como é configurada e de quais dependências precisa.
* **Erros e concorrência:** Como lida com erros, exceções e concorrência.
* **Verificabilidade:** Define como **validar** que a implementação está correta — desloca a discussão de intenção para verificabilidade.
* **Utilidade ampla:** Serve para construir, revisar, testar e aceitar a feature com **critérios objetivos**.

### 4. FDD como ponte entre arquitetura e código
* **Fronteira deliberada:** Detalha o suficiente para orientar a implementação sem virar padrão de codificação ou detalhe interno de classe.
* **Onde moram os erros:** Muitos surgem no espaço entre "a arquitetura permite" e "o código realmente faz".
* **O que fecha esse espaço:** Explicitar contratos públicos, fluxos, erros e regras operacionais que o HLD apenas sinaliza em alto nível.

### 5. Contratos públicos e comportamento detalhado
* **O que outros podem esperar:** Define o que outras partes do sistema ou consumidores externos esperam da feature.
* **O que entra:** Assinaturas, endpoints, headers, exemplos de uso e semântica de resposta — integração exige **precisão, não intenção genérica**.
* **No rate limiter:** O foco deixa de ser a topologia (Redis + middleware) e passa ao comportamento exato — quais headers retorna, em que condição responde `429` e qual contrato expõe.

### 6. Erros, exceções e fallbacks
* **Comportamento especificado:** Erros e exceções não são detalhe deixado para o implementador decidir no meio do desenvolvimento.
* **Fallbacks na mesma camada:** Representam a resposta esperada quando dependências falham ou o modo principal não opera.
* **No rate limiter:** A decisão arquitetural de **fallback open** vira regra operacional clara, com condições de acionamento e efeito observável.

### 7. Configuração, dependências e compatibilidade
* **Configuração no uso real:** Quais opções existem, como são fornecidas via código e quais combinações são válidas ou inválidas.
* **Dependências concretas:** Deixam de ser componentes do desenho e viram requisitos para a feature funcionar, com impacto em integração e rollout.
* **Compatibilidade explícita:** Tratada quando a mudança altera contratos, comportamento esperado ou a forma de adoção por consumidores existentes.

### 8. Critérios de aceite como núcleo da especificação
* **Tornam o doc verificável:** Transformam o FDD em referência testável, não apenas descritiva.
* **O que definem:** As condições sob as quais a feature é considerada correta — comportamento, contratos, erros e restrições técnicas.
* **Diferença prática:** Sem eles, o time implementa algo plausível; com eles, implementa algo **testável contra uma definição compartilhada**.

### 9. Estrutura típica do documento
* **Seções comuns:** Contexto e motivação técnica, objetivos, escopo e exclusões, fluxos detalhados, diagramas (quando necessários), contratos públicos, erros, fallbacks, dependências, compatibilidade, critérios de aceite, riscos e mitigação.
* **Foco no que diverge:** Não serve só para descrever a feature, mas para **cercar os pontos que mais geram divergência** na implementação.
* **Reaproveitamento:** Observabilidade e riscos podem ser referenciados a partir do que já foi estabelecido antes, desde que fique claro como afetam a operação específica da feature.

### 10. Implicações para implementação com IA
* **Contexto operacional reutilizável:** Um FDD bem escrito melhora a implementação assistida por IA em vez de depender de prompts vagos.
* **Menos interpretação arbitrária:** Com contratos, erros, configuração e critérios de aceite explícitos, a IA gera código e testes com menos ambiguidade.
* **Ganho principal:** Não é "automatizar a feature", mas **reduzir a distância** entre o que foi decidido e o que será implementado.

## Aula 7: Exemplo de um FDD

Esta aula percorre um **FDD concreto** — o mesmo **rate limiter**, agora no nível de comportamento verificável. O documento sai da topologia geral e fixa **como cada microsserviço aplica limites**: API pública (`Check`/`Middleware`/`Decision`), semântica de headers, estratégias em runtime, configuração via `options`, atomicidade (Lua no Redis, mutex por chave na memória), fallback open, observabilidade e critérios de aceite. Mostra também que a **pesquisa técnica** é o que abastece a IA com os pontos que mais afetam a implementação.

> 📄 **Exemplo de referência:** o FDD completo está em [FDD — Rate Limiter](/docs/design-docs/templates-design-arquitetura/ex_FDD_Rate_Limiter.md). Compare com o [HLD — Rate Limiter](/docs/design-docs/templates-design-arquitetura/ex_HLD_Rate_Limiter.md) e o [PRD de Feature — Rate Limiter](/docs/design-docs/templates-prd/ex_PRD_Feature_Rate_Limiter.md) para ver a mesma feature do produto à implementação.

---

### 1. Contexto técnico do FDD
* **Desce para o verificável:** Sai do desenho arquitetural e foca o comportamento.
* **Novo foco:** Padronizar como cada microsserviço aplica limites, expõe contratos e reage a falhas — não a topologia geral.
* **Por que existe:** Inconsistência entre serviços produz sobrecarga, integração confusa e implementação divergente.
* **Solução:** Um SDK embutido in-process em cada serviço, com estado em Redis ou memória local.

### 2. Objetivos técnicos e escopo
* **Intenção vira requisito:** API pública estável, integração por middleware HTTP, fixed window e token bucket, telemetria nativa, baixa latência e fallback open.
* **Escopo delimita a entrega:** Inclui Redis Cluster, memória local e geração de headers; exclui autenticação, gestão de políticas e estratégias alternativas.
* **Exclusão é parte do contrato:** Em FDD, o que fica **de fora** é tão importante quanto o incluído — evita absorver responsabilidades de outros componentes.

### 3. API `Check`, `Middleware` e `Decision`
* **Contratos concretos:** A API expõe operações reais, não a ideia genérica de "aplicar rate limiting".
* **`Check`:** Decisão síncrona — recebe o contexto da requisição e retorna uma `Decision` com campos observáveis (permitido/bloqueado, restante, `retry_after`).
* **`Middleware`:** Encapsula o processo no pipeline HTTP, convertendo a decisão em continuação ou `429`.
* **`Decision`:** Separa cálculo interno de efeito externo — facilita teste, reuso e consistência entre integração programática e HTTP.

### 4. Semântica de headers de rate limiting
* **Significado explícito:** `RateLimit`, `RateLimit-Reset` e `Retry-After` precisam de semântica clara — clientes os usam para backoff e exibição de limites.
* **O que o FDD fixa:** Quais headers existem e o **significado operacional** de cada valor (quanto resta, quando reinicia, quanto esperar).
* **Evita incompatibilidade:** Impede que cada serviço publique convenções divergentes.
* **Header é API pública:** Em feature de infraestrutura, o contrato de header faz parte da interface.

### 5. Estratégias: fixed window e token bucket
* **Não reabre a decisão:** O FDD especifica como as estratégias se comportam na interface e no runtime, não a escolha arquitetural.
* **Fixed window:** Contagem agrupada por janela discreta; reset acompanha o limite temporal.
* **Token bucket:** Recarga contínua por `rate` até um teto de `burst`, absorvendo rajadas controladas.
* **Utilidade:** Tornar as diferenças **implementáveis e testáveis**, inclusive nos headers e exemplos.

### 6. Padrão `options` para configuração
* **Configuração incremental:** Evita construtores com muitos parâmetros posicionais.
* **Opções composáveis:** Storage, estratégia, credenciais, pool e comportamento operacional.
* **Builder idiomático:** Cada opção altera a configuração final de forma explícita e extensível.
* **Para múltiplos modos:** Melhora legibilidade, compatibilidade futura e ergonomia de uso.

### 7. Validação no construtor
* **Falhar cedo:** Impede que uma instância inválida exista em runtime.
* **O que checar:** Coerência dos parâmetros da estratégia, combinações obrigatórias e configuração suficiente para o modo selecionado.
* **Erro mais barato:** Desloca falha de produção para falha de inicialização.
* **Em infraestrutura:** Falhar na criação é preferível a aceitar configuração inconsistente e gerar comportamento imprevisível.

### 8. Modos de storage: Redis e memória local
* **Não são só intercambiáveis:** Cada modo define um comportamento operacional distinto.
* **Redis:** Cenário distribuído — múltiplas instâncias compartilham estado e aplicam o mesmo limite global.
* **Memória local:** Desenvolvimento ou instâncias isoladas, com **perda de estado** ao reiniciar pod/container.
* **Por que registrar:** Evitar adoção incorreta do modo local em cenários que exigem coordenação distribuída.

### 9. Atomicidade em Redis com scripts Lua
* **Problema central:** Múltiplas requisições disputam o mesmo contador simultaneamente.
* **Solução Lua:** Executa leitura, cálculo e atualização como **operação atômica** no servidor, evitando condições de corrida.
* **Sem round-trips frágeis:** Protege a correção do limite e preserva latência no caminho síncrono.

### 10. Concorrência em memória com mutex por chave
* **Sincronização local:** No modo memória, a atomicidade depende de sincronização dentro do processo.
* **Mutex por chave:** Evita atualização concorrente do mesmo identificador sem serializar chaves independentes.
* **Menos contenção:** Reduz disputa frente a um lock global e preserva paralelismo seguro.
* **Decisão obrigatória:** Concorrência correta não é detalhe opcional em um limitador.

### 11. Compatibilidade e dependências
* **Piso técnico:** Go 1.22 e Redis 6.2 delimitam a base e evitam dependência acidental de recursos mais novos.
* **Dependências verificáveis:** Prometheus, OpenTelemetry, Collector, Linux e AMD64/ARM viram requisitos de build, execução e observação.
* **Por que importa:** Uma feature pode estar correta em código e falhar por **incompatibilidade de ambiente**.
* **Em FDD:** Dependência é condição operacional, não nota de rodapé.

### 12. Segurança e proteção de dados
* **Concreta, não abstrata:** Logs, métricas, tracing e chaves não podem vazar identificadores sensíveis (ex: IP em texto cru).
* **O que indicar:** TLS entre app e Redis quando disponível, tratamento seguro de credenciais e cuidado com atributos/spans exportados.
* **Restrição operacional:** Segurança limita o que pode circular em observabilidade e infraestrutura.
* **Objetivo:** Impedir que um componente de controle introduza **exposição indevida** de dados.

### 13. Fallback open e comportamento em falha
* **Já decidido, agora especificado:** O avanço é definir **quando dispara** e qual efeito observável produz.
* **Comportamento em falha:** Com modo permissivo ativo, a requisição segue, mas o sistema registra logs, métricas e transições de conectividade.
* **Sem isso:** A feature continua disponível, mas o comportamento fica **opaco e difícil de auditar**.
* **Regra:** Fallback útil é fallback explicitamente operacionalizado.

### 14. Observabilidade aplicada ao componente
* **Vira contrato operacional:** Deixa de ser preocupação arquitetural genérica.
* **O que cobrir:** Métricas de decisões, erros, fallback e desempenho; logs estruturados sem dados sensíveis; tracing com atributos e spans úteis.
* **Depurável em produção:** Esse detalhamento é o que permite localizar gargalos e falhas.
* **Bom FDD:** Não diz apenas que "haverá telemetria" — define **o que será observável**.

### 15. Critérios de aceite e prontidão para código
* **Checklist verificável:** Convertem o documento em referência testável.
* **Sinais objetivos:** Contratos funcionando, testes passando, desempenho sob carga validado e resiliência confirmada.
* **Alinha o time:** Reduz discussão subjetiva sobre "estar pronto".
* **Ponte para o código:** Nesse nível, o FDD já serve diretamente para tarefas de implementação.

### 16. Pesquisa técnica como insumo para IA
* **IA não inventa requisitos:** Acelera a escrita, mas não cria requisitos corretos que não foram fornecidos ou compreendidos.
* **Risco da omissão:** Sem saber que Lua, mutex por chave, semântica de headers ou modos de fallback importam, o rascunho omite o que mais afeta a implementação.
* **Pesquisa amplia o repertório:** Mais decisões podem ser pedidas e revisadas.
* **Onde a IA rende:** Quando recebe **contexto técnico real**, não quando substitui o entendimento do problema.

## Aula 8: Prompt para FDD

Esta aula fecha o trio de prompts mostrando como gerar um **FDD com apoio da IA**. O template funciona como **contrato de completude** e as perguntas como **checklist de definição** — úteis mesmo sem rodar a entrevista. O draft melhora quando a IA recebe insumos combinados (HLD, PRD, codebase, template), tratando a geração como **composição guiada**, não improviso, e mantendo a revisão humana como fechamento do ciclo.

> 📄 **Exemplo de referência:** o prompt completo está em [Prompt para geração de um FDD](/docs/design-docs/templates-design-arquitetura/ex_prompt_gerar_FDD.md). O resultado aplicado está no [FDD — Rate Limiter](/docs/design-docs/templates-design-arquitetura/ex_FDD_Rate_Limiter.md), partindo do [HLD](/docs/design-docs/templates-design-arquitetura/ex_HLD_Rate_Limiter.md) e do [PRD](/docs/design-docs/templates-prd/ex_PRD_Feature_Rate_Limiter.md) da mesma feature.

---

### 1. Template do FDD como contrato de completude
* **Define o mínimo:** Quais partes precisam existir para o documento ser útil como especificação, mesmo antes de estar perfeito.
* **Evita o implícito:** Impede que contratos, comportamento, dependências e critérios fiquem esquecidos.
* **IA preenche, não escreve livre:** Com o template como estrutura de saída, ela preenche um espaço técnico já delimitado.
* **Ganho:** Mais consistência entre documentos e draft mais revisável.

### 2. Perguntas do prompt como checklist de definição
* **Mais que entrevista:** As perguntas revelam lacunas de escopo, comportamento e integração.
* **Valem mesmo sem rodar:** Você ainda precisa respondê-las em algum formato — elas explicitam o que precisa ser decidido.
* **Validação de prontidão:** Ler as perguntas com atenção verifica se a feature já está **suficientemente definida** para virar FDD.

### 3. Codebase como contexto adicional
* **Quando entra:** Se o conhecimento necessário não está todo no PRD, no HLD ou na memória de quem escreve.
* **O que agrega:** Trechos de código, contratos existentes, convenções internas e estruturas reais ajudam a IA a produzir um draft aderente ao sistema.
* **Menos genérico:** Aproxima o documento do ambiente real da empresa.
* **Limite:** Melhora a precisão do ponto de partida, mas **não substitui** decisão técnica.

### 4. Combinação de insumos para gerar o draft
* **Múltiplas fontes:** Template, HLD, PRD, respostas da entrevista e contexto do projeto.
* **Qualidade depende dos insumos:** O FDD pode existir sozinho, mas melhora muito com os artefatos que já fixaram arquitetura e objetivos.
* **No rate limiter:** Aproveitar o que já foi decidido e usar o template para detalhar sem recomeçar do zero.
* **Composição guiada:** A combinação transforma geração em composição, não em improviso.

### 5. IA como geradora de rascunho
* **Melhor como rascunho:** É mais confiável gerando draft do que como autora final autônoma.
* **No que ajuda:** Acelera a estruturação inicial, preenche seções recorrentes e organiza para revisão.
* **Ainda depende de:** Contexto suficiente e validação humana.
* **Objetivo correto:** Sair do zero para um draft útil — **não** automatizar a decisão técnica inteira.

### 6. Workflow de geração assistida
* **Sequência prática:** Reunir insumos → usar template e perguntas como guia → gerar draft → revisar.
* **Entrevista é opcional:** Se for mais lenta que escrever direto, preenche-se o mesmo template com contexto já preparado.
* **O centro não é a entrevista:** É a **qualidade dos insumos e da estrutura de saída**.
* **Revisão humana fecha o ciclo:** Corrige suposições e adiciona o que só o contexto da empresa fornece.

### 7. Aplicação prática no exemplo do rate limiter
* **Não redefine a feature:** Gera a especificação detalhada com apoio da IA.
* **Insumos anexáveis:** HLD existente, PRD correspondente, trechos da codebase e o template do FDD para pedir um draft inicial.
* **Pode pular a entrevista:** Usar diretamente o esqueleto de saída como instrução de preenchimento.
* **Resultado esperado:** Um documento próximo do que a empresa precisa, ainda tratado como **rascunho a refinar**.

## Aula 9: Realizando Deep Research

Esta aula apresenta o **deep research** como um modo em que a IA navega por fontes, cruza informações e produz uma pesquisa longa que vira **insumo técnico reutilizável** para os documentos seguintes. O foco prático é a **formulação do pedido**: um fluxo em **duas etapas** — entrevista curta que gera um resumo preparatório, depois a pesquisa longa — produz resultados muito melhores que um "pesquise sobre X" genérico.

> 📄 **Exemplo de referência:** o prompt da **Fase 1** (entrevista de contexto que gera o resumo preparatório) está em [Prompt para Deep Research — Fase 1](/docs/design-docs/templates-design-arquitetura/ex_prompt_deep_research_fase1.md).

---

### 1. Deep research como navegação longa da IA
* **Além do conhecimento imediato:** A IA navega por fontes, cruza informações e produz um documento longo de pesquisa.
* **Quando importa:** Para especificar uma feature sem depender só de memória, intuição ou conhecimento parcial.
* **No rate limiter em Go:** A pesquisa vira insumo técnico para o detalhamento não ser escrito **no escuro**.

### 2. Pesquisa técnica como insumo para documentos
* **Não é o fim:** O documento extenso funciona como **contexto reutilizável** para gerar artefatos posteriores.
* **Drafts melhores:** Com esse insumo, a IA produz design docs mais completos e menos genéricos.
* **No rate limiting:** Sustenta decisões sobre estratégias, bibliotecas, restrições e pontos de atenção antes da especificação.

### 3. Limite prático de modelos para documentos longos
* **Comportamento varia:** Modelos diferentes lidam de forma diferente com arquivos extensos.
* **Casos:** Às vezes o Gemini produz documentos maiores e mais completos; em outros, o ChatGPT atende bem, e o Claude também pode servir.
* **Decisão prática, não ideológica:** Escolher o modelo que sustenta melhor **volume, completude e consistência** para o tipo de pesquisa.

### 4. O erro mais comum ao pedir deep research
* **Não é a ferramenta:** O problema está na formulação do pedido.
* **"Faça uma pesquisa sobre X":** Gera resultado amplo demais, superficial ou desalinhado com o uso real.
* **O que falta:** Contexto explícito sobre problema, ambiente, profundidade, tecnologias relevantes e resultado esperado.

### 5. Prompt em duas etapas
* **Separar em fases:** Primeiro a IA conduz uma entrevista curta e monta um resumo preparatório; depois dispara a pesquisa longa com base nele.
* **Por que melhora:** A navegação longa parte de um **briefing técnico calibrado**, não de um pedido genérico.

### 6. Resumo preparatório antes da pesquisa longa
* **O que consolida:** Tema técnico, motivação, foco, contexto de uso, profundidade, stack relevante, necessidade de exemplos reais e resultado esperado.
* **Reduz ambiguidade:** Força o autor a explicitar **o que quer aprender**.
* **Sem alinhamento:** A IA gasta esforço em tópicos irrelevantes ou deixa lacunas nas decisões que mais importam.

### 7. Entrevista guiada aplicada à pesquisa
* **Outra função:** A entrevista guiada aqui **prepara a pesquisa**, não gera o documento final.
* **Como opera:** Uma pergunta por vez, usando as respostas para compor o briefing.
* **No exemplo:** Perguntas sobre motivação, contexto HTTP em microserviços, profundidade e uso de Redis refinam o escopo antes da busca longa.

### 8. Como disparar a pesquisa depois da preparação
* **Disparo simples:** Com o resumo pronto, basta pedir que a IA pesquise com base naqueles aspectos.
* **O ganho está antes:** Não na sofisticação do comando final, mas na **qualidade do contexto acumulado**.
* **Trate como job:** A etapa longa pode levar dezenas de minutos — é coleta e síntese, não resposta instantânea.

### 9. Por que não forçar um template na primeira fase
* **Esqueleto rígido falha:** Em deep research, a IA nem sempre respeita bem um formato fixo de saída.
* **Mais eficaz:** Deixar a pesquisa sair em **formato livre** e só depois adaptar para um template próprio.
* **Duas preocupações separadas:** Primeiro obter conhecimento amplo e útil; depois reorganizar para consumo recorrente.

### 10. Segunda etapa: adaptar a pesquisa para formato reutilizável
* **Boa base ≠ bom insumo operacional:** Uma pesquisa de 27 páginas pode ser excelente como base e ruim como entrada para o fluxo.
* **Função da 2ª etapa:** Converter o conteúdo em uma estrutura compatível com os templates do time, inclusive para gerar FDDs depois.
* **O valor:** Transformar **pesquisa bruta em contexto estruturado**, pronto para alimentar novos prompts e documentos.

## Aula 10: Adaptando Deep Research a um novo formato

Esta aula é a **Fase 2** do deep research: pegar a pesquisa bruta (PDF/Markdown) e **transpô-la** para um template estruturado de 16 seções, sem resumir. O ponto crítico é que a IA precisa **reorganizar, não condensar** — e a revisão humana cobra fidelidade estrutural e densidade técnica. O resultado é pesquisa virando **contexto reutilizável** que alimenta FDDs e demais documentos do pipeline.

> 📄 **Exemplo de referência:** o prompt da **Fase 2** (reformatação para o *Deep Research Document* de 16 seções) está em [Prompt para Deep Research — Fase 2](/docs/design-docs/templates-design-arquitetura/ex_prompt_deep_research_fase2.md). Veja também a [Fase 1](/docs/design-docs/templates-design-arquitetura/ex_prompt_deep_research_fase1.md).

---

### 1. Reformatar sem perder conteúdo
* **Rica, mas pouco operacional:** Uma Deep Research em PDF/Markdown serve mal a fluxos posteriores.
* **Transpor, não resumir:** O objetivo é levar o **mesmo conteúdo** para um template estruturado que facilite consulta, revisão e reaproveitamento.
* **Muda o prompt:** A IA precisa **reorganizar, não condensar** — se resumir demais, o documento vira só uma visão geral e perde valor técnico.

### 2. Template com 16 seções como estrutura de trabalho
* **Documento coringa:** Seções como contexto, fundamentos, conceitos-chave, panorama, arquiteturas, estratégias, algoritmos, tecnologias, boas práticas, métricas, casos de uso, riscos, segurança, tendências e impacto.
* **Por que funciona:** Distribui a pesquisa em blocos previsíveis, facilitando localizar decisões e alimentar outros artefatos.
* **Não é regra fixa:** As 16 seções são um ponto de partida adaptável ao tipo de projeto.
* **Equilíbrio:** Padronização suficiente para reutilização, sem engessar o conteúdo.

### 3. Procedimento prático no chat
* **Fluxo direto:** Abrir um novo chat, colar o prompt de adaptação e anexar o PDF/Markdown da Deep Research.
* **Instrução central:** Pedir **preservação integral** do conteúdo e adequação ao formato, sem resumo indevido.
* **No rate limiter:** A pesquisa bruta vira um documento com arquiteturas, tipos de rate limiting, estratégias, algoritmos e referências por seção.
* **Resultado:** Continua extenso, mas agora **navegável** e mais útil como insumo técnico.

### 4. Avaliação crítica da saída da IA
* **Tendência a resumir:** A IA tenta condensar documentos longos mesmo quando o prompt pede o contrário.
* **O que revisar:** Se exemplos, estratégias, links, trechos técnicos e nuances foram preservados.
* **Saída superficial:** A correção é **iterar o prompt**, repetir a geração ou trocar de modelo — não aceitar o texto como está.
* **Critério de qualidade:** Não é elegância do texto, e sim **fidelidade estrutural e densidade técnica**.

### 5. Escolha de modelo para documentos extensos
* **Comportamento varia:** Modelos diferem em tarefas longas de reestruturação.
* **O que comparar:** Capacidade de manter volume, detalhamento e aderência ao pedido de **não resumir**.
* **Citações:** O Gemini é apontado como útil para saídas extensas (inclusive em cenários gratuitos); o ChatGPT pode resumir além do desejado em alguns casos.
* **Decisão empírica:** Testar o mesmo insumo em mais de um modelo e avaliar qual preserva melhor.

### 6. Pesquisa estruturada como insumo para outros documentos
* **Mais que "texto bonito":** Transforma pesquisa crua em **contexto reutilizável**.
* **O que habilita:** Servir de base para drafts de FDD, complementar prompts e reduzir a dependência de entrevistas quando o contexto já está consolidado.
* **Compor em vez de começar do zero:** O fluxo passa a montar documentos a partir de insumos preparados.
* **Efeito:** Menos fricção e melhor qualidade do rascunho inicial.

### 7. Escopo explícito evita pesquisa desequilibrada
* **Extensão não garante equilíbrio:** Uma Deep Research longa pode aprofundar demais alguns pontos e tratar mal outros decisivos.
* **Pedido genérico:** Leva a cobertura desigual do que importa para design e implementação.
* **Estruturar o escopo:** Distribui a atenção entre fundamentos, mecanismos, riscos, segurança e aplicação prática.
* **Resultado:** Um documento mais útil para engenharia, não apenas mais longo.

### 8. Documentação como pipeline de implementação
* **Contexto acumulado:** Pesquisa técnica, documentos de alto nível, detalhes técnicos e guidelines somados dão à IA um contexto muito mais forte.
* **O que permite:** Quebrar a implementação em partes, derivar tarefas, sugerir snippets e apoiar decisões com menos improviso.
* **Não substitui programar:** Antecipa decisões e reduz ambiguidade na execução.
* **Valor crescente:** Aumenta quando os documentos continuam sendo **atualizados** conforme o desenvolvimento avança.

## Aula 11: Gerando um FDD a partir da Deep Research

Esta aula fecha o módulo mostrando o **fluxo completo de composição**: gerar um FDD combinando três insumos — **HLD** (decisões já tomadas), **Deep Research** (densidade técnica) e o **esqueleto do FDD** (contrato de saída). Com contexto forte, até um prompt simples produz um draft valioso. A IA não entrega um documento final, mas **evita a página em branco** e organiza o refinamento — que continua sendo trabalho humano e iterativo.

> 📄 **Exemplos de referência:** o fluxo combina [HLD — Rate Limiter](/docs/design-docs/templates-design-arquitetura/ex_HLD_Rate_Limiter.md) + a Deep Research (Fases [1](/docs/design-docs/templates-design-arquitetura/ex_prompt_deep_research_fase1.md) e [2](/docs/design-docs/templates-design-arquitetura/ex_prompt_deep_research_fase2.md)) + o [Prompt para geração de um FDD](/docs/design-docs/templates-design-arquitetura/ex_prompt_gerar_FDD.md), resultando em algo como o [FDD — Rate Limiter](/docs/design-docs/templates-design-arquitetura/ex_FDD_Rate_Limiter.md).

---

### 1. Composição explícita dos insumos
* **Três artefatos, papéis distintos:** O **HLD** fixa decisões arquiteturais; a **Deep Research** amplia o repertório técnico; o **esqueleto do FDD** define a estrutura de saída.
* **Não é "do zero":** A IA preenche uma moldura com contexto real, em vez de inventar tudo.
* **No rate limiter:** A geração deixa de depender só de entrevista e **reutiliza** tudo o que já foi produzido no fluxo.

### 2. Montagem prática do prompt
* **Simples pode bastar:** Com contexto forte, o prompt não precisa ser elaborado.
* **Instrução central:** Pedir o FDD da feature, informar que o HLD será fornecido, anexar a pesquisa e exigir aderência ao esqueleto.
* **Orientar o nível:** Não descer à implementação linha por linha, mas explicitar contratos, comportamento e decisões suficientes para orientar a execução.

### 3. HLD como insumo operacional
* **Âncora, não decoração:** Serve para fixar o draft nas decisões já estabilizadas.
* **A IA herda:** Arquitetura escolhida, componentes, responsabilidades e trade-offs — sem reinventar a solução.
* **No exemplo:** Mantém coerência com middleware, estratégias, storage e limites arquiteturais já definidos.

### 4. Deep Research como insumo de densidade técnica
* **Complementa o HLD:** Traz o que não cabe no documento arquitetural — comparações, estratégias, referências externas e implicações operacionais.
* **No mesmo prompt:** Dá base para enriquecer seções do FDD com mais precisão técnica.
* **Efeito prático:** Um rascunho menos genérico, mais próximo de uma especificação utilizável.

### 5. Template como contrato de saída
* **Força cobertura mínima:** Garante contexto, escopo, exclusões, erros, observabilidade, compatibilidade e critérios de aceite.
* **Obrigação estrutural:** Em vez de confiar que a IA lembrará espontaneamente, o template transforma isso em exigência.
* **Mais revisável:** A equipe inspeciona **lacunas por seção**, não apenas pela impressão geral do texto.

### 6. Prompt imperfeito com contexto forte
* **Insumos compensam:** Um prompt "mal feito" ainda produz draft valioso quando o contexto é bom.
* **Por quê:** A qualidade depende da densidade e complementaridade do contexto, não só da redação do comando.
* **Lição:** Não abandonar bons prompts, mas entender que **contexto forte reduz fragilidade** e acelera a primeira versão.

### 7. Grounding e rastreabilidade da saída
* **O que é:** Indicação explícita de onde a IA tirou cada informação (trechos/referências do PDF anexado).
* **Trade-off:** Pode poluir a leitura quando se quer um documento limpo, mas ajuda em auditoria, validação e revisão crítica.
* **Em design:** Funciona como **rastreabilidade** — mostra se a saída está apoiada nos insumos ou se começou a improvisar.

### 8. Leitura crítica do draft gerado
* **Bom draft ≠ perfeito:** É o que **evita a página em branco** e organiza o refinamento.
* **O que revisar:** Aderência ao HLD, cobertura das seções, presença de detalhes úteis e distorções de foco (ex: ênfase excessiva em multi-tenant quando não era central).
* **No rate limiter:** A utilidade aparece quando já traz contratos, headers, erros, fallback, observabilidade, dependências, compatibilidade e critérios de aceite em formato editável.

### 9. Iterações manuais como parte do processo
* **Não elimina edição:** Desloca o esforço de **escrever tudo** para revisar, corrigir, compor e ajustar.
* **Por que importa:** Nem tudo virá correto, completo ou no nível certo de detalhe.
* **Fluxo robusto:** Assume várias **iterações curtas** sobre um rascunho inicial, não uma geração automática definitiva.

### 10. Documentação como ativo de engenharia
* **Dá trabalho, mas rende:** Vira ativo quando passa a ser lida por pessoas e por IA ao longo do ciclo de vida.
* **O que acelera:** Implementação, revisão técnica, testes, manutenção e futuras automações baseadas em contexto.
* **Por que importa:** O desenvolvimento não termina no código — decisões, contratos e restrições precisam continuar **acessíveis, atualizáveis e reutilizáveis**.
