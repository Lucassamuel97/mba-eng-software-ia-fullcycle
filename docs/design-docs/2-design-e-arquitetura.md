# Módulo 2 - Design e Arquitetura

## Sumário

- [Aula 1: Documentos de Design e Arquitetura](#aula-1-documentos-de-design-e-arquitetura)

- [Aula 2: Documentos comuns](#aula-2-documentos-comuns)

- [Aula 3: High Level Design](#aula-3-high-level-design)


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
