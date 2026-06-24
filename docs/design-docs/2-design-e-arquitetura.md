# Módulo 2 - Design e Arquitetura

## Sumário

- [Aula 1: Documentos de Design e Arquitetura](#aula-1-documentos-de-design-e-arquitetura)

- [Aula 2: Documentos comuns](#aula-2-documentos-comuns)


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
