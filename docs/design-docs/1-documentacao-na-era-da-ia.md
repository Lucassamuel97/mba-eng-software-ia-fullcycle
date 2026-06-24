# Módulo - Documentação na era da IA

## Sumário

- [Aula 1: Documentação como parte do desenvolvimento](#aula-1-documentação-como-parte-do-desenvolvimento)

- [Aula 2: Tipos de documentação e Design Docs](#aula-2-tipos-de-documentação-e-design-docs)

- [Aula 3: PRDs - Product Requirement Document](#aula-3-prds---product-requirement-document)

- [Aula 4: Seções de um PRD](#aula-4-seções-de-um-prd)

- [Aula 5: PRDs de Alto Nível](#aula-5-prds-de-alto-nível)

- [Aula 6: PRDs de Feature e casos de uso](#aula-6-prds-de-feature-e-casos-de-uso)

- [Aula 7: Principais Seções em um PRD de Feature](#aula-7-principais-seções-em-um-prd-de-feature)


## Aula 1: Documentação como parte do desenvolvimento

Esta aula reposiciona a documentação: ela deixa de ser um artefato preparatório ou burocrático e passa a **participar diretamente do desenvolvimento**. Na era da IA, o documento é lido por modelos para orientar implementação, manutenção e planejamento — aproximando-se do papel dos testes automatizados como preservador de conhecimento e como insumo operacional dentro do workflow.

---

### 1. Documentação como parte do desenvolvimento
* **Deixa de ser artefato burocrático:** A documentação não é mais apenas um passo preparatório ou um custo de processo.
* **Participa da implementação:** Por ser legível por modelos, ela orienta diretamente implementação, manutenção e planejamento.
* **Paralelo com os testes:** Assim como testes automatizados, a documentação **preserva conhecimento útil ao longo do tempo** e reduz erro em mudanças futuras.

### 2. IA como consumidora de documentação
* **Contexto operacional:** Modelos de IA usam a documentação como **contexto** para executar trabalho técnico.
* **Geração mais assertiva:** Com contexto claro, a IA não depende apenas de inferência a partir do código-fonte ou de prompts curtos — e gera código mais alinhado.
* **Novo valor:** A documentação não serve só para registrar decisões humanas; serve para **alimentar um agente** que produz com base nelas.

### 3. A analogia do novo chat como funcionário novo
* **Cada chat começa do zero:** Um novo chat se comporta como um funcionário recém-chegado — não conhece domínio, restrições, objetivos nem convenções locais.
* **Código não basta:** O código ajuda, mas não substitui a visão explícita do sistema, das intenções e dos limites do produto.
* **Ambientação mais curta:** A documentação reduz o tempo de onboarding e aumenta a confiança de que a IA **entendeu o projeto** antes de propor mudanças.

### 4. Documentação como ativo de longo prazo
* **Continua útil depois da criação:** Um documento bem escrito segue valioso muito tempo depois, desde que permaneça alinhado ao sistema.
* **Paralelo com testes antigos:** Assim como um teste antigo ainda protege contra regressão, um documento antigo ainda **fornece contexto** para novas interações com IA.
* **Ativo acumulativo:** Isso transforma documentação em ativo que acumula valor, e não em **custo descartável**.

### 5. Geração de documentação com IA
* **IA não gera só código:** Ela também **acelera a produção de documentação**.
* **Mais rapidez, consistência e cobertura:** O ganho aparece quando o time sabe quais tipos de documento quer produzir e como orientar essa geração.
* **Os dois lados do ciclo:** O valor real surge quando a IA apoia tanto **implementar** quanto **documentar**.

### 6. Documento vivo viabilizado por workflow com IA
* **Sempre desejável, raramente sustentado:** A ideia de documento vivo existe há tempo, mas dificilmente se mantinha de forma manual.
* **Atualização incorporada ao fluxo:** Com um workflow bem definido, sempre que uma feature é desenvolvida ou o software é alterado, a IA recebe a tarefa de **revisar e atualizar** os documentos afetados.
* **Menos dependência de disciplina:** O documento deixa de depender só de esforço manual e passa a ser mantido como **parte do processo**.

### 7. Atualização automática após mudanças no software
* **Ciclo de sincronização contínua:** Quando a IA gera o código e depois atualiza a documentação com base nesse mesmo código, cria-se um ciclo que mantém os dois lados alinhados.
* **Reduz a defasagem:** Esse ciclo combate o principal motivo pelo qual documentos perdem valor — a distância entre sistema e documentação.
* **Condição necessária:** Funciona apenas com um **workflow explícito** que trate a atualização documental como etapa obrigatória da mudança.

### 8. Workflow de documentação na era da IA
* **Entrada e saída:** A documentação entra no workflow nos dois sentidos — primeiro como **contexto** para um desenvolvimento mais assertivo, depois como **artefato regenerado** para refletir o que foi implementado.
* **Documentação operacional:** Ela não apenas descreve o sistema; **participa ativamente da sua evolução**.
* **Fluxo completo:** Esse ciclo torna o documento parte do processo de engenharia, e não um anexo desconectado dele.

## Aula 2: Tipos de documentação e Design Docs

Esta aula organiza a documentação de software em uma **taxonomia funcional**: cada categoria responde a uma pergunta diferente do projeto (o quê, por quê, como, onde, como manter, como trabalhar). A distinção central separa **documentação de produto** (intenção) de **design docs** (decisão técnica), tratando design docs como um guarda-chuva de artefatos técnicos com públicos e finalidades distintos.

---

### 1. Taxonomia dos tipos de documentação
* **Cada categoria, uma pergunta:** A documentação se organiza melhor quando cada tipo responde a uma questão específica do projeto.
* **Camadas principais:** **produto** define o que será construído e por quê; **design e arquitetura** definem como implementar; **infraestrutura** define onde e com quais recursos provisionar; **operacional** cobre como manter a aplicação; **conhecimento e referência** registram como trabalhar no contexto do sistema.
* **Evita misturar objetivos:** Essa separação impede que documentos combinem propósitos incompatíveis e melhora a comunicação entre produto, engenharia e operação.

### 2. Documentação de produto versus Design Docs
* **Fronteira principal:** Está entre **intenção de produto** e **decisão técnica**.
* **Produto:** Descreve escopo, objetivo e motivação — não detalha a estrutura técnica de implementação.
* **Design docs:** Começam quando a discussão sai do "o quê" e "por quê" e entra no "como", "onde", "como manter" e "como trabalhar".
* **Não são sinônimos:** O capítulo trata **PRD** como documento de produto e **design docs** como documentos técnicos complementares.

### 3. Design Docs como guarda-chuva de documentos técnicos
* **Não é um documento único:** Design docs formam um **conjunto de artefatos técnicos** com finalidades diferentes, não um documento universal.
* **O que entra:** documentos de design e arquitetura, infraestrutura, operação e conhecimento técnico de referência — todos orientam construção, provisão, manutenção e evolução.
* **Risco de generalizar:** Chamar qualquer documento de "design doc" apaga diferenças importantes de **público, profundidade e uso**.
* **Ganho:** Tratá-los como categoria técnica ajuda a escolher formato, nível de detalhe e momento de criação de cada documento.

### 4. Mapeamento funcional: o quê, por quê, como, onde, como manter, como trabalhar
* **Mapa simples resolve ambiguidade:** produto → **o quê e por quê**; design e arquitetura → **como**; infraestrutura → **onde e com o quê**; operacional → **como manter**; conhecimento e referência → **como trabalhar**.
* **Vocabulário comum:** Esse mapeamento funciona como linguagem compartilhada do time e reduz a confusão sobre o papel de cada artefato.
* **Documento que responde tudo:** Tende a ficar genérico, redundante ou difícil de manter.
* **Separar por função:** Torna a documentação mais útil tanto para pessoas quanto para sistemas de IA.

### 5. Implicações práticas para o projeto
* **Necessidades diferentes, decisões diferentes:** Cada tipo de documentação exige escolhas próprias de escopo, timing e granularidade.
* **Sem repetição desnecessária:** Um documento técnico não precisa repetir toda a justificativa de produto; um documento de produto não precisa antecipar detalhes de provisionamento ou operação.
* **Tipo antes do template:** O tamanho e a profundidade dependem da situação — **o tipo correto de documento vem antes do template correto**.
* **Visão crítica:** Evita produzir artefatos extensos, mas pouco acionáveis.

### 6. IA apoiando geração e manutenção de documentação
* **Contexto organizado por categoria:** Retomando a documentação como contexto operacional para IA, o avanço aqui é **organizar esse contexto por tipo documental**.
* **Geração mais coerente:** Com tipos bem delimitados, a IA gera artefatos alinhados ao objetivo de cada um, em vez de misturar requisitos, arquitetura e operação.
* **Manutenção mais simples:** Atualizar um documento fica mais fácil quando seu papel está claramente definido.
* **Resultado:** Estrutura documental clara melhora tanto a produção quanto a **preservação do contexto** ao longo do tempo.

### 7. Prompt Engineering aplicado à documentação
* **Instrução determina qualidade:** O documento gerado depende de instruir a IA sobre **tipo, objetivo, escopo e profundidade** do artefato.
* **Vago vs. delimitado:** Pedir "gere um design doc" é vago; pedir um documento técnico de arquitetura, infraestrutura ou operação produz saídas mais úteis porque a categoria já delimita o conteúdo.
* **Prompt operacionaliza a taxonomia:** O enquadramento por categoria transforma o prompt em mecanismo de aplicação da taxonomia documental.
* **Quanto mais claro o enquadramento:** Maior a chance de a IA **gerar e manter** artefatos consistentes.

## Aula 3: PRDs - Product Requirement Document

Esta aula apresenta o **PRD** como o documento de produto que explicita **o quê** está sendo construído, **por quê** existe e qual valor entrega. Mesmo não sendo um design doc, ele entra **antes** dos documentos técnicos por fornecer o contexto que a IA precisa para interpretar o projeto. A aula define quando algo merece um PRD, seus níveis de granularidade (produto, módulo, feature) e seu papel como transferência de contexto explícito para uma IA que não conhece o produto.

---

### 1. PRD no fluxo de documentação orientado por IA
* **Documento de produto:** Explicita o que está sendo construído, por que existe e qual valor entrega.
* **Vem antes do técnico:** Embora não seja um design doc, entra **antes** dos documentos técnicos porque a IA precisa desse contexto para interpretar corretamente o projeto.
* **Sem essa camada:** O modelo tende a inferir objetivos apenas a partir de código ou pedidos isolados, aumentando a ambiguidade.
* **Papel central:** Servir como **base de contextualização** para pessoas e para IA.

### 2. Documento de produto versus documento técnico
* **Separação melhora o contexto:** Retomando a distinção produto × design docs, o ponto novo é que ela **eleva a qualidade do contexto** fornecido à IA.
* **O que o PRD não faz:** Não descreve como implementar em termos de arquitetura, componentes ou infraestrutura — mas pode incluir questões técnicas relevantes para alinhar produto e engenharia.
* **Intenção ≠ implementação:** Evita misturar os dois e preserva a função de cada artefato.
* **Efeito:** Partindo de um PRD claro, os design docs passam a responder ao **problema certo**.

### 3. Quando algo merece um PRD
* **Critério de valor:** Deve existir quando há **entrega de valor percebido** para o usuário ou para o negócio.
* **Evita formalismo excessivo:** Impede transformar qualquer requisito funcional em documento formal e concentra esforço no que altera objetivos, métricas, escopo ou impacto.
* **Limiar:** Iniciativas que geram dúvidas relevantes, têm alto valor percebido ou exigem alinhamento entre áreas já se aproximam do ponto em que um PRD faz sentido.
* **Unidade de produto:** O documento trata o item como unidade de produto, não apenas como tarefa de implementação.

### 4. Feature relevante como subproduto
* **Nem toda feature merece PRD:** Especialmente quando é apenas mais um requisito funcional dentro de algo maior.
* **A exceção:** Quando a feature é tão expressiva que funciona como **subproduto** — objetivo próprio, métricas próprias, escopo próprio e impacto alto.
* **Risco de não isolar:** Tratá-la como uma linha em um documento geral reduz clareza e dificulta o alinhamento.
* **Função do PRD:** Isolar a feature para que **valor e limites** fiquem explícitos.

### 5. Clareza de escopo, objetivos e métricas
* **Mensurável e delimitado:** Um PRD útil organiza o item como algo mensurável e com fronteiras claras.
* **O que explicitar:** objetivos, escopo e formas de medir resultado — evitando que uma iniciativa relevante vire descrição vaga de intenção.
* **Beneficia coordenação e IA:** A clareza ajuda o alinhamento produto × engenharia e reduz as **suposições implícitas** da IA.
* **Sem delimitação:** Quando não fica claro o que está dentro e fora, o contexto perde valor operacional.

### 6. Granularidade de PRD
* **Níveis diferentes:** PRDs existem em granularidades distintas.
* **PRD de produto:** Cobre o produto como um todo — mais amplo e menos detalhado.
* **PRD de módulo:** Aprofunda uma parte relevante da aplicação.
* **PRD de feature:** Foca uma entrega específica com contexto próprio.
* **Escolha do nível:** Depende do tamanho da iniciativa e da autonomia conceitual do recorte — evita documentos genéricos demais ou fragmentação excessiva.

### 7. Produto, módulo e feature como níveis possíveis
* **Coexistência:** Um mesmo software pode ter um PRD macro do produto e PRDs específicos para módulos ou features críticas.
* **EPIC como aproximação:** O termo pode aparecer como equivalente ao nível de módulo no vocabulário de produto, mesmo sem correspondência exata em todos os contextos.
* **O que importa:** Reconhecer quando uma parte do sistema concentra **valor, complexidade de alinhamento ou identidade** suficiente para merecer documentação própria.
* **Resultado:** O PRD vira ferramenta de organização de contexto, não burocracia fixa.

### 8. PRD como apoio direto ao desenvolvedor
* **Não é exclusivo de PMs:** Desenvolvedores também podem participar da criação do documento.
* **Quando ajuda:** Ao aumentar clareza sobre a feature a ser construída, especialmente em iniciativas de alto impacto ou escopo sensível.
* **Valor em ambientes com IA:** O documento torna **explícito** o contexto que o time já conhece informalmente.
* **Aparente redundância:** O que parece óbvio para pessoas experientes ainda é **informação ausente** para a IA.

### 9. Contexto explícito para uma IA que não conhece o produto
* **O problema simples:** O time conhece o produto, mas a IA não.
* **Registrar o óbvio não é desperdício:** É a forma de transferir **intenção, valor e limites** para um agente que entra sem histórico do projeto.
* **Ordem correta:** Antes de pedir **como** construir algo, é preciso explicitar **o que** esse algo é e **por que** existe.
* **Efeito:** Essa ordem melhora a qualidade das respostas técnicas geradas depois.

## Aula 4: Seções de um PRD

Esta aula detalha a **anatomia típica de um PRD**, tratando suas seções como uma estrutura **flexível**, não um template rígido. Cada bloco (visão, contexto, público, objetivos, escopo, requisitos, estratégia, riscos, KPIs, stakeholders) cobre uma parte do contexto necessário para "contar a história do produto" — mas a presença e o detalhamento de cada seção variam conforme tamanho, granularidade e necessidade real de alinhamento.

---

### 1. Flexibilidade estrutural do PRD
* **Não é template rígido:** Não há campos obrigatórios nem nomenclatura fixa.
* **Seções variam:** Conforme o tamanho da iniciativa, o nível de granularidade e a necessidade real de alinhamento.
* **Omissão contextual:** Em projetos pequenos, partes podem ser omitidas sem prejuízo; em iniciativas amplas, a ausência costuma reduzir clareza.
* **Ponto central:** Cobrir o contexto necessário para contar a história do produto de forma útil para o time e para a IA.

### 2. Visão e propósito
* **Ideia central:** Registra a ideia do produto e a razão de sua existência.
* **Responde a justificativa:** Qual problema ou direção estratégica motiva a iniciativa, sem entrar em implementação técnica.
* **Referência de intenção:** Quando decisões posteriores geram dúvida, a visão ajuda a verificar se o produto continua coerente com o motivo original.

### 3. Contexto do produto e oportunidade
* **Cenário e benefício:** Explica em que contexto o produto surge e qual ganho pode gerar para a organização.
* **Mercado ou interno:** Vale para produtos de mercado e soluções internas, desde que haja ganho operacional, estratégico ou econômico.
* **Sem enquadramento:** O documento vira apenas uma lista de desejos; com ele, cada decisão se vincula a uma **necessidade real do negócio**.

### 4. Público e personas
* **Público:** Identifica quem será afetado ou atendido pelo produto.
* **Personas:** Transformam o público em perfis concretos de uso — por faixa etária, domínio técnico, idioma, rotina, tipo de problema ou papéis internos de um departamento.
* **Objetivo real:** Não é inventar personagens por formalidade, mas explicitar **para quem** o produto faz sentido e quais necessidades orientam decisões.

### 5. Objetivos e métricas
* **Objetivo + medição:** Os objetivos definem o que a iniciativa quer alcançar e precisam vir acompanhados de **medição verificável**.
* **Sem forma de observar:** Um objetivo sem medição não orienta priorização nem valida sucesso.
* **PRD auditável:** Métricas permitem saber se o produto avançou na direção esperada ou apenas gerou entrega sem impacto.

### 6. Escopo
* **Função estrutural:** Dentro do PRD, o escopo delimita **fronteiras explícitas**.
* **Dentro e fora:** Informa o que faz parte do produto e o que fica de fora, evitando crescimento por expectativa implícita.
* **Sem limite declarado:** Times diferentes passam a assumir versões diferentes do mesmo projeto.

### 7. Requisitos de alto nível
* **Capacidades macro:** Descrevem o que o produto deve oferecer, sem decompor fluxos técnicos ou regras detalhadas.
* **Exemplo e-commerce:** Vender camisetas, permitir escolha de cor, realizar checkout e viabilizar entrega são **capacidades**, não especificações de implementação.
* **Transição:** Organiza o produto em blocos funcionais amplos e prepara a passagem para documentos mais detalhados.

### 8. Estratégia e fases de desenvolvimento
* **Da intenção à execução:** Registra como a iniciativa pretende sair da ideia para a realização.
* **Conteúdo:** Pode incluir fases, marcos ou sequência de evolução, ajudando o time a entender ordem, dependências e prioridades.
* **Evita o monolito:** Mesmo sem detalhe técnico, impede tratar o produto como um pacote único entregue de uma vez só.

### 9. Riscos
* **Fatores de comprometimento:** Documentam o que pode afetar prazo, valor, adoção ou viabilidade.
* **Antecipar, não listar tudo:** Servem para prever incertezas relevantes, não qualquer possibilidade genérica de problema.
* **Maturidade na decisão:** Explicitar riscos permite avaliar trade-offs com mais maturidade, em vez de planejar como se o cenário fosse estável.

### 10. KPIs
* **Indicadores de progresso:** Acompanham se a iniciativa avança corretamente ou se já atingiu o resultado esperado.
* **Observável:** Diferente de uma meta genérica, um KPI precisa ser monitorável durante o desenvolvimento ou após a entrega.
* **Sucesso monitorável:** Essencial para decidir continuidade, ajuste de rota ou encerramento.

### 11. Stakeholders
* **Quem tem interesse:** Pessoas ou papéis diretamente interessados — desenvolvimento, produto, liderança executiva e demais áreas impactadas.
* **Reduz ruído:** Torná-los explícitos esclarece quem influencia decisões, valida entregas e precisa ser consultado.
* **Documento não é só do produto:** Evita que o PRD seja lido como assunto exclusivo do time de produto.

### 12. Leitura correta dessa estrutura
* **Anatomia típica, não checklist:** A lista representa um PRD amplo, não um modelo universal obrigatório.
* **Detalhe varia com a granularidade:** Um PRD macro tende a ser mais amplo e menos minucioso que um PRD de feature.
* **O que realmente importa:** A utilidade depende menos do nome exato das seções e mais da **presença das informações** necessárias para contextualizar o produto.

## Aula 5: PRDs de Alto Nível

Esta aula apresenta o **PRD de alto nível** como o recorte mais amplo da documentação de produto: um artefato **macro** que enquadra a iniciativa como produto antes de qualquer decomposição em módulos, EPICs ou features. Sua estrutura nasce de um conjunto de **perguntas estratégicas** que conectam o produto à direção da empresa — e a capacidade de respondê-las é o próprio critério de clareza do projeto.

---

### 1. PRD de alto nível como artefato macro
* **Recorte mais amplo:** É o nível mais abrangente da documentação de produto.
* **Sem detalhe operacional:** Não entra no detalhamento de uma feature; **enquadra a iniciativa como produto** e expõe o contexto macro que orienta decisões posteriores.
* **Valor em projetos grandes:** Importa sobretudo quando várias decisões dependem de uma **visão comum** antes da decomposição em módulos, EPICs ou features.
* **Sem enquadramento:** O time pode até executar entregas, mas segue sem clareza sobre o que está construindo como produto.

### 2. Perguntas estratégicas como estrutura do documento
* **Documento como conjunto de perguntas:** O nível alto pode ser entendido pelas questões que o produto precisa responder.
* **Exemplos:** por que o produto existe, o que se quer alcançar, o que entra e o que fica fora, para quem se constrói, qual problema é resolvido, como os objetivos serão perseguidos, quais capacidades gerais são necessárias, como reconhecer o sucesso, o que pode dar errado, qual roadmap orienta a evolução, quem participa e como tudo se conecta à estratégia da empresa.
* **Não é questionário solto:** Funciona como **estrutura de raciocínio** para consolidar visão, direção e critérios de decisão.

### 3. Caráter estratégico do nível alto
* **Antecede a implementação:** Muitas perguntas são estratégicas porque vêm antes da execução — e às vezes nem nascem no time de desenvolvimento.
* **Decisões top-down:** Parte delas pode vir de liderança, diretoria ou da estratégia mais ampla da empresa.
* **Útil ao desenvolvedor:** Em vez de reduzir o valor do documento, isso **explicita premissas** que normalmente chegariam de forma implícita.
* **Sem registro:** A execução técnica passa a depender de interpretação parcial.

### 4. Conexão entre produto e estratégia da empresa
* **Produto não é item isolado:** No PRD macro, ele aparece como **resposta a uma direção maior** da empresa.
* **A pergunta relevante:** Não é só "o que vamos construir", mas "por que isso merece existir nesta organização agora".
* **Justifica prioridade:** Essa conexão fundamenta investimento e trade-offs e dá sentido a objetivos, riscos e métricas.
* **Sem vínculo:** Um produto pode parecer coerente localmente e ainda estar **desalinhado com a estratégia global**.

### 5. Roadmap e papéis como enquadramento macro
* **Organização no tempo e nas pessoas:** O documento mostra como a iniciativa se organiza no tempo e quem participa dela.
* **Roadmap macro:** Não é cronograma detalhado de execução, mas uma **visão de evolução** em etapas ou direções.
* **Papéis:** Tornam explícito quem influencia, decide, valida ou executa partes da iniciativa, reduzindo ambiguidade organizacional.
* **Foco:** É um enquadramento de **coordenação e alinhamento**, não de especificação técnica.

### 6. Clareza como critério de qualidade do PRD macro
* **Perguntas sem resposta = falta de clareza:** Se as questões centrais não podem ser respondidas, o problema não é documental, mas de clareza sobre o próprio projeto.
* **Importância para a IA:** Retomando o uso de PRD para IA, o modelo depende desse contexto para interpretar a iniciativa **antes de qualquer design doc**.
* **Menos inferência implícita:** Um documento macro bem formulado melhora a qualidade das decisões derivadas.
* **Próximo passo:** Descer do enquadramento estratégico para o nível em que uma **feature específica** passa a exigir seu próprio contexto.

## Aula 6: PRDs de Feature e casos de uso

Esta aula define **quando uma feature merece um PRD próprio** — não por existir no backlog, mas por concentrar **decisões de produto**. Usando o caso do login em dois cenários (commodity × plataforma estratégica), a aula mostra que o nome da funcionalidade não determina sua granularidade documental: o que decide é a **densidade de decisões de produto** ali concentradas.

---

### 1. Regra prática de decisão
* **Não basta existir no backlog:** Uma feature merece PRD quando deixa de ser requisito funcional commodity e passa a **concentrar decisões de produto**.
* **Sinais a observar:** valor percebido próprio, objetivos mensuráveis, impacto na experiência, regras específicas e trade-offs que precisam ser explicitados antes do design técnico.
* **Quando não justifica:** Se a implementação segue um padrão conhecido, com decisões essencialmente técnicas, o item cabe como requisito funcional dentro de um PRD maior.
* **Critério final:** Quando a feature vira uma **unidade real de decisão**, passa a justificar contexto documental próprio.

### 2. Feature versus requisito funcional
* **Requisito funcional:** Descreve uma capacidade necessária para o sistema operar — nem sempre com identidade de produto suficiente para virar documento separado.
* **A virada:** Quando a funcionalidade deixa de ser "o sistema precisa fazer X" e passa a exigir objetivos, políticas, métricas, integrações e restrições de negócio.
* **Risco de simplificar:** Tratá-la como simples linha de requisito **empobrece o contexto** disponível para o time e para a IA.
* **Papel do PRD de feature:** Registrar esse contexto adicional **sem misturá-lo com implementação**.

### 3. Caso 1: login commodity
* **Pré-requisito técnico:** Um login básico pode ser apenas a porta de entrada da plataforma.
* **Fluxo padrão:** Se segue o que frameworks já oferecem, não altera a proposta do produto, não inova e não exige decisões de produto.
* **Enquadramento natural:** Requisito funcional de um sistema maior.
* **PRD específico aqui:** Tende a gerar mais formalidade do que clareza — basta registrar que o sistema **precisa de autenticação**.

### 4. Por que o caso 1 não justifica PRD próprio
* **Não gera valor por si:** O login commodity apenas viabiliza o acesso, sem valor de negócio próprio.
* **Decisões são técnicas:** Recaem sobre tecnologia, framework e configuração, não sobre posicionamento de produto.
* **Pouca necessidade de contexto autônomo:** Reduz a demanda por objetivos próprios, métricas dedicadas ou escopo separado.
* **Documento mais útil:** O **PRD macro**, com o login aparecendo como mais um requisito funcional.

### 5. Caso 2: login como produto estratégico
* **Muda de categoria:** Em uma plataforma multi-tenant com foco em segurança corporativa, o login ganha outro peso.
* **Capacidades envolvidas:** single sign-on, autenticação de dois fatores, logout centralizado e políticas de acesso — com impacto em experiência, adoção e integrações.
* **De detalhe a organizador de valor:** O login passa a estruturar uma parte relevante do que é entregue.
* **Resultado:** Uma feature com **identidade própria** e decisões suficientes para merecer PRD específico.

### 6. Compliance e multi-tenant como sinais de enquadramento
* **Saída do commodity:** Compliance e multi-tenant indicam que a funcionalidade não é mais padronizada.
* **Novas restrições:** Atender exigências corporativas, suportar múltiplos clientes, controlar regras distintas de acesso e servir de base compartilhada gera decisões que não cabem em uma linha de requisito.
* **O que o documento precisa explicitar:** quem usa, quais políticas se aplicam, quais integrações são necessárias e quais riscos existem.
* **Sem isso:** Time técnico e IA tendem a **subestimar a complexidade real** da feature.

### 7. Objetivos de produto tornam a feature documentável
* **Sucesso mensurável:** Reduzir fricção no login, aumentar adoção, diminuir acessos indevidos ou viabilizar integrações corporativas.
* **Além de "funcionar":** Esses objetivos mostram que a feature existe para produzir **resultado observável** no negócio e na operação.
* **Produto em escala menor:** Quando esse tipo de meta aparece, a feature deixa de ser só implementação.
* **Função do PRD:** Organiza esse raciocínio **antes** que o design técnico transforme tudo em componentes e fluxos.

### 8. Utilidade do PRD de feature para desenvolvedores
* **Clareza mesmo sem papel de produto:** Desenvolvedores ganham ao escrever ou consumir um PRD de feature.
* **Menos inferência implícita:** O documento explicita por que a funcionalidade existe e melhora a contextualização usada pela IA em design docs, código e decisões auxiliares.
* **Features estratégicas:** Evita tratar como detalhe técnico algo que afeta segurança, onboarding, integração e operação corporativa.
* **Natureza do ganho:** Não é burocracia, é **precisão de contexto**.

### 9. Síntese comparativa
* **O nome não decide:** Nos dois casos existe "login", mas isso não determina a granularidade documental.
* **Caso 1:** Capacidade necessária e padronizada → requisito funcional.
* **Caso 2:** Plataforma de autenticação com impacto estratégico, regras próprias e valor de produto → PRD de feature.
* **O que realmente decide:** A **densidade de decisões de produto** concentradas na entrega — não a tecnologia usada. Quando ela existe, o PRD de feature prepara o terreno para as seções que detalham escopo, objetivos, regras e restrições.

## Aula 7: Principais Seções em um PRD de Feature

Esta aula detalha a **estrutura mínima de um PRD de feature**: as seções que aumentam clareza operacional para uma entrega específica, já no nível de execução do time (e não no enquadramento estratégico do PRD macro). Usando o exemplo de um **rate limiter**, percorre os blocos que concentram as decisões necessárias para alinhar produto, desenvolvimento e IA antes da implementação.

---

### 1. Estrutura mínima de um PRD de feature
* **Não replica o PRD macro:** Mantém apenas as seções que aumentam clareza operacional para a entrega.
* **Diferença de tipo, não só de tamanho:** Sai do enquadramento estratégico e entra no **nível de execução** do time.
* **Flexível, mas recorrente:** Alguns blocos aparecem com frequência porque concentram as **decisões mínimas** para alinhar produto, desenvolvimento e IA.

### 2. Resumo da feature e contexto do problema
* **Abre o documento:** Descrição curta do que está sendo construído e do problema que resolve.
* **Evita lista solta:** Impede que a feature seja lida como requisitos sem causa nem finalidade.
* **Exemplo rate limiter:** Não é "implementar limitação de requisições", mas registrar que há **excesso de acessos derrubando o sistema** e que a feature existe para conter isso.
* **Efeito:** Com o contexto explícito, decisões posteriores deixam de depender de interpretação implícita.

### 3. Objetivos e métricas no nível da feature
* **Mais diretos e verificáveis:** No nível da feature, ficam mais concretos que no PRD macro.
* **Formato típico:** Objetivos como bullet points declarando o resultado esperado; métricas definindo como confirmá-lo.
* **Sucesso ≠ implementado:** A combinação impede considerar a feature bem-sucedida só porque foi construída.
* **Critério real:** Impacto observável, não apenas conclusão técnica.

### 4. Escopo
* **Dentro e fora:** Delimita o que entra e o que fica de fora, reduzindo expansão informal durante a execução.
* **Precisa ser objetivo:** Pequenas ambiguidades viram retrabalho técnico rapidamente.
* **Protege a unidade da entrega:** O time sabe o que resolver agora e o que pertence a outra iniciativa.
* **Sem fronteira:** A feature cresce por acúmulo de expectativas não registradas.

### 5. Requisitos funcionais
* **Capacidades concretas:** Descrevem quais recursos existirão e quais comportamentos o sistema deve suportar.
* **O que registrar:** Se a feature exigir login, uma estratégia específica de storage ou outros componentes necessários, isso aparece aqui.
* **Funcionalidade observável:** Organiza o que será entregue sem virar **desenho técnico detalhado**.

### 6. Requisitos não funcionais
* **Restrições de qualidade e operação:** Condições em que o sistema precisa funcionar — latência, disponibilidade, limites de indisponibilidade.
* **Além do "o que faz":** Definem o "em que condições" a solução precisa operar.
* **Exemplo rate limiter:** Exigir latência mínima e controlar o tempo máximo de indisponibilidade muda diretamente a **viabilidade da implementação**.
* **Sem esse bloco:** A solução pode cumprir a função e ainda falhar no contexto real de uso.

### 7. Fluxo do usuário
* **Como a feature é usada:** Descreve interação e sequência de uso do ponto de vista do usuário.
* **Detecta lacunas:** Torna explícito o caminho esperado, revelando distâncias entre requisito listado e experiência real.
* **Evita capacidades isoladas:** Ajuda o time a entender ordem, dependência e lógica de uso entre os elementos.
* **Para a IA:** Reduz inferências erradas sobre comportamento esperado.

### 8. Dependências
* **O que a feature precisa:** Outro sistema, módulo, serviço ou decisão prévia para existir ou funcionar.
* **Não é autônoma:** Explicitá-las evita planejar a implementação como se não houvesse integrações ou pré-condições.
* **Melhora coordenação:** Beneficia estimativa, priorização e alinhamento entre times.
* **Para a IA:** Ajuda a não propor soluções desconectadas do ecossistema real do projeto.

### 9. Critérios de aceitação
* **Checklist de "pronto":** Condições que precisam estar satisfeitas para a feature ser considerada completa.
* **De vago a verificável:** Transformam a noção subjetiva de concluído em condições objetivas, úteis para produto, desenvolvimento e testes.
* **Referência de encerramento:** Substituem julgamento subjetivo por um critério explícito.
* **Especialmente útil:** Quando a feature envolve múltiplas regras e pode parecer concluída antes de atender ao esperado.

### 10. Riscos e considerações gerais
* **Riscos de execução:** Conectados à construção concreta da feature, não ao enquadramento macro — incertezas que comprometem desenvolvimento, adoção ou comportamento.
* **Considerações gerais:** Fechamento do documento; reúnem observações complementares que ainda influenciam decisão e implementação.
* **Preserva nuances:** Mantém detalhes importantes sem forçar encaixes artificiais em blocos inadequados.

### 11. Relação com desenvolvimento e com IA
* **Próximo do cotidiano:** O PRD de feature organiza informações **diretamente acionáveis** para construir a entrega.
* **Contexto para a IA:** Reduz ambiguidade sobre o que o código precisa resolver, quais limites respeitar e quais condições definem sucesso.
* **Não substitui design docs:** Mas melhora muito a qualidade dos artefatos técnicos produzidos depois.
* **Quanto mais explícito:** Menor a chance de time ou IA preencherem lacunas com **suposições erradas**.
