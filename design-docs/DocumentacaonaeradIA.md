# Módulo - Documentação na era da IA

## Sumário

- [Aula 1: Documentação como parte do desenvolvimento](#aula-1-documentação-como-parte-do-desenvolvimento)

- [Aula 2: Tipos de documentação e Design Docs](#aula-2-tipos-de-documentação-e-design-docs)

- [Aula 3: PRDs - Product Requirement Document](#aula-3-prds---product-requirement-document)


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
