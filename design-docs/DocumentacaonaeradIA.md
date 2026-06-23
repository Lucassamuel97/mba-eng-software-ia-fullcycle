# Módulo - Documentação na era da IA

## Sumário

- [Aula 1: Documentação como parte do desenvolvimento](#aula-1-documentação-como-parte-do-desenvolvimento)

- [Aula 2: Tipos de documentação e Design Docs](#aula-2-tipos-de-documentação-e-design-docs)


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
