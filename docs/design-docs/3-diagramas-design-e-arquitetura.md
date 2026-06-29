# Módulo 3 - Diagramas (Design e Arquitetura)

## Sumário

- [Aula 1: Introdução a Diagramas](#aula-1-introdução-a-diagramas)

- [Aula 2: Introdução aos diagramas C4](#aula-2-introdução-aos-diagramas-c4)


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
