# Módulo 3 - Diagramas (Design e Arquitetura)

## Sumário

- [Aula 1: Introdução a Diagramas](#aula-1-introdução-a-diagramas)


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
