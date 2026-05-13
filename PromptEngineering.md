# Módulo Prompt Engineering

## Sumário

- [Aula 1: Introdução e overview da disciplina](#aula-1-introdução-e-overview-da-disciplina)

- [Aula 2: O que realmente é Prompt Engineering](#aula-2-o-que-realmente-é-prompt-engineering)

- [Aula 3: Outras Aplicabilidades do Prompt Engineering](#aula-3-outras-aplicabilidades-do-prompt-engineering)

- [Aula 4: Técnicas e Tipos de Prompt](#aula-4-técnicas-e-tipos-de-prompt)

- [Aula 5: Role Prompting (Prompt de Persona)](#aula-5-role-prompting-prompt-de-persona)

## Aula 1: Introdução e overview da disciplina

Esta aula inaugural estabelece as bases de funcionamento do MBA, destacando a flexibilidade da ementa. O objetivo é permitir que os encontros ao vivo acompanhem as constantes inovações da área, como *prompt engineering* e novos workflows, sem ficar preso estritamente ao cronograma das aulas gravadas.

A dinâmica dos encontros ao vivo, com duração aproximada de 1h30, foca na aplicação prática e na interação. O professor utiliza uma analogia com séries de TV para explicar que as aulas são abrangentes, permitindo que o aluno acompanhe o conteúdo mesmo que não tenha assistido a todos os episódios anteriores, embora temas complexos exijam o estudo dos pré-requisitos disponíveis na plataforma.


## Aula 2: O que realmente é Prompt Engineering

Esta aula aborda a transição do *Prompt Engineering* de um termo de "hype" para uma competência técnica essencial no dia a dia do desenvolvedor.

---

### 1. Definição e Conceito
* **Não é apenas uma palavra da moda:** É a disciplina de estruturar instruções de forma clara e muitas vezes criativa para obter resultados específicos e de alta qualidade.
* **O "Salto" de Qualidade:** A diferença entre um prompt ruim e um bem estruturado é comparada à diferença entre interagir com um "adolescente" ou com um "pós-doutor".
* **Mecanismo de Resposta:** A capacidade de um modelo de IA compreender instruções complexas e técnicas depende diretamente da construção do prompt.

### 2. Contexto de Software (Domínio Específico)
O *Prompt Engineering* deve ser aplicado dentro do domínio de especialidade (Engenharia de Software) para que a linguagem e a complexidade sejam adequadas. 

**Onde ele impacta o fluxo de trabalho:**
* **Produtividade:** Atua como um catalisador em praticamente todas as etapas do ciclo de vida do software.
* **Desenvolvimento e Manutenção:** Escrita de código novo, refatoração e manutenção de sistemas legados.
* **Documentação Técnica:** Criação de *Design Docs* e documentações de sistemas.
* **Qualidade e Governança:** Condução de *Code Reviews* e auxílio no brainstorming de arquitetura.
* **Automação de Rotina:** Geração de mensagens de commit, automação de tarefas repetitivas e orientação de agentes de IA especializados (agentes de codificação).

### 3. Conclusão da Aula
O foco central é entender que o Prompt Engineering não é uma tarefa isolada, mas um recurso que garante a confiabilidade do software, desde a sua concepção inicial até a entrega em produção, aumentando a precisão das ferramentas de IA generativa.

## Aula 3: Outras Aplicabilidades do Prompt Engineering

Esta aula explora a dualidade do uso de prompts: como ferramenta de produtividade para o desenvolvedor e como componente estrutural em aplicações que integram IA.

---

### 1. IA como Ferramenta vs. IA como Produto
Existem dois cenários distintos para o uso de prompts:
* **Apoio ao Desenvolvimento:** Uso de prompts extensos e contextos ricos (muitos documentos) para gerar código e ganhar produtividade.
* **Integração em Aplicações:** Desenvolvimento de agentes ou sistemas multiagentes. Aqui, o foco muda para **prompts enxutos** visando reduzir custos de tokens e latência das requisições.

### 2. Definição de Escopo e Proatividade
O prompt é o que "doma" o agente de IA, definindo:
* **Comportamento e Transações:** Regras para tarefas críticas (ex: cálculo de juros baseado em histórico).
* **Nível de Autonomia:** Define se a IA deve ser proativa (tomar ações) ou apenas esclarecer dúvidas, evitando execuções indesejadas.
* **Acesso a Dados Externos:** Como a IA deve interagir com bancos de dados, APIs ou conectores (HubSpot, CRM, etc.).

### 3. Segurança e Modelos Especializados
* **Segurança e Privacidade:** Prompts mal estruturados podem permitir vazamento de dados sensíveis ou exploração de vulnerabilidades no sistema.
* **Modelos Menores (Small LLMs):** Em modelos com poucos parâmetros, a sensibilidade é maior; uma pequena alteração no prompt pode alterar drasticamente o comportamento do agente.

### 4. A Mudança de Paradigma: Programação Probabilística
* **A Nova Linguagem:** O Prompt Engineering é apresentado como a nova "linguagem de programação".
* **Determinístico vs. Probabilístico:** No desenvolvimento tradicional, as regras são definidas por `if/else` (determinístico). Na IA, o fluxo de decisão é baseado em probabilidades. O prompt substitui as regras rígidas de código para orientar o comportamento do sistema.

### 5. Ecossistema e Uso como Usuário Final
Aplicações para produtividade diária em plataformas como ChatGPT, Gemini e Claude:
* **Conectores e Ferramentas:** Uso de protocolos como MCP para ler planilhas, acessar calendários, Google Drive e gerar apresentações.
* **Intencionalidade:** Dominar as técnicas de prompt permite reduzir tarefas que levariam dias para poucos minutos, tornando o uso da IA mais assertivo e menos acidental.

## Aula 4: Técnicas e Tipos de Prompt

Esta seção da aula introduz a necessidade de entender que o *Prompt Engineering* é uma disciplina baseada em estudos científicos e benchmarks, e não apenas em "tentativa e erro" intuitiva.

---

### 1. O Mito do Detalhamento Excessivo
* **Qualidade vs. Quantidade:** Escrever um prompt não significa fornecer o máximo de detalhes possível. 
* **Ruído de Informação:** Fornecer mais dados do que o necessário pode confundir o modelo e comprometer a qualidade da resposta final.

### 2. Fundamentação Teórica e Benchmarks
* **Base Científica:** As técnicas de prompt (como as que serão vistas a seguir) são baseadas em *papers*, estatísticas e estudos de comportamento de modelos.
* **Impacto nos Resultados:** Os benchmarks dos modelos mostram claramente como diferentes formatos de instrução alteram a eficácia da IA em tarefas específicas.

### 3. Metodologia de Aprendizado
* **Foco na Prática:** O aprendizado não será apenas teórico. Envolverá:
    * Análise de código e execução de prompts em tempo real.
    * Uso de ferramentas como ChatGPT e ambientes de desenvolvimento.
    * Entendimento do que acontece "por baixo do capô" dos modelos.
* **Objetivo:** Visualizar como as variações nos prompts influenciam diretamente o comportamento e a geração de código.

### 4. Próximos Passos
* A introdução às técnicas específicas começará pelo **Role Prompting** (Prompt de Persona), uma das técnicas mais comuns, mas que exige entendimento técnico para ser usada com máxima eficiência.

## Aula 5: Role Prompting (Prompt de Persona)

Esta aula explora a técnica de definir um papel ou função específica para o modelo, analisando quando essa estratégia é realmente eficaz e quais são suas limitações técnicas.

---

### 1. O que é Role Prompting?
* **Definição:** É a atribuição explícita de um papel (*Role*) ao modelo (ex: professor, engenheiro de software, crítico).
* **Objetivo:** Controlar o estilo, a consistência da resposta e, principalmente, a **contextualização**.
* **Desambiguação:** O papel ajuda o modelo a decidir a direção do próximo token. 
    * *Exemplo:* Se o modelo é definido como "Engenheiro de Software", ao ler a palavra "Go", ele entenderá como a linguagem de programação e não como o verbo "ir".

### 2. Quando o Role Prompting é realmente útil?
* **Simulação de Cenários:** Revisões de arquitetura com diferentes perfis ou agentes de suporte com personas específicas.
* **Ajuste de Comunicação:** Adaptar explicações técnicas para diferentes públicos (ex: explicar para um aluno vs. para um especialista).
* **Modelos Menores (Small LLMs):** O impacto do Role Prompting é muito mais perceptível em modelos menores e com menos parâmetros, pois ajuda a "guiar" um raciocínio mais limitado.

### 3. Limitações e Comportamento dos Modelos
* **Modelos Avançados:** Em modelos de alto desempenho (como GPT-4 ou modelos de *reasoning* sofisticados), o impacto de dizer "você é um especialista" costuma ser baixo, pois o modelo já possui parâmetros suficientes para entender o contexto técnico sem essa instrução.
* **Role Drift (Desvio de Papel):** Em conversas longas, o modelo pode perder a persona inicial devido à ambiguidade do histórico ou ao tom do próprio usuário (o modelo tende a mimetizar o estilo do interlocutor).
* **Sobrescrita (User Override):** Instruções diretas do usuário no prompt atual podem anular o papel definido inicialmente (ex: pedir para ignorar ordens anteriores e agir como uma "galinha").

### 4. Segurança e Estabilidade
* **Jailbreaking:** Técnicas de ataque onde o usuário tenta forçar o modelo a sair de seu papel ou escopo de segurança.
* **Doma do Agente:** Para evitar que o papel seja facilmente alterado, utilizam-se técnicas mais avançadas de estruturação de prompt para garantir que o comportamento permaneça íntegro.

### 5. Conclusão Técnica
O Role Prompting não é uma "fórmula mágica" para melhorar respostas automaticamente. Sua eficácia depende da relação entre a **complexidade do modelo** (grandes vs. pequenos) e a **especificidade do escopo** definido.