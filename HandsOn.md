# Módulo Hands on 

## Sumário

- [Aula 1: Introdução à Geração de Respostas e Tokenização na Prática](#aula-1-introdução-à-geração-de-respostas-e-tokenização-na-prática)
- [Aula 2: Top K e Top P na Geração de Textos](#aula-2-top-k-e-top-p-na-geração-de-textos)


## Aula 1: Introdução à Geração de Respostas e Tokenização na Prática

Esta aula marca o início de um novo módulo. Após compreender os conceitos fundamentais, a arquitetura e os desafios das LLMs, o foco agora é entender **como** a máquina gera suas respostas e como podemos manipulá-la (tunar seus parâmetros) para obter o resultado exato que precisamos.

### 1. Objetivos do Novo Módulo (O "Motor" da LLM)
O objetivo principal é deixar de ser apenas um usuário comum e passar a dominar os bastidores da geração de texto. Os próximos tópicos do curso irão abordar:

* **Mecânica de Geração:** Entender como ocorre a inferência, a predição de tokens (um a um) e o uso do contexto para gerar respostas coerentes.
* **Ajuste de Parâmetros (Tuning):** Aprender a configurar os "botões" do motor da LLM para controlar estilo, criatividade e nível de alucinação. Você aprenderá a manipular configurações críticas como:
  * *Temperatura (Temperature)*
  * *Top K*
  * *Top P*
* **Controle de Tom e Aplicação:** Como forçar a IA a ser técnica, empática ou informal, e como aplicar esse conhecimento de forma avançada no cotidiano (resumos, códigos, e-mails, chatbots corporativos).

### 2. Hands-on: Entendendo a Tokenização na Prática
Para ilustrar a teoria, a aula apresentou uma demonstração prática utilizando a ferramenta oficial da OpenAI: **[Tokenizer](https://platform.openai.com/tokenizer)**.

#### O que é a ferramenta Tokenizer?
É uma "calculadora" visual que permite digitar textos e ver em tempo real como diferentes modelos da OpenAI quebram as palavras em fragmentos (*tokens*) para processamento vetorial.

#### Observações Práticas da Demonstração:
1. **Palavras Comuns vs. Palavras Raras:**
   * Palavras extremamente comuns no treinamento (ex: *"cachorro"*, *"praça"*) costumam ser processadas como **1 único token** nos modelos mais modernos.
   * Palavras menos frequentes, complexas ou muito longas (ex: *"paralelepípedo"*) são "fatiadas" em vários pedaços menores (ex: *par-ale-lep-ip-edo*), pois é matematicamente mais fácil para a IA calcular a probabilidade dessas sílabas separadas do que da palavra inteira.
2. **A Evolução dos Modelos:**
   * Ao comparar diferentes versões de motores na ferramenta (GPT-3 Legacy vs. GPT-4), nota-se que os modelos mais antigos quebram as palavras em muito mais pedaços do que os modelos recentes.
   * **Por quê?** Modelos mais modernos (como o GPT-4o) possuem um vocabulário de tokens de treinamento muito maior e mais refinado, permitindo agrupar palavras maiores em um único token, o que melhora a eficiência e a precisão do modelo.
3. **Custo e Espaços:**
   * Até mesmo os espaços em branco e a pontuação são contabilizados como tokens.
   * Entender essa quebra é fundamental, pois, em usos comerciais (API), **você paga por token gerado e consumido**.

## Aula 2: Temperatura nas LLMs

Nesta aula, iniciamos o aprofundamento prático nos parâmetros que controlam o comportamento de uma LLM, focando no mais conhecido deles: a **Temperatura**.

### 1. O que é Temperatura?
Apesar do nome, a Temperatura nas LLMs não tem relação com clima ou aquecimento de hardware (GPUs). Ela é um parâmetro matemático (uma configuração) que controla o **nível de aleatoriedade e criatividade** na predição do próximo token (palavra) que a Inteligência Artificial vai gerar.

Lembre-se: as LLMs são probabilísticas. Elas não decidem qual é a palavra *certa*, mas sim qual é a *mais provável*. A Temperatura ajusta o quão "rígida" ou "livre" será essa escolha.

### 2. A Escala de Temperatura (0.0 a 1.0)
A Temperatura funciona, na maioria das plataformas, em uma escala que vai de 0.0 (ou 0.1) até 1.0. 

#### A. Baixa Temperatura (0.1 a 0.3)
* **Comportamento:** O modelo torna-se extremamente rígido, previsível e conservador. Ele reduzirá a aleatoriedade ao máximo, escolhendo sempre os tokens com a maior probabilidade estatística matemática.
* **Casos de Uso Ideais:** Textos que exigem precisão técnica e factualidade, como:
  * Contratos jurídicos.
  * Códigos de programação.
  * Bulas médicas.
  * Respostas factuais e análises de dados.

#### B. Alta Temperatura (0.7 a 1.0)
* **Comportamento:** O modelo ganha mais "liberdade" para explorar a cauda longa das probabilidades estatísticas. Ele vai ignorar escolhas óbvias e buscar tokens menos comuns, tornando a resposta mais fluída e imprevisível.
* **Casos de Uso Ideais:** Tarefas criativas, como:
  * *Brainstorming* de ideias.
  * Escrita de poemas, contos ou livros.
  * Criação de letras de música.
  * Textos de marketing persuasivos.

> **💡 A Analogia da Biblioteca:** Imagine pedir um livro sobre Direito para um especialista em uma biblioteca. Com temperatura 0.1, ele vai direto na prateleira de Direito e traz um *Vade Mecum*. Com temperatura 1.0, a criatividade dele aumenta, e ele pode trazer o *Vade Mecum*, mas junto trará um livro do Harry Potter para tentar fazer uma correlação criativa e inusitada sobre as leis da magia.


## Aula 2: Top K e Top P na Geração de Textos
- [ Sumário ](#sumário)

Esta aula aprofunda o entendimento sobre como as LLMs escolhem as palavras durante a geração de texto. Exploramos as técnicas de amostragem estatística **Top K** e **Top P**, que inserem o grau de aleatoriedade necessário para evitar que as respostas fiquem robóticas, repetitivas ou sem sentido.

### 1. A Necessidade de Aleatoriedade na Geração de Texto
As LLMs funcionam prevendo o próximo token (palavra). No entanto, o modelo não escolhe automaticamente a palavra com a maior probabilidade todas as vezes.
* **O Problema da Previsibilidade:** Se a IA sempre escolhesse a palavra número 1 do ranking de probabilidade, o texto gerado seria excessivamente previsível, mecânico (semelhante a chatbots antigos baseados em regras) e pobre em criatividade.
* **A Solução:** Para soar de forma fluida e humana, as LLMs aplicam um grau de aleatoriedade, sorteando a próxima palavra dentro de um grupo restrito de tokens altamente prováveis. Os parâmetros que controlam o tamanho desse grupo são o Top K e o Top P.

### 2. O que é o Top K Sampling?
O Top K é uma técnica estatística que limita a escolha do próximo token a um **número fixo (K)** de opções mais prováveis.

* **Como Funciona:** O modelo calcula as probabilidades para todas as palavras do vocabulário, as ordena da maior para a menor probabilidade e "corta" a lista no número limite K definido pelo usuário.
* **Exemplo Prático:** Para prever a próxima palavra da frase *"O sol está brilhando no..."* com um **Top K definido como 5**, a IA cria o seguinte ranking:
  1. Céu (35%)
  2. Horizonte (25%)
  3. Mar (15%)
  4. Campo (10%)
  5. Deserto (8%)
  * *Observação:* Se a 6ª palavra for "Luz" (4%), ela é sumariamente descartada. A LLM utilizará a sua "temperatura" para decidir a próxima palavra **apenas** entre essas 5 opções do topo.

### 3. O que é o Top P Sampling (Nucleus Sampling)?
Diferente do Top K (que limita a *quantidade* de palavras), o Top P usa como limite uma **soma percentual (P)** da probabilidade total. O número de palavras consideradas será dinâmico.

* **Como Funciona:** O modelo desce pela lista ordenada de tokens, somando suas respectivas probabilidades até que o valor acumulado atinja o limite `P` estipulado (ex: 0.9, que representa 90%). 
* **Exemplo Prático:** Utilizando o mesmo contexto anterior com um **Top P definido em 0.9 (90%)**:
  * Céu (35%)
  * Horizonte (25%) -> *Soma parcial acumulada: 60%*
  * Mar (15%) -> *Soma parcial acumulada: 75%*
  * Campo (8%) -> *Soma parcial acumulada: 83%*
  * Deserto (7%) -> *Soma parcial acumulada: 90%*
  * *Observação:* Neste cenário, as opções viáveis se encerram em "Deserto", pois a meta somada de 90% das probabilidades foi exatamente atingida. Qualquer token que vier abaixo (e que represente os 10% restantes da distribuição total) será ignorado pelo motor na tomada de decisão.