# Módulo Hands on 

## Sumário

- [Aula 1: Introdução à Geração de Respostas e Tokenização na Prática](#aula-1-introdução-à-geração-de-respostas-e-tokenização-na-prática)

- [Aula 2: Temperatura nas LLMs](#aula-2-temperatura-nas-llms)

- [Aula 3: Top K e Top P na Geração de Textos](#aula-3-top-k-e-top-p-na-geração-de-textos)

- [Aula 4: Arquiteturas e Tipos de Modelos (Foco em Encoder-Only)](#aula-4-arquiteturas-e-tipos-de-modelos-foco-em-encoder-only)

- [Aula 5: Arquiteturas e Tipos de Modelos (Foco em Decoder-Only)](#aula-5-arquiteturas-e-tipos-de-modelos-foco-em-decoder-only)

- [Aula 6: Aprofundando em Embeddings e Similaridade Vetorial](#aula-6-aprofundando-em-embeddings-e-similaridade-vetorial)

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


## Aula 3: Top K e Top P na Geração de Textos
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

## Aula 4: Arquiteturas e Tipos de Modelos (Foco em Encoder-Only)
- [ Sumário ](#sumário)

Esta aula avança para a categorização formal das arquiteturas de Inteligência Artificial baseadas em Transformers. A intenção é organizar os conceitos aprendidos até aqui e detalhar quando, como e por que utilizar cada tipo específico de modelo.

### 1. As Três Ramificações dos Transformers
Após o lançamento do artigo *"Attention is All You Need"* (Google, 2017), ficou claro que usar a arquitetura completa do Transformer (que une um *Encoder* e um *Decoder*) era muito "pesado" e ineficiente para certas tarefas. A partir daí, o mercado dividiu a arquitetura em três tipos principais:
1. **Encoder-Only** (Apenas o Codificador).
2. **Decoder-Only** (Apenas o Decodificador).
3. **Encoder-Decoder** (A arquitetura completa).

O foco desta aula é destrinchar os modelos **Encoder-Only**.

### 2. O que é a Arquitetura Encoder-Only?
A arquitetura Encoder-Only utiliza apenas a primeira metade do Transformer original. 
* **Objetivo Principal:** **Compreensão Textual profunda**, e não a geração/criação de textos. 
* **Como funciona:** Diferente do GPT (que tenta adivinhar sempre a *próxima* palavra olhando para o passado), o modelo Encoder-Only não prevê o próximo token. Ele utiliza a **Atenção Bidirecional**, ou seja, ele lê e analisa a frase inteira de uma só vez, olhando simultaneamente para a esquerda (passado) e para a direita (futuro) de cada palavra.
* **O Pioneiro:** O modelo **BERT** (*Bidirectional Encoder Representations from Transformers*), lançado pelo Google em 2018, foi o primeiro grande modelo desta categoria.

### 3. Como o Modelo é Treinado?
O treinamento foca em forçar a IA a entender o contexto como um todo:
* **Tokenização e Embeddings:** O texto vira vetor e cada palavra ganha um "carimbo" de posição.
* **Atenção Bidirecional:** A IA calcula o peso relativo de cada palavra em relação a todas as outras na frase.
* **MLM (Masked Language Model):** A principal técnica de treino. Esconde-se propositalmente algumas palavras no meio do texto (como um texto com lacunas), e o modelo deve deduzir qual é a palavra correta baseando-se no contexto geral (o que vem antes e depois da lacuna).

### 4. Casos de Uso (Quando usar Encoder-Only?)
Sempre que a tarefa for **classificar, extrair ou comparar dados**, os modelos Encoder-Only serão mais rápidos, precisos e baratos que modelos geradores como o ChatGPT.
* **Classificação de Sentimentos:** Analisar se os e-mails de clientes ou comentários em redes sociais sobre uma marca são positivos, neutros ou negativos.
* **NER (Named Entity Recognition):** Extrair e categorizar informações de um texto. (Ex: Em *"Steve Jobs fundou a Apple em Cupertino"*, a IA extrai e classifica Steve Jobs como *Pessoa*, Apple como *Organização* e Cupertino como *Local*).
* **Busca Semântica e Similaridade Vetorial:** Transformar documentos em vetores para buscar informações pelo *significado*, e não por palavras-chave exatas. (Ex: Entender matematicamente que as frases *"como fazer um bolo"* e *"receita para preparar um bolo"* possuem a mesma intenção).

### 5. A Família de Modelos BERT
Além do BERT original, surgiram variações otimizadas para diferentes necessidades:
* **RoBERTa:** Uma versão aprimorada e mais robusta, treinada com muito mais dados e processamento do que o BERT original.
* **DistilBERT:** Uma versão menor, mais rápida e mais barata, ideal para ambientes com restrição computacional (embora perca um pouco de precisão).
* **ALBERT:** Versão altamente eficiente que compartilha parâmetros internamente. É mais rápida na execução (*inferência*) e consegue resultados superiores, mas o treinamento é mais complexo.

## Aula 5: Arquiteturas e Tipos de Modelos (Foco em Decoder-Only)
- [ Sumário ](#sumário)

Nesta aula, avançamos no estudo das arquiteturas baseadas em Transformers, focando agora nos modelos **Decoder-Only** (Apenas Decodificador). Enquanto o modelo *Encoder* (visto na aula anterior) foca na compreensão global do texto, o *Decoder* abre mão de olhar para o "futuro" da frase para se tornar um especialista em **geração textual** (como o ChatGPT).

### 1. O Propósito do Decoder-Only
Ao isolar apenas a parte decodificadora do Transformer, reduz-se a complexidade arquitetural, tornando o modelo mais rápido, barato e extremamente eficiente para tarefas **autorregressivas** (prever a próxima palavra).
* **Casos de Uso Ideais:** Geração de *storytelling*, autocompletar textos e formulação de respostas conversacionais.

### 2. O Mecanismo Central: Causal Self-Attention
O grande segredo do Decoder-Only é a **Atenção Causal Autorregressiva**. O modelo é forçado a olhar apenas para o passado (os tokens que já foram gerados) para descobrir qual será a próxima palavra, sem nunca "espiar" o futuro. Para garantir isso matematicamente, utiliza-se a **Máscara Triangular Inferior**.

* **Como a Máscara Funciona:** Imagine uma matriz onde as linhas e colunas são os tokens (vetores) da frase *"o cachorro correu rápido"*. A máscara aplica "zeros" (cegueira) na parte superior direita da matriz e "checks/1s" (visibilidade) na parte inferior esquerda, formando um triângulo. 
* **O Efeito Prático:** Quando o modelo está tentando prever a palavra *"correu"*, ele só tem permissão matemática para enxergar *"o"* e *"cachorro"*. O resto da matriz fica oculto para evitar alucinações matemáticas e garantir a coerência preditiva.

> **💡 A Analogia do Aluno:** Imagine um aluno fazendo uma prova de 10 questões. Ao chegar na questão 6, ele tem permissão para ler e revisar as questões de 1 a 5 (seu passado/contexto) para focar na resposta da 6. Se ele lesse as questões 7 a 10 ao mesmo tempo, perderia o foco e a atenção na resposta atual. A máscara funciona como um "tampão visual" para manter o foco da IA apenas no passo atual.

### 3. Paralelismo sem Perder a Ordem (Substituindo a RNN)
Diferente das antigas Redes Neurais Recorrentes (RNNs) que liam palavra por palavra de forma lenta, o Transformer processa os dados em múltiplas camadas simultâneas (*multi-thread*). A máscara triangular, combinada com os carimbos de posição dos vetores, é o que garante que a máquina respeite a ordem cronológica do texto mesmo processando tudo paralelamente.

### 4. A Anatomia das Camadas Internas do Decoder
Cada bloco dentro do decodificador é dividido em três partes fundamentais para processar a informação:

1. **Multi-Head Attention com Máscara Causal:** O modelo avalia os tokens anteriores para decidir quais importam mais para prever o próximo. A técnica *Multi-Head* permite que o modelo olhe para essa mesma informação sob múltiplas perspectivas simultâneas (ex: uma "cabeça" analisa a gramática, a outra analisa o tom emocional).
2. **Feedforward Position-wise:** Atua como um "mini cérebro" individual para cada palavra. Ele capta padrões não-lineares e combina significados abstratos dos vetores.
3. **Layer Norm com Conexões Residuais:** * *Layer Norm:* Mantém os cálculos matemáticos equilibrados, impedindo que os números "explodam" ou "morram" (normalizando *outliers*).
   * *Conexões Residuais:* Transportam o aprendizado da camada anterior diretamente para a próxima, garantindo que o modelo não "esqueça" o contexto no meio do processamento.

## Aula 6: Aprofundando em Embeddings e Similaridade Vetorial
- [ Sumário ](#sumário)

Nesta aula prática, o objetivo é consolidar o conceito de **Embeddings**, mostrando como as máquinas transformam a semântica da linguagem humana em números para calcular a "distância" e a similaridade entre palavras e textos.

### 1. O que são Embeddings? (Revisão Conceitual)
As máquinas (LLMs) não compreendem o texto da mesma forma que os humanos. Enquanto nós usamos nossa vivência, gramática e neurônios para interpretar o significado de uma frase, as máquinas precisam de cálculos matemáticos.

* **A Transformação Vetorial:** Para a máquina entender o texto, ela transforma cada palavra (ou token) em um **Vetor** (uma representação numérica de múltiplas dimensões). 
* **O Papel do Embedding:** O *Embedding* é uma representação vetorial "densa". Ele não apenas armazena a palavra, mas embuti o seu **significado, contexto e relação** com outras palavras dentro de um espaço multidimensional.
* **Proximidade Semântica:** A grande mágica dos Embeddings é organizar esses vetores de forma que palavras com significados semelhantes fiquem "matematicamente próximas".
  * *Exemplo:* O vetor da palavra "gato" estará muito mais próximo, no espaço vetorial, do vetor de "cachorro" do que do vetor de "avião". Essa distância é geralmente calculada através de fórmulas matemáticas (como a distância Euclidiana ou a Similaridade de Cosseno).

### 2. Hands-on: Criando uma Calculadora de Similaridade (Hugging Face)
Para tangibilizar o conceito, a aula apresentou um laboratório prático na plataforma **Hugging Face**, criando um *Space* interativo para calcular a similaridade vetorial entre duas frases.

#### Passo a passo executado no Hugging Face:
1. Acessar a aba **Spaces** e criar um novo projeto (ex: `aula-emb2`).
2. Configurar o ambiente com a licença MIT e alocar o processamento (CPU gratuita).
3. **Criação do Código Fonte (`app.py`):**
   * Importação da biblioteca `Gradio` (para criar a interface visual) e da biblioteca de `Transformers`.
   * Escolha de um modelo pré-treinado do Hugging Face.
   * Criação de uma função matemática que:
     * Recebe duas frases como entrada.
     * Converte as frases em Embeddings (vetores).
     * Calcula a **Similaridade de Cosseno** entre os dois vetores.
     * Retorna um *score* numérico (de 0.0 a 1.0) representando o quão semanticamente próximas são as frases.
4. **Criação do `requirements.txt`:** Arquivo obrigatório para indicar ao servidor do Hugging Face quais bibliotecas instalar (neste caso, `gradio`).

### 3. Testes e Observações Práticas
Com a interface rodando, foram realizados testes inserindo frases para avaliar o comportamento matemático do Embedding:

* **Teste 1 (Semântica Próxima):**
  * Frase 1: *"Hoje cedo meu cachorro passou mal pois comeu algo que não deveria."*
  * Frase 2: *"Hoje cedo meu gato comeu algo estragado por isso não está bem."*
  * *Resultado:* Score de similaridade moderado/alto (0.69). Os animais e os sintomas são similares matematicamente.
* **Teste 2 (Mudança Sutil):**
  * Alterando "gato" para "cachorro" na Frase 2, o *score* aumenta consideravelmente, refletindo a aproximação vetorial exata do sujeito.
* **Teste 3 (Semântica Distante):**
  * Frase 1: *"Hoje cedo meu cachorro passou mal..."*
  * Frase 2: *"Meu avião precisa de um mecânico novo pois está com problemas no trem de pouso."*
  * *Resultado:* O *score* despenca drasticamente, pois os vetores de "avião/mecânico" estão extremamente distantes de "cachorro/passar mal" no espaço vetorial.
* **Teste 4 (O "Ponto Cego" da Máquina):**
  * Frase 1: *"Hoje cedo meu cachorro passou mal pois comeu algo que **não** deveria."*
  * Frase 2: *"Hoje cedo meu cachorro passou mal pois comeu algo que deveria."*
  * *Resultado:* A similaridade é quase de 100%. **Por quê?** Porque a máquina não "entende" a lógica da frase. Para o cálculo vetorial, a presença de quase todos os tokens exatos (cachorro, comer, passar mal) torna as frases matematicamente idênticas, mesmo que o significado prático/humano (o uso da negação "não") altere totalmente o sentido. Isso demonstra uma das limitações práticas de depender exclusivamente da similaridade de tokens sem análise profunda de