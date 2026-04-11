# Módulo Aprofundando na IA e LLM

## Sumário

- [Aula 1: Cadeias de Markov](#aula-1-cadeias-de-markov)
- [Aula 2: Redes Neurais Profundas e a Evolução do NLP](#aula-2-redes-neurais-profundas-e-a-evolução-do-nlp)


## Aula 1: Cadeias de Markov

### Resumo Expresso: Modelos Ocultos de Markov (HMM)

Para sair da limitação dos bigramas (que só olhavam pares de palavras), a IA precisou de uma evolução matemática introduzida por Andrei Markov: a capacidade de inferir contextos invisíveis através dos Modelos Ocultos de Markov (HMM).

### O Grande Diferencial: O "Estado Oculto"

A grande sacada do HMM é que ele não olha apenas para os dados diretos. Ele entende que existe uma estrutura interna (invisível) que dita o que está acontecendo na superfície (visível).

**A analogia do clima:** Um meteorologista não precisa "ver" o sistema atmosférico para saber o clima. Ele observa sinais indiretos (céu limpo, vento, guarda-chuvas abertos) e, com base nisso, infere o Estado Oculto (sol, chuva ou nublado).

### Como o HMM Funciona (As 4 Engrenagens)

- **Estados ocultos:** O contexto que não vemos diretamente (ex.: a regra gramatical ou o clima real).
- **Observações (outputs):** Os dados visíveis que temos em mãos (ex.: as palavras faladas ou o céu cinza).
- **Matriz de transição:** A probabilidade de pular de um estado para outro (ex.: qual a chance de sair de "sol" para "chuva"?).
- **Matriz de emissão:** A probabilidade de um evento acontecer dado um estado (ex.: se está chovendo, qual a chance de eu ver um guarda-chuva aberto?).

### A "Matemática" por Trás (Os 3 Algoritmos)

Para fazer o HMM rodar, usam-se cálculos específicos:

- **Forward-Backward:** Calcula qual é a probabilidade de uma sequência de eventos acontecer.
- **Viterbi:** Acha o caminho mais lógico/provável dos estados ocultos.
- **Baum-Welch:** Ajusta e treina o modelo para que ele aprenda com os dados.

### Onde o HMM Brilha (Aplicações Práticas)

Como o modelo é bom em analisar sequências curtas e probabilidade, ele ajudou a criar:

- **Reconhecimento de fala:** O áudio é a observação; o fonema (som real) é o estado oculto.
- **Gramática (POS tagging):** A palavra é a observação; se ela é um verbo ou substantivo é o estado oculto.
- **Biologia (DNA):** Previsão de sequências genéticas.

### Onde o HMM Falha (As Limitações)

Apesar de ser genial, o modelo de Markov ainda não era suficiente para criar ferramentas como o ChatGPT devido a duas falhas graves:

- **Linearidade e amnésia (o paradigma de Markov):** A regra de Markov diz que o momento atual só depende do momento imediatamente anterior. Ou seja, ele é uma linha reta sem memória de longo prazo. Ele não consegue cruzar informações complexas, voltar atrás e conectar o final de um texto com o primeiro parágrafo.
- **Custo de treinamento:** Exige dados estritamente rotulados (treinamento supervisionado), o que torna o processo muito caro e lento.

## Aula 2: Redes Neurais Profundas e a Evolução do NLP

### O Ponto de Virada da Inteligência Artificial
A transição de modelos simples (como bigramas e Cadeias de Markov) para as **Redes Neurais Profundas** marcou o verdadeiro "boom" da IA moderna. Dois fatores principais possibilitaram esse avanço a partir da década de 1990:
* **Disponibilidade Massiva de Dados:** A popularização da internet, redes sociais e sistemas digitais gerou a matéria-prima essencial para treinar as IAs em larga escala.
* **Aumento do Poder Computacional:** A chegada e o uso de GPUs potentes permitiram o processamento viável de algoritmos muito mais complexos.

---

### Redes Neurais Profundas
Diferente dos modelos anteriores, onde o caminho de decisão era curto e claro, as redes neurais profundas simulam o comportamento e as sinapses do cérebro humano.
* O termo **"profundo"** refere-se à vasta quantidade de camadas que a informação percorre.
* Um dado de entrada passa por uma cadeia gigantesca de transformações, combinações e reinterpretações até gerar uma saída coerente.

---

### Processamento de Linguagem e "Word Embeddings"
Redes neurais não processam palavras de forma direta, nem usam a conversão literal de letras para bits (0 e 1). O segredo para a IA entender a linguagem natural está nos **Embeddings**.
* **O que são:** É uma representação vetorial de palavras em um espaço contínuo de alta dimensão.
* Na prática, uma palavra deixa de ser um texto e vira um vetor matemático (ex: `0.25, -0.14, 0.88`).
* **A vantagem:** Palavras com significados e contextos semelhantes geram vetores com valores próximos. Isso permite que a IA faça correlações matemáticas para entender a linguagem (base para modelos como GloVe, BERT e GPT).

---

### Redes Feedforward e Janelas de Contexto
Para tentar prever a próxima palavra em uma frase, a IA começou a usar redes *feedforward* (fluxo unidirecional de dados, apenas para a frente) unidas a uma estratégia de **Janelas de Contexto Fixas**.
* O modelo analisa um grupo fixo de palavras anteriores para prever a próxima.
* Exemplo em uma janela de 4 palavras: Ao processar os vetores de "O", "cachorro", "está", "no", a rede calcula a maior probabilidade e prevê que a palavra 5 seja "jardim".

---

### Limitações Críticas Desse Modelo
Apesar do grande salto tecnológico, essas estruturas esbarraram em problemas que impediam a IA de ter diálogos complexos:
* **Falta de Memória:** O modelo só conhece o que está processando naquele instante. Ele não serve para manter a coerência ao longo de um diálogo longo, um grande resumo ou uma tradução de texto completo.
* **Contexto Limitado (Janela Fixa):** Como a janela de leitura olha apenas para um número fixo de palavras passadas, a IA perde o contexto do que foi dito no início de um parágrafo longo.
* **Problema de Ordem/Sequência temporal:** Como os vetores são frequentemente somados ou concatenados, a rede tem dificuldade em distinguir o papel do sujeito e do objeto. Para ela, "O cachorro mordeu o homem" e "O homem mordeu o cachorro" podem gerar resultados matemáticos tão próximos que a IA perde a semântica real da ação.

---

### O Próximo Passo
A necessidade de superar a falta de memória e a dependência de ordem prepara o terreno para o surgimento de uma nova arquitetura revolucionária: os **Transformers** (a base das IAs generativas atuais).