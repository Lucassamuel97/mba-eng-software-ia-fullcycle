# Módulo Aprofundando na IA e LLM

## Sumário

- [Aula 1: Cadeias de Markov](#aula-1-cadeias-de-markov)


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
