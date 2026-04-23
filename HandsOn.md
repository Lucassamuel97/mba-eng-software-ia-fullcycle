# Módulo Hands on 

## Sumário

- [Aula 1: Introdução à Geração de Respostas e Tokenização na Prática](#aula-1-introdução-à-geração-de-respostas-e-tokenização-na-prática)



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