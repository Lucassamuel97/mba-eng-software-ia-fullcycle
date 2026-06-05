# Módulo 2 - Estrutura básica de Prompts

## Sumário

- [Aula 1: Estruturação de Prompts](#aula-1-estruturação-de-prompts)




## Aula 1: Estruturação de Prompts

Esta aula apresenta a ideia central do módulo: o prompt deve ser tratado como um **artefato projetado**, cuja estrutura é definida pelo caso de uso, e não como um texto improvisado ou uma fórmula universal.

---

### 1. Prompt como artefato projetado
* **Não é improviso textual:** O prompt é um artefato **projetado** para induzir um comportamento específico do modelo.
* **Intenção > estética:** A qualidade da resposta depende menos de "escrever bonito" e mais de explicitar **intenção, contexto e formato esperado**.
* **Ajuste por caso de uso:** Quando tratado como projeto, o prompt é ajustado conforme a necessidade, em vez de reaproveitado como fórmula universal.

### 2. Não existe estratégia coringa
* **Nenhuma técnica resolve tudo:** Nenhuma estratégia de prompting entrega o melhor resultado em qualquer cenário.
* **Necessidades distintas:** Um workflow de desenvolvimento, um agente de atendimento, uma exploração de arquitetura e a geração de um documento exigem estruturas diferentes — pedem raciocínios, restrições e saídas distintas.
* **Consequência prática:** Escolher a estrutura do prompt **faz parte da solução**, não é um detalhe de implementação.

### 3. O caso de uso determina a estrutura
A estrutura adequada nasce do **objetivo operacional** do prompt:
* **Workflow de desenvolvimento:** privilegia etapas, critérios e continuidade entre interações.
* **Atendimento ao cliente final:** foca em robustez, interpretação de entradas variadas e consistência de resposta.
* **Exploração de arquitetura:** abre espaço para comparação e análise de alternativas.
* **Geração de documento:** prioriza formato, seções e completude.

### 4. Organização em partes e seções
* **Evita misturar intenções:** Separar o prompt em partes reduz a sobreposição de objetivos no mesmo pedido.
* **Seções típicas:** distinguir **objetivo, contexto, tarefa e saída esperada** torna a instrução mais legível para quem escreve e para o modelo.
* **Não é template fixo:** É uma forma de tornar **explícito** o que antes estava implícito e ambíguo.

### 5. Redução de ambiguidade e previsibilidade
* **Origem da ambiguidade:** Surge quando o prompt mistura exploração, análise e geração sem dizer qual tem prioridade.
* **Efeito da reorganização:** Em seções, fica claro **o que** o modelo deve fazer, **com base em qual** contexto e **em que formato** responder.
* **Resultado:** Maior previsibilidade — a resposta varia menos de forma indesejada e fica mais alinhada ao que o caso de uso pede.

### 6. Reconhecimento de padrões em prompts
* **Aprender observando:** Analisar prompts prontos ajuda a identificar padrões recorrentes de estruturação.
* **Não é cópia cega:** Serve para perceber como diferentes tipos de problema pedem diferentes arranjos de instruções.
* **Repertório:** Com o tempo, permite **diagnosticar prompts confusos** e reorganizá-los de forma mais intencional.