# Módulo 4 - Architecture Decision Record (ADR)

## Sumário

- [Aula 1: Introdução a ADRs](#aula-1-introdução-a-adrs)


## Aula 1: Introdução a ADRs

Esta aula apresenta o **ADR (Architecture Decision Record)** como o documento que registra **decisões arquiteturais e, sobretudo, o porquê** delas. O código mostra o "o quê"; o ADR preserva restrições, trade-offs e pressões que motivaram a escolha — virando **memória técnica explícita** que sobrevive ao tempo e à rotatividade do time. Em ambientes com IA, esse registro deixa o modelo trabalhar sobre **contexto declarado** em vez de inferir motivos a partir do código.

---

### 1. ADR como registro do porquê arquitetural
* **Registra decisão + motivo:** Documenta decisões arquiteturais relevantes e, principalmente, **por que** foram tomadas.
* **Código não basta:** Ele mostra o que foi implementado, mas raramente preserva restrições, trade-offs e pressões que levaram à escolha.
* **Memória técnica explícita:** Transforma o raciocínio em registro acessível mesmo anos depois.
* **Evita tradição oral:** Reduz a perda de contexto e impede que decisões importantes virem só lembrança do time.

### 2. Memória técnica explícita e continuidade ao longo do tempo
* **Sistemas duram, pessoas rotacionam:** O projeto evolui por anos enquanto gente entra e sai.
* **Risco da memória individual:** Depender da lembrança de poucos gera interpretações incompletas, retrabalho e reversões mal informadas.
* **Continuidade do raciocínio:** Registrar preserva a escolha final **e** o porquê de ela ter feito sentido naquele momento.
* **Valor crescente:** Aumenta conforme a vida útil do sistema cresce.

### 3. Valor histórico da documentação arquitetural
* **Timeline arquitetural:** O histórico de decisões mostra a arquitetura como uma **sequência de escolhas**, não uma fotografia estática.
* **O que ajuda a entender:** Por que partes do sistema mudaram, quais direções foram priorizadas e quais consequências já eram conhecidas.
* **Manutenção mais segura:** Documentar o percurso torna evolução e manutenção menos arriscadas.

### 4. Quando ADR se torna especialmente útil
* **Impacto estrutural:** Mais valioso quando a decisão afeta múltiplas partes do sistema ou tende a gerar dúvidas no futuro.
* **Casos típicos:** Mudanças arquiteturais relevantes, escolhas com trade-offs importantes e decisões cujo motivo não fica evidente no código.
* **Nem tudo vira ADR:** Decisões operacionais corriqueiras não precisam — o foco é o que influencia a arquitetura.
* **Critério central:** Se **esquecer o porquê** prejudica a evolução do sistema, vale registrar.

### 5. Relação entre ADR e IA
* **IA opera melhor com contexto explícito:** Modelos rendem mais com registro claro do que com artefatos dispersos.
* **O que a IA faz com ADRs:** Recupera intenção arquitetural, relaciona mudanças ao longo do tempo e apoia análise, geração e revisão.
* **Sem registro:** A IA infere motivos a partir do código e de sinais indiretos — mais frágil.
* **Com memória documentada:** A IA deixa de **adivinhar** e passa a trabalhar sobre **contexto declarado**.
