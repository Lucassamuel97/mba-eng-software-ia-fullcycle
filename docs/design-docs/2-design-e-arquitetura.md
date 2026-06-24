# Módulo 2 - Design e Arquitetura

## Sumário

- [Aula 1: Documentos de Design e Arquitetura](#aula-1-documentos-de-design-e-arquitetura)


## Aula 1: Documentos de Design e Arquitetura

Esta aula posiciona os **design docs** como o artefato que converte uma necessidade ainda difusa em uma **direção técnica compartilhada**. Eles reduzem ambiguidade ao iniciar projetos, implementar features ou evoluir produtos — mas só agregam valor quando o ganho de alinhamento supera o custo de produzir e manter o documento. A aula também reforça que a IA depende de **contexto confiável**: ela acelera a redação, mas não inventa o domínio.

---

### 1. Design docs como orientação de implementação
* **Transformam o difuso em direção:** Convertem uma necessidade ainda vaga em uma direção técnica compartilhada.
* **Reduzem ambiguidade:** Importam ao iniciar um projeto, implementar uma feature ou evoluir um produto existente.
* **Entendimento comum:** Em vez de cada pessoa assumir uma interpretação diferente, o documento explicita **o que** fazer e em **qual nível de detalhe**.
* **Crítico em arquitetura:** Decisões mal alinhadas geram retrabalho e inconsistência entre times.

### 2. Quando documentar e quando evitar excesso
* **Documentar tudo não é o objetivo:** Documentação útil não significa registrar tudo o tempo todo.
* **Depende do contexto:** Em alguns casos poucos documentos bastam; em outros, artefatos demais só adicionam redundância e custo de manutenção.
* **Variáveis de decisão:** Contexto organizacional, tipo de mudança e quantidade de pessoas e áreas envolvidas.
* **Critério:** Documentar faz sentido quando o **ganho de alinhamento supera o esforço** de produzir e manter.

### 3. Níveis de abstração e adequação ao público
* **Públicos diferentes, respostas diferentes:** Documentos existem em níveis distintos de abstração.
* **Alto nível × operacional:** Áreas estratégicas precisam de visão ampla; quem implementa precisa de decisões concretas.
* **Empresas maiores:** O desdobramento é mais forte — mais pontos de decisão, camadas organizacionais e necessidade de coordenação.
* **Qualidade real:** Depende menos do formato e mais da **adequação entre profundidade, audiência e objetivo**.

### 4. Documentação como ponte entre negócio e técnica
* **A necessidade chega fragmentada:** Raramente nasce pronta para implementação — vem implícita e distribuída entre pessoas.
* **O design doc é a ponte:** Converte esse contexto em uma decisão técnica compartilhável por produto, engenharia e liderança.
* **Mais que registrar solução:** Consolida **premissas, restrições e direção**.
* **Evita suposições isoladas:** Impede que a arquitetura seja definida pelo entendimento parcial de cada participante.

### 5. IA depende de contexto confiável
* **Não inventa o domínio:** A IA não recria corretamente o contexto específico da empresa, do domínio e da feature.
* **Apoia, mas depende de insumos:** Mesmo ajudando a estruturar texto e acelerar a redação, segue dependente de informações reais sobre necessidades e decisões.
* **Limitação principal:** Não é a capacidade de escrever, mas a **ausência de contexto confiável** para orientar o conteúdo.
* **Risco:** Sem contexto, o resultado pode parecer plausível e ainda estar desalinhado com o problema real.

### 6. Estrutura, prompts e entrevistas como apoio
* **Saem da página em branco:** Templates, prompts e entrevistas reduzem o esforço inicial e organizam a coleta de informação.
* **Não substituem o domínio:** Ajudam a transformar conhecimento implícito em perguntas, seções e decisões registráveis.
* **Orientam stakeholders:** Bem usados, tornam a documentação mais consistente entre projetos.
* **Onde está o ganho:** Na **estruturação do processo**, não na promessa de geração automática perfeita.

### 7. Documentação como ativo cumulativo reutilizável
* **Além do momento inicial:** Depois de produzidos, os documentos compõem o **contexto reutilizável** do ciclo de desenvolvimento.
* **Sustentam o futuro:** Ajudam novas pessoas a entender o que se constrói e embasam revisões, extensões e refinamentos arquiteturais.
* **A IA se beneficia:** Com documentos anteriores como contexto, ela apoia melhor as tarefas posteriores.
* **Ativo cumulativo:** Preserva raciocínio e **reduz perda de contexto** ao longo do tempo.
