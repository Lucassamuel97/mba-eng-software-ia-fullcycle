# Módulo 2 - Estrutura básica de Prompts

## Sumário

- [Aula 1: Estruturação de Prompts](#aula-1-estruturação-de-prompts)

- [Aula 2: Uma simples correção de Bug](#aula-2-uma-simples-correção-de-bug)

- [Aula 3: Minha IA produziu um Lixo](#aula-3-minha-ia-produziu-um-lixo)

- [Aula 4: Prompts no processo de desenvolvimento](#aula-4-prompts-no-processo-de-desenvolvimento)

- [Aula 5: Utilização em larga escala](#aula-5-utilização-em-larga-escala)


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

## Aula 2: Uma simples correção de Bug

Esta aula usa um cenário concreto de correção de bug para mostrar como a falta de limites no prompt gera **proatividade excessiva** do modelo e como a estruturação intencional (barreiras de comportamento, limites de escopo e saída definida) mantém a entrega previsível.

---

### 1. Proatividade excessiva do modelo
* **O problema:** Um prompt mal delimitado transforma uma correção simples em uma alteração ampla demais.
* **Sintomas:** O modelo corrige o bug pedido, mas **também** refatora funções, reescreve testes, adiciona documentação e aplica padrões em arquivos fora do objetivo.
* **Custo real:** Parece "tecnicamente melhor", mas piora o fluxo da equipe — uma PR que deveria validar uma troca de `>` por `>=` passa a exigir revisão extensa.

### 2. Impacto no code review e no workflow
* **Desvio de escopo:** O problema central não é a qualidade do código, e sim o desvio de escopo no processo de entrega.
* **Sobrecarga do revisor:** Mudanças espalhadas em vários arquivos transformam uma correção pontual em uma **refatoração implícita**, aumentando tempo, atrito e risco percebido.
* **Atraso de entrega:** Pode atrasar a entrada de um bugfix urgente em produção, mesmo sem introduzir defeitos. **Previsibilidade de revisão importa tanto quanto a correção.**

### 3. Ambiguidade operacional no prompt
* **Falha operacional:** A ambiguidade aqui aparece como falta de fronteiras, não só de clareza.
* **Exemplo problemático:** "Valide a função, entenda o problema e faça a correção" informa intenção, mas não impõe limites.
* **Efeito colateral:** Como o modelo tenta ser útil, interpreta o espaço aberto como **permissão** para melhorar o contexto ao redor.

### 4. Barreira de comportamento
* **Definição:** Parte do prompt que restringe **explicitamente** o que o modelo pode e não pode fazer.
* **Por que existe:** Contém a proatividade excessiva entre versões e famílias de modelos, cujo comportamento muda com o tempo.
* **Exemplo de barreira:** "Corrigir apenas o bug, não refatorar, não alterar arquivos fora do escopo, não reescrever testes além do necessário."

### 5. Limites do prompt
* **Definir antes, não corrigir depois:** Limitar é estabelecer o escopo **antes** da execução, não consertar o excesso após o fato.
* **Tipos de limite:** quantidade de arquivos, tipo de mudança permitida, proibição de melhorias oportunistas e foco exclusivo na causa do bug.
* **Objetivo:** Não é "melhorar o módulo", mas resolver uma condição errada sem expandir a intervenção. Quanto mais explícito o recorte, menor a chance de a PR crescer sem necessidade.

### 6. Formato e saída esperada
* **Contrato de entrega:** Definir a saída impede que o modelo invente a forma de resposta.
* **Exemplo de saída:** patch mínimo, lista de arquivos alterados, justificativa da mudança — e nada além disso.
* **Sem isso:** O modelo escolhe sozinho como responder e pode embutir mudanças extras por considerá-las úteis.

### 7. Variação entre versões de modelos
* **Iniciativa muda com a versão:** Mesmo com o prompt igual, modelos diferentes têm graus distintos de proatividade.
* **Centrado vs. agressivo:** Um modelo "centrado" executa só o pedido; outro tenta otimizar todo o código ao redor do bug.
* **Conclusão:** Um prompt suficiente em uma versão pode falhar em outra — a estruturação intencional reduz essa dependência do "temperamento" do modelo.

### 8. Estruturação intencional no caso de uso de desenvolvimento
* **A estrutura segue o caso de uso:** Correção de bug exige composição orientada a **controle de escopo** e **previsibilidade de revisão**.
* **Separar com clareza:** problema, limites de atuação e formato da resposta.
* **Redefinição de "utilidade":** Em desenvolvimento, útil não é maximizar mudanças positivas; é produzir a **menor alteração correta** para o objetivo atual.

### 9. Passo a passo prático para uma correção de bug
1. **Descreva o bug de forma localizável:** qual função está errada, qual condição falha e qual o comportamento esperado após a correção.
2. **Imponha a barreira de comportamento:** alterar apenas o necessário — sem refatorar, sem reorganizar código, sem expandir documentação, sem mexer em arquivos fora do escopo.
3. **Declare a saída esperada:** patch mínimo, arquivos alterados e explicação curta da correção.

> Essa combinação transforma um pedido aberto em uma **instrução operacional revisável**.

### 10. Consequência prática: PRs pequenas e previsíveis
* **Não é só estética:** PRs pequenas preservam a revisão rápida e reduzem o custo cognitivo do time.
* **Com limites + saída definida:** menor chance de mudanças laterais; o code review volta a refletir o objetivo original.
* **Ganho real:** Não elimina o comportamento probabilístico da IA, mas melhora a previsibilidade entre execuções e versões — em engenharia, isso vale mais do que uma refatoração não solicitada.

## Aula 3: Minha IA produziu um Lixo

Esta aula trata uma saída ruim da IA não como fim do processo, mas como **insumo de diagnóstico**. A partir dos erros, constrói-se uma base de conhecimento (guidelines, exemplos e restrições) que reduz reincidência e transforma o aprendizado local em ativo do projeto.

---

### 1. Saída ruim como insumo de projeto
* **Não encerrar na frustração:** Uma resposta ruim deve **iniciar um diagnóstico**, não terminar o trabalho.
* **Code Review da saída:** Mapear falhas concretas — bibliotecas antigas, arquivos grandes demais, funções desnecessárias, implementação abandonada no meio, quebra de padrões do projeto.
* **De reclamação a especificação:** O mapeamento transforma o "ficou um lixo" difuso em **problema observável e corrigível**.

### 2. A metáfora da IA como funcionário novo
* **Expectativa ajustada:** A IA é como um funcionário no primeiro dia — erra por não conhecer regras locais, convenções do time, versões de dependências e limites de escopo. Isso se repete a cada novo chat.
* **Muda o foco:** De "a ferramenta falhou" para "o processo de instrução ainda está incompleto".
* **Efeito prático:** Passa-se a **treinar o sistema com regras explícitas**, em vez de esperar alinhamento implícito.

### 3. Documentar erros para treinar prompts futuros
* **Erro recorrente vira instrução permanente:** Cada falha repetida pode se tornar uma regra fixa.
* **Exemplos:** Importou pacote obsoleto? Registre a regra correta, mostre o import errado e o certo, e exija consultar a documentação antes de escolher bibliotecas. Cria funções desnecessárias? Adicione verificação obrigatória de redundância após cada implementação.
* **Memória operacional:** Documentar reduz repetição de falhas e cria **memória fora do modelo**.

### 4. Guidelines como restrições reutilizáveis
* **Definição:** Regras curtas e acionáveis que orientam comportamento recorrente do modelo.
* **Exemplos:** limitar tamanho de arquivo, exigir funções limpas, pedir reaproveitamento de código existente, obrigar retomada de contexto após interrupção.
* **Diferencial:** Diferente de uma correção pontual no chat, a guideline é escrita para ser **reutilizada** — transforma aprendizado local em padrão de trabalho.

### 5. Few-shot no contexto do erro corrigido
* **Definição:** Fornecer **poucos exemplos** para induzir o padrão desejado.
* **Aplicação:** Mostrar explicitamente um import incorreto e outro correto, ou exemplificar a forma esperada de saída.
* **Por que funciona:** O modelo deixa de ter só uma regra abstrata e passa a ter **pares concretos de comparação** para imitar — útil quando certo/errado depende de convenções da stack ou do projeto.

### 6. Chain of Thought como encadeamento de verificação
* **Estrutura de passos:** Instruções como "após criar qualquer função, revise se ela é realmente necessária" forçam uma sequência operacional.
* **Foco aqui:** Menos como raciocínio interno do modelo e mais como roteiro: **implementar → revisar → comparar com o existente → manter**.
* **Resultado:** Reduz respostas impulsivas, aumenta autocorreção e contém a proliferação de código desnecessário.

### 7. Skeleton of Thoughts e especificação da saída
* **Definição:** Orientar a estrutura do pensamento/resposta por meio de um **esqueleto explícito**.
* **Combinação com exemplos:** Delimita não só o conteúdo, mas a **organização** da resposta.
* **Quando ajuda:** Quando a IA entende a tarefa, mas entrega em formato confuso ou difícil de revisar. Ganho principal: **previsibilidade estrutural**.

### 8. Estrutura clara evita que a base de prompts vire bagunça
* **Prompt é artefato projetado — a base também:** A base de conhecimento criada a partir dos erros precisa de organização.
* **Risco do crescimento:** Conforme guidelines, exemplos e restrições crescem, um arquivo desestruturado gera **mais ambiguidade**, não mais controle.
* **Mesmo cuidado dos testes:** clareza, separação de responsabilidades e manutenção contínua. Mais informação só ajuda quando está organizada.

### 9. Limites de comportamento aplicados ao fluxo real
* **Função de treinamento acumulativo:** Os limites de comportamento ganham aqui caráter permanente, não só correção do erro atual.
* **Exemplo:** Se a IA abandona um arquivo ao ser interrompida, a regra futura deve ser: terminar o arquivo atual antes de mudar de direção, ou corrigir e **retomar do ponto onde parou**.
* **Objetivo:** Reduzir bugs introduzidos pelo próprio fluxo conversacional — deixa de ser "responder melhor" e passa a ser **operar melhor dentro do workflow**.

### 10. Base de conhecimento de prompts como ativo
* **De improviso a ativo:** Erros documentados viram guidelines, exemplos e restrições reutilizáveis — um ativo do projeto.
* **Benefícios:** reduz dependência de memória individual, acelera onboarding de novos chats e melhora consistência entre desenvolvimento, agentes e geração de documentos.
* **Realismo:** Não elimina falhas (modelos continuam probabilísticos), mas **diminui reincidência** e encurta o caminho até uma saída útil. O ganho é acumulado ao longo do tempo, não em uma única interação.

## Aula 4: Prompts no processo de desenvolvimento

Esta aula muda o critério de otimização: em desenvolvimento, **contexto suficiente vale mais que economia de tokens**. O foco passa a ser recortar o codebase (branches/leafs, modularização, documentação local) para dar à IA contexto certo com fronteiras explícitas de navegação.

---

### 1. Não economizar tokens em desenvolvimento
* **Contexto > economia:** Em software, contexto costuma valer mais do que reduzir tokens.
* **Domínio interdependente:** Uma mudança pequena pode depender de convenções locais, fluxo entre módulos e restrições implícitas do projeto.
* **Risco de cortar cedo:** Reduzir contexto cedo demais faz a IA interpretar errado o problema ou propor alterações desconectadas. O critério é **"fornecer contexto suficiente para a tarefa"**, não "usar menos tokens".

### 2. Trade-offs entre custo, uso e previsibilidade
* **Custo ≠ só tokens:** Depende também do **modelo de cobrança** da ferramenta.
* **Consumo percebido:** Em IDEs e assistentes integrados, o custo aparece como plano de uso, limite operacional ou assinatura — não como contagem por token.
* **Decisão prática:** Em vez de otimizar cada token, comparar **previsibilidade de custo, produtividade e qualidade**. A escolha certa é a que sustenta o workflow com menos surpresa operacional.

### 3. Comparação entre planos e ferramentas
* **Mesmos resultados, custos distintos:** Ferramentas diferentes entregam resultados parecidos com estruturas de custo muito diferentes.
* **API vs. plano fixo:** API pode ficar cara em uso intensivo; plano fixo dá previsibilidade financeira mesmo sem ser barato em valor absoluto.
* **Entrada gratuita/limites generosos:** úteis para experimentação e validação antes de escalar.
* **Comparação relevante:** Não "qual é melhor em abstrato", mas **"qual combinação de custo, limite e qualidade atende este time agora"**.

### 4. Contexto necessário para a tarefa
* **Não exponha tudo:** Nem toda tarefa exige o codebase inteiro.
* **Recorte seguro:** Entregar o contexto necessário sem abrir escopo desnecessário — indicar onde começa a investigação, quais módulos importam, qual fluxo seguir e o que ignorar.
* **Ganho:** Estruturar esse recorte melhora a previsibilidade ao reduzir liberdade de navegação e interpretação.

### 5. Branches e leafs no codebase
* **Codebase como árvore:** Orienta a IA por ramificações, em vez de despejar todos os arquivos de uma vez.
* **Branches vs. leafs:** *Branches* são caminhos/agrupamentos de componentes; *leafs* são pontos terminais, onde a implementação concreta costuma estar.
* **Tarefa localizada:** Instruir a IA a seguir apenas uma ramificação específica até os leafs relevantes — restringe a busca, reduz ruído e evita contaminar a solução com partes irrelevantes.

### 6. Modularização para restringir contexto
* **Caso de uso determina a estrutura:** Em desenvolvimento, isso significa **modularizar também o contexto** fornecido à IA.
* **Isolamento:** Componentes desacoplados permitem trabalhar em um pedaço do sistema sem exigir compreensão global.
* **Módulo como mecanismo de prompting:** Quanto melhor o módulo delimita responsabilidades, mais fácil instruir a IA a permanecer **naquela bolha**.

### 7. Documentação local e instruções por módulo
* **Menos ambiguidade no recorte:** Documentação e instruções próprias por módulo permitem à IA operar com mais segurança dentro do escopo.
* **O que registrar:** fluxo esperado, limites do módulo, convenções e pontos de integração permitidos.
* **Contexto distribuído:** Em vez de um prompt central gigantesco para tudo, distribui-se contexto útil ao longo da estrutura do projeto — impedindo que o modelo "escape" para outras áreas.

### 8. Estruturação para previsibilidade
* **Vem da organização, não só do texto:** A previsibilidade depende de organizar sistema e contexto para que a IA saiba **por onde começar, até onde ir e o que ignorar**.
* **Camada operacional do controle de escopo:** Além de limitar comportamento, limita-se a **superfície de leitura** do codebase.
* **Prompt previsível:** combina contexto suficiente com **fronteiras explícitas de navegação** — reduz respostas dispersas e melhora aderência ao fluxo real de engenharia.

### 9. Aplicação prática no workflow
1. **Indique o ponto de entrada:** módulo de entrada, a ramificação relevante e o fluxo a seguir até os arquivos terminais.
2. **Anexe só o local:** se o sistema for modular, junte apenas a documentação e instruções daquele trecho, deixando explícito que o restante deve ser ignorado.
3. **Defina a saída:** peça o formato adequado ao workflow já definido, mantendo escopo e revisão controláveis.

> A combinação entre **contexto suficiente** e **recorte estrutural** é o que torna o uso da IA viável em codebases grandes.

## Aula 5: Utilização em larga escala

Esta aula contrasta o **uso pontual** (onde contexto amplo é desejável) com o **uso em larga escala** (onde economia de tokens vira requisito de viabilidade). Em produção, o tamanho do prompt deixa de ser só decisão técnica e passa a ser **decisão arquitetural, financeira e de produto**.

---

### 1. Uso pontual vs. uso em larga escala
* **Não há estrutura única:** A mesma estrutura de prompt não serve igualmente a todos os cenários.
* **Uso pontual:** Em desenvolvimento ou exploração, o objetivo é maximizar entendimento e qualidade, mesmo com contexto amplo.
* **Restrição que muda:** Em uso pontual, a restrição principal é **não estourar a janela de contexto** — não reduzir o prompt a qualquer custo. O critério muda quando a interação é multiplicada por volume.

### 2. Processos de exploração aceitam contexto maior
* **O que são:** Cenários em que o usuário ainda descobre o problema, o espaço de solução ou o que precisa perguntar.
* **Mais contexto faz sentido:** Permitir que a IA pesquise no código-fonte e na internet para produzir algo contextualizado (documentação, entendimento técnico).
* **Por que tolera prompts pesados:** Não roda milhares de vezes em paralelo — o custo extra existe, mas não domina a decisão quando a execução é esporádica.

### 3. Economia de tokens como requisito de escala
* **De otimização a viabilidade:** Em produção em larga escala, economizar tokens deixa de ser otimização e vira **requisito de viabilidade**.
* **Efeito multiplicador:** Cada token de entrada/saída é multiplicado pelo volume de chamadas — um excesso pequeno por requisição vira custo mensal capaz de inviabilizar o produto.
* **Decisão arquitetural:** Um prompt excelente em uso humano pontual pode ser **economicamente ruim** quando executado milhares ou milhões de vezes.

### 4. Latência cresce com prompts mais pesados
* **Causas:** Prompts longos e instruções que exigem mais processamento (pensar passo a passo, revisar etapas) aumentam a latência.
* **Impacto em produção:** Afeta tempo de resposta percebido, throughput do sistema e necessidade de infraestrutura.
* **Atraso acumulado:** O problema não é esperar alguns segundos a mais, mas **acumular atraso** em cada chamada de um fluxo repetido em escala. Sofisticação de prompt é custo operacional de **tempo**, não só de tokens.

### 5. API por token muda a lógica de decisão
* **Assinatura esconde o custo marginal:** Em ferramentas pagas por assinatura, o custo de cada interação parece invisível.
* **API torna explícito:** A cobrança por milhão de tokens de entrada/saída revela quanto cada requisição consome, forçando mais disciplina no desenho do prompt.
* **Prompt como unidade de custo:** Reduzir verbosidade, instruções redundantes e contexto desnecessário passa a ser **parte do design do sistema**.

### 6. Trade-off entre custo, latência e qualidade
* **Não dá para maximizar tudo:** Em escala, busca-se o ponto em que a **qualidade é suficiente** para o caso de uso sem empurrar custo e latência a níveis desproporcionais.
* **A pergunta certa:** Não é se um prompt maior melhora um pouco a resposta, mas se essa melhora **justifica o impacto** quando multiplicada pelo volume.
* **Exemplo:** Um ganho marginal de assertividade pode não compensar milhares de dólares extras por mês.

### 7. Decisão de produto, não só de engenharia
* **Variáveis de produto:** Custo, latência e qualidade devem ser avaliados como variáveis de produto, não só técnicas.
* **Quando o caso de uso deixa de fazer sentido:** Se a resposta é lenta demais, cara demais ou melhora pouco frente à alternativa enxuta, o problema é **econômico**, não só técnico.
* **Escalar IA com viabilidade:** Projetar prompts para produção exige pensar em sustentabilidade do serviço, experiência do usuário e margem de operação. **Escalar IA não é só fazer funcionar; é fazer funcionar de modo viável.**