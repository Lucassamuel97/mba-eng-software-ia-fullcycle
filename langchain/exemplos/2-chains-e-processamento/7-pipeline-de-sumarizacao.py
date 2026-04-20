from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

# Carrega variáveis de ambiente do arquivo .env (ex.: chave da API do Gemini).
load_dotenv()

# Texto longo de exemplo que será dividido e resumido pelo pipeline.
long_text = """Dawn threads a pale gold through the alley of glass.
The city yawns in a chorus of brakes and distant sirens.
Windows blink awake, one by one, like sleepy eyes.
Streetcloth of steam curls from manholes, a quiet river.
Coffee steam spirals above a newspaper's pale print.
Pedestrians sketch light on sidewalks, hurried, loud with umbrellas.
Buses swallow the morning with their loud yawns.
A sparrow perches on a steel beam, surveying the grid.
The subway sighs somewhere underground, a heartbeat rising.
Neon still glows in the corners where night refused to retire.
A cyclist cuts through the chorus, bright with chrome and momentum.
The city clears its throat, the air turning a little less electric.
Shoes hiss on concrete, a thousand small verbs of arriving.
Dawn keeps its promises in the quiet rhythm of a waking metropolis.
The morning light cascades through towering windows of steel and glass,
casting geometric shadows on busy streets below.
Traffic flows like rivers of metal and light,
while pedestrians weave through crosswalks with purpose.
Coffee shops exhale warmth and the aroma of fresh bread,
as commuters clutch their cups like talismans against the cold.
Street vendors call out in a symphony of languages,
their voices mixing with the distant hum of construction.
Pigeons dance between the feet of hurried workers,
finding crumbs of breakfast pastries on concrete sidewalks.
The city breathes in rhythm with a million heartbeats,
each person carrying dreams and deadlines in equal measure.
Skyscrapers reach toward clouds that drift like cotton,
while far below, subway trains rumble through tunnels.
This urban orchestra plays from dawn until dusk,
a endless song of ambition, struggle, and hope."""

# Divide o texto em partes menores para facilitar a sumarização em etapas.
spliter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
parts = spliter.create_documents([long_text])

# Descomente para inspecionar os chunks gerados pelo splitter.
# for part in parts:
#     print(part.page_content)
#     print("-" * 10)

# Inicializa o modelo de chat que será usado nas fases map e reduce.
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# Etapa MAP: cria um resumo curto para cada chunk individualmente.
map_prompt = PromptTemplate.from_template("Write a concise summary of the following text:\n{context}")
map_chain = map_prompt | llm | StrOutputParser()

# Converte a lista de documentos no formato de entrada esperado pelo map_chain.
prepare_map_inputs = RunnableLambda(lambda docs: [{"context": d.page_content} for d in docs])
map_stage = prepare_map_inputs | map_chain.map()

# Etapa REDUCE: combina todos os resumos parciais em um resumo final.
reduce_prompt = PromptTemplate.from_template("Combine the following summaries into a single concise summary:\n{context}")
reduce_chain = reduce_prompt | llm | StrOutputParser()

# Junta as saídas da etapa map em um único campo de contexto para a etapa reduce.
prepare_reduce_input = RunnableLambda(lambda summaries: {"context": "\n".join(summaries)})

# Encadeia as etapas map -> preparação do reduce -> reduce final.
pipeline = map_stage | prepare_reduce_input | reduce_chain

# Executa o pipeline completo e imprime o resumo consolidado.
result = pipeline.invoke(parts)
print(result)