from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory

# Carrega variaveis de ambiente (ex.: GOOGLE_API_KEY).
load_dotenv()

# Define o prompt com historico + input do usuario.
prompt = ChatPromptTemplate.from_messages([
    ("system", "Voce e um assistente util. Responda em pt-br."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# Instancia o modelo Gemini.
chat_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.9)

# Encadeia prompt e modelo em um pipeline.
chain = prompt | chat_model

# Armazena historicos por sessao.
session_store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

# Adiciona memoria ao pipeline para manter contexto.
conversational_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# Identifica a sessao para recuperar o historico correto.
config = {"configurable": {"session_id": "demo-session"}}

# Interacoes de exemplo.
response1 = conversational_chain.invoke({"input": "Ola, meu nome e Samuca. Como voce esta?"}, config=config)
print("Assistente: ", response1.content)
print("-"*30)

response2 = conversational_chain.invoke({"input": "Voce pode repetir meu nome?"}, config=config)
print("Assistente: ", response2.content)
print("-"*30)

response3 = conversational_chain.invoke({"input": "Voce pode repetir meu nome em uma frase motivacional?"}, config=config)
print("Assistente: ", response3.content)
print("-"*30)