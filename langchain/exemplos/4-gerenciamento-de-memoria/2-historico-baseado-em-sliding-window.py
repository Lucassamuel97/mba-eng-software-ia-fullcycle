from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import trim_messages
from langchain_core.runnables import RunnableLambda

# Carrega variaveis de ambiente (ex.: GOOGLE_API_KEY).
load_dotenv()

# Define o prompt com historico e instrucao em pt-br.
prompt = ChatPromptTemplate.from_messages([
    ("system", "Voce e um assistente util e, quando possivel, responde com uma piada curta. Responda em pt-br."),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])

# Instancia o modelo Gemini.
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.9)

# Limita o historico a uma janela curta (sliding window).
def prepare_inputs(payload: dict) -> dict:
    raw_history = payload.get("raw_history", [])
    trimmed = trim_messages(
        raw_history,
        token_counter=len,
        max_tokens=2,
        strategy="last",
        start_on="human",
        include_system=True,
        allow_partial=False,
    )
    return {"input": payload.get("input", ""), "history": trimmed}

# Encadeia o preparo, prompt e modelo em um pipeline.
prepare = RunnableLambda(prepare_inputs)
chain = prepare | prompt | llm

# Armazena historicos por sessao.
session_store: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]


# Adiciona memoria ao pipeline.
conversational_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="raw_history"
)

# Identifica a sessao para recuperar o historico correto.
config = {"configurable": {"session_id": "demo-session"}}

resp1 = conversational_chain.invoke({"input": "Meu nome e Samucaa. Responda apenas com 'OK' e nao mencione meu nome."}, config=config)
print("Assistente:", resp1.content)

resp2 = conversational_chain.invoke({"input": "Diga uma curiosidade em uma frase. Nao mencione meu nome."}, config=config)
print("Assistente:", resp2.content)

resp3 = conversational_chain.invoke({"input": "Qual e o meu nome?"}, config=config)
print("Assistente:", resp3.content)