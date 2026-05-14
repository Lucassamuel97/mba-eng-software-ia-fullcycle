
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from utils import print_llm_result
from dotenv import load_dotenv

# Carrega variaveis de ambiente do arquivo .env (ex: GOOGLE_API_KEY).
load_dotenv()

# Persona 1: tom formal e tecnico.
system = ("system",
"""You are a university professor of computer science who is very technical and explain 
concepts with formal definitions and pseudocode.""")

# Persona 2: explicacao simples e acessivel para iniciantes.
system2 = ("system", """You are a high school student that is starting learning coding. 
You are not very technical and you prefer to explain concepts with simple words and examples.""")

# Entrada do usuario em comum para os dois cenarios.
user = ("user", "Explain recursion in 50 words.")

chat_prompt = ChatPromptTemplate([system, user])
chat_prompt2 = ChatPromptTemplate([system2, user])
messages = chat_prompt.format_messages()

# Modelo do Gemini definido explicitamente conforme solicitado.
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
result = model.invoke(messages)
print_llm_result(str(system), result)

result2 = model.invoke(chat_prompt2.format_messages())
print_llm_result(str(system2), result2)