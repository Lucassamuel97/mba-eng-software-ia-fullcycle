from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from utils import print_llm_result

# Carrega variaveis de ambiente (ex.: GOOGLE_API_KEY)
load_dotenv()

# Mensagens de exemplo para zero-shot
msg1 = "What's Brazil's capital?"

msg2 = """
Find the user intent in the following text:
I'm looking for a restaurant around São Paulo who has a good rating for Japanese food.
"""

msg3 = "What's Brazil's capital? Respond only with the city name."

# Usa o Gemini como modelo de chat
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Executa as chamadas ao modelo
response1 = llm.invoke(msg1)
response2 = llm.invoke(msg2)
response3 = llm.invoke(msg3)

# Exibe os resultados
print_llm_result(msg1, response1)
print_llm_result(msg2, response2)
print_llm_result(msg3, response3)