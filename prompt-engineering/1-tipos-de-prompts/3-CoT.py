from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from utils import print_llm_result

# Carrega variaveis de ambiente (ex.: GOOGLE_API_KEY)
load_dotenv()
msg1 = """
Classify the log severity.

Input: "Disk usage at 85%."
Answer only with INFO, WARNING, or ERROR.
"""

msg2 = """
Classify the log severity.

Input: "Disk usage at 85%."
Think step by step about why this is INFO, WARNING, or ERROR. 
At the end, give only the final answer after "Answer:".
"""


msg3 = """
Question: How many "r" are in the word "strawberry"?
Answer only with the number of "r".
"""

msg4 = """
Question: How many "r" are in the word "strawberry"?
Explain step by step by breaking down each letter in bullet points, pointing out the "r" before giving the final answer. 
Give the final result after "Answer:".
"""

# Usa o Gemini como modelo de chat
# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# usando modelo mais fraco 
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")


# Executa as chamadas ao modelo
response1 = llm.invoke(msg1)
response2 = llm.invoke(msg2)
response3 = llm.invoke(msg3)
response4 = llm.invoke(msg4)

# Exibe os resultados
print_llm_result(msg1, response1)
print_llm_result(msg2, response2)
print_llm_result(msg3, response3)
print_llm_result(msg4, response4)