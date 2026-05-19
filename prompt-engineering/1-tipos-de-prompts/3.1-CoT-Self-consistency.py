from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from utils import print_llm_result

# Carrega variaveis de ambiente (ex.: GOOGLE_API_KEY)
load_dotenv()
msg1 = """
Question: In an API endpoint that returns a list of users and their posts, the developer wrote:

users := db.FindAllUsers()
for _, u := range users {
    u.Posts = db.FindPostsByUserID(u.ID)
}

How many database queries will this code execute if there are N users?

Generate 3 different reasoning paths step by step.
At the end, summarize the answers and choose the most consistent one, ignoring outliers.
If there are 3 different answers, ONLY reply: "I can't find a consistent answer".
"""


# Usa o Gemini como modelo de chat
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


# Executa a chamada ao modelo
response1 = llm.invoke(msg1)
print_llm_result(msg1, response1)