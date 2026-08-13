from langchain.openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
model=ChatOpenAI(model='Model name')
result=model.invoke('WHAT IS THE CAPITAL OF INDIA')
print(result)
print(result.content)



#Bro 👨‍🏫 Temperature in an LLM controls how random/creative the model's answer is.

#Low temperature (e.g. 0–0.3) → more focused, predictable, consistent answers.
#Medium (0.5–0.7) → balanced.
#High (0.8–1.0+) → more varied/creative answers.

llm = OpenAI(
    model="...",
    temperature=0.2
)

#max_completion_tokens:Model maximum kitne tokens ka answer generate kar sakta hai, uski limit.
llm = OpenAI(
    model="...",
    max_completion_tokens=100
)