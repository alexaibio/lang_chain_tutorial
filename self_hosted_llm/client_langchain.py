from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


llm = ChatOpenAI(
    openai_api_base="http://localhost:8000/v1",
    request_timeout=600,  # secs, I guess.
    openai_api_key="loremIpsum",
    max_tokens=32768,
)
chat_history = [
    SystemMessage(content="You are a bartender."),
    HumanMessage(content="What do I enjoy drinking?"),
]
print(llm(chat_history))