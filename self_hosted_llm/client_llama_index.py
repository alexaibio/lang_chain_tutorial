"""
Test LlamaIndex client
"""

from llama_index.llms.openai_like import OpenAILike
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate


llm = OpenAILike(
    api_base="http://localhost:8000/v1",
    timeout=600,  # secs
    api_key="loremIpsum",
    is_chat_model=True,
    context_window=32768,
)


chat_history = [
    ChatMessage(role="system", content="You are a marketing expert for a small and medium businesses."),
    ChatMessage(role="user", content="What marketing activities you may recommend to start a hardware STM32 company?"),
]
output = llm.chat(chat_history)
print(output)

