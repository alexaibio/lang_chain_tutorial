"""
 https://docs.llamaindex.ai/en/stable/examples/customization/prompts/chat_prompts/
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

qa_prompt_str = (
    "Context information is below.\n"
    "---------------------\n"
    "I already own a small software service company, the economic is bad\n"
    "---------------------\n"
    "Given the context information and no prior knowledge, "
    "answer the question: {query_str}\n"
)
query_str = 'What marketing activities you may recomment to start a hardware STM32 company?'

# Text QA Prompt
text_qa_template = ChatPromptTemplate(
    [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=("You are a marketing expert for small and medium businesses."),
        ),
        ChatMessage(role=MessageRole.USER, content=qa_prompt_str.format(query_str=query_str)),
    ]
)

# TODO: not working!
#output = llm.completion_to_prompt(text_qa_template)
#print(output)