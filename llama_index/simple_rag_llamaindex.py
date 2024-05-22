from llama_index.llms.openai_like import OpenAILike
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate, ServiceContext, SimpleDirectoryReader, VectorStoreIndex


llm = OpenAILike(
    api_base="http://localhost:8000/v1",
    timeout=600,  # secs
    api_key="loremIpsum",
    is_chat_model=True,
    context_window=32768,
)


service_context = ServiceContext.from_defaults(
    embed_model="local",
    llm=llm, # This should be the LLM initialized in the task above.
)
documents = SimpleDirectoryReader(
    input_dir="mock_notebook/",
).load_data()
index = VectorStoreIndex.from_documents(
    documents=documents,
    service_context=service_context,
)
engine = index.as_query_engine(
    service_context=service_context,
)
output = engine.query("What do I like to drink?")
print(output)