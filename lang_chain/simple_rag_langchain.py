from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from settings import get_project_root


# pip install "unstructured[md]"
loader = DirectoryLoader(get_project_root() / "docs/", glob="*.pdf")
docs = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)



vectorstore = Chroma.from_documents(documents=splits, embedding=FastEmbedEmbeddings())
retriever = vectorstore.as_retriever()



# pip install langchainhub
prompt = hub.pull("rlm/rag-prompt")


##### run LLM and do RAG
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

llm = ChatOpenAI(
    openai_api_base="http://localhost:8000/v1",
    request_timeout=600,  # secs, I guess.
    openai_api_key="loremIpsum",
    max_tokens=32768,
)


rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm  # This should be the LLM initialized in the task above.
)
print(rag_chain.invoke("What do I like to drink?"))