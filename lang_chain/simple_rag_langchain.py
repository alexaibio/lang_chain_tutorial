from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from settings import get_project_root


# pip install "unstructured[md]"
# follow this for pillow installation: https://pillow.readthedocs.io/en/latest/installation/building-from-source.html
loader = DirectoryLoader(get_project_root() / "docs/", glob="*.md")
docs = loader.load()

# split all documents in a folder
splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)

# place sptils into vector chroma db (in memory
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