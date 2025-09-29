from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    collection_name="legal_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_data"   # local folder
)