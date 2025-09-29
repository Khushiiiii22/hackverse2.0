# chains.py
import os
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_google_genai import ChatGoogleGenerativeAI  # ✅ Gemini LLM
from .prompts import template
from .vectorstore import vectorstore

# Load API key
load_dotenv()

# Initialize Gemini (Chat model, not LLM — more accurate for RAG)
try:
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        temperature=0.1,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
except Exception as e:
    # Avoid crashing at import time if ADC / credentials are not available.
    # Users will see a clear error when they actually try to build/use the chain.
    print("Warning: could not initialize Gemini LLM (ChatGoogleGenerativeAI):", str(e))
    llm = None

def build_chain(doc_id: str):
    if llm is None:
        raise RuntimeError(
            "LLM is not configured. Set GOOGLE_API_KEY / Application Default Credentials before calling build_chain."
        )
    retriever = vectorstore.as_retriever(
        search_kwargs={"filter": {"doc_id": doc_id}, "k": 10}
    )
    combine_docs_chain = create_stuff_documents_chain(llm, template)
    qa_chain = create_retrieval_chain(retriever, combine_docs_chain)
    return qa_chain