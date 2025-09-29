from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def load_and_split(file_path: str, metadata: Optional[dict] = None) -> List[Document]:
    suffix = file_path.lower().split(".")[-1]
    if suffix == "pdf":
        loader = PyPDFLoader(file_path)
    elif suffix == "docx":
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("Only PDF/DOCX supported")

    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
    docs = loader.load_and_split(splitter)

    if metadata:
        for d in docs:
            d.metadata.update(metadata)
    return docs