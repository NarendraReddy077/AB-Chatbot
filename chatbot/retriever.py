import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import settings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

CHROMA_PATH = "local_chroma_db"

def get_vectorstore(collection_name: str):
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
    )

def get_retriever(collection_name: str):
    vectorstore = get_vectorstore(collection_name)
    return vectorstore.as_retriever(search_kwargs={"k": 3})
