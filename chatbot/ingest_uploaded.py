import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chatbot.retriever import get_vectorstore

UPLOADED_COLLECTION = "uploaded_docs"

def init_uploaded_collection():
    vectorstore = get_vectorstore(UPLOADED_COLLECTION)
    try:
        vectorstore.client.get_collection(UPLOADED_COLLECTION)
    except:
        from qdrant_client.http.models import Distance, VectorParams
        vectorstore.client.recreate_collection(
            collection_name=UPLOADED_COLLECTION,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

def ingest_uploaded_doc(filepath: str):
    init_uploaded_collection()
    loader = PyPDFLoader(filepath)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    
    vectorstore = get_vectorstore(UPLOADED_COLLECTION)
    vectorstore.add_documents(chunks)
