import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chatbot.retriever import get_vectorstore
from config import settings

DEFAULT_PDF_DIR = "data/default_pdfs"
DEFAULT_COLLECTION = "default_docs"

def ingest_default_docs():
    vectorstore = get_vectorstore(DEFAULT_COLLECTION)
    
    try:
        count = vectorstore.client.count(collection_name=DEFAULT_COLLECTION).count
        if count > 0:
            print("Default documents already ingested.")
            return
    except Exception as e:
        print(f"Collection {DEFAULT_COLLECTION} doesn't exist or error checking count: {e}. Will create/ingest.")
        from qdrant_client.http.models import Distance, VectorParams
        vectorstore.client.recreate_collection(
            collection_name=DEFAULT_COLLECTION,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        
    all_docs = []
    if not os.path.exists(DEFAULT_PDF_DIR):
        os.makedirs(DEFAULT_PDF_DIR)
        
    for filename in os.listdir(DEFAULT_PDF_DIR):
        if filename.endswith(".pdf"):
            filepath = os.path.join(DEFAULT_PDF_DIR, filename)
            loader = PyPDFLoader(filepath)
            docs = loader.load()
            all_docs.extend(docs)
            
    if all_docs:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(all_docs)
        vectorstore.add_documents(chunks)
        print("Successfully ingested default documents.")
    else:
        print("No default documents found to ingest.")
