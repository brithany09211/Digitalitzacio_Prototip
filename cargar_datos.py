"""
Carga los documentos de la knowledge base,
los divide en fragmentos y crea la base vectorial ChromaDB.
Solo hace falta ejecutarlo cuando se añaden nuevos documentos.
"""
import shutil
import os
import glob
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    CSVLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# CONFIGURACIÓN
EMBED_MODEL = "nomic-embed-text"
DB_NAME = "hotelmar_db"
KNOWLEDGE_PATH = "knowledge-base"

# EMBEDDINGS
embeddings = OllamaEmbeddings(model=EMBED_MODEL)

# FUNCIÓN PARA CARGAR DOCUMENTOS
def load_hotel_context():

    text_loader_kwargs = {'encoding': 'utf-8'}
    documents = []

    # TXT Y MD
    for ext in ["**/*.txt", "**/*.md"]:

        loader = DirectoryLoader(
            KNOWLEDGE_PATH,
            glob=ext,
            loader_cls=TextLoader,
            loader_kwargs=text_loader_kwargs
        )

        docs = loader.load()

        for d in docs:
            d.metadata["category"] = "institucional"

        documents.extend(docs)

    # CSV
    csv_files = glob.glob(
        os.path.join(KNOWLEDGE_PATH, "**/*.csv"),
        recursive=True
    )

    if csv_files:

        loader = DirectoryLoader(
            KNOWLEDGE_PATH,
            glob="**/*.csv",
            loader_cls=CSVLoader
        )

        csv_docs = loader.load()

        for d in csv_docs:
            d.metadata["category"] = "clients"

        documents.extend(csv_docs)

    print(f"Documentos cargados: {len(documents)}")

    return documents


# CARGAR DOCUMENTOS
all_documents = load_hotel_context()

# CHUNKING
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(all_documents)

print(f"Chunks generados: {len(chunks)}")

# ELIMINAR BASE ANTERIOR
if os.path.exists(DB_NAME):

    print("Base vectorial existente detectada.")
    print("Eliminando y reconstruyendo...")

    shutil.rmtree(DB_NAME)

# CREAR BASE VECTORIAL
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=DB_NAME,
    collection_name="hotel_knowledge"
)

print("Base vectorial creada correctamente.")