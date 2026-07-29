"""Módulo de ingesta: lee documentos, los fragmenta y los persiste en ChromaDB.

Lee los archivos .md/.txt de ./data, aplica un chunking recursivo medido en
TOKENS (500 con 50 de overlap) y guarda los fragmentos —con su fuente como
metadato— en una base ChromaDB local persistente (./vectorstore).

Regla clave: se usa el MISMO modelo de embeddings para indexar y para consultar
(ver rag.py). Si ya existe la base, no se reindexa (ahorra tiempo y costo).
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import tiktoken
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = "./data"
PERSIST_DIR = "./vectorstore"
COLLECTION = "apuntes_rag"
EMBEDDING_MODEL = "models/gemini-embedding-001"

_ENCODER = tiktoken.get_encoding("cl100k_base")


def _token_len(text: str) -> int:
    """Longitud en tokens reales (no caracteres), con tiktoken."""
    return len(_ENCODER.encode(text))


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Modelo de embeddings. El MISMO se usa para indexar y para consultar."""
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
    )


def _load_and_split() -> list[Document]:
    """Lee los archivos de ./data y los parte en chunks con su fuente."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,          # tokens (por length_function)
        chunk_overlap=50,        # tokens de solapamiento
        length_function=_token_len,
        separators=["\n\n", "\n", " ", ""],
    )
    paths = sorted(glob.glob(f"{DATA_DIR}/*.md")) + sorted(glob.glob(f"{DATA_DIR}/*.txt"))
    docs: list[Document] = []
    for path in paths:
        texto = Path(path).read_text(encoding="utf-8")
        for chunk in splitter.split_text(texto):
            docs.append(Document(page_content=chunk, metadata={"source": os.path.basename(path)}))
    return docs


def _db_exists() -> bool:
    p = Path(PERSIST_DIR)
    return p.exists() and any(p.iterdir())


def get_vectorstore(force_reindex: bool = False) -> Chroma:
    """Devuelve la vector DB: la carga si ya existe, o la construye desde ./data."""
    load_dotenv()
    embeddings = get_embeddings()

    if _db_exists() and not force_reindex:
        print(f"[ingest] Base existente encontrada en {PERSIST_DIR}, se reutiliza.")
        return Chroma(
            collection_name=COLLECTION,
            embedding_function=embeddings,
            persist_directory=PERSIST_DIR,
        )

    print("[ingest] Indexando documentos desde ./data ...")
    docs = _load_and_split()
    print(f"[ingest] {len(docs)} chunk(s) generados. Calculando embeddings y persistiendo...")
    vs = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION,
        persist_directory=PERSIST_DIR,
    )
    print(f"[ingest] Listo: base persistida en {PERSIST_DIR}.")
    return vs


if __name__ == "__main__":
    # Ejecutar directamente reindexar la base desde cero.
    vs = get_vectorstore(force_reindex=True)
    print(f"Documentos en la colección: {vs._collection.count()}")
