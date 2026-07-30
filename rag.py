"""Cadena RAG asíncrona: recupera contexto y genera una respuesta fundamentada.

get_rag_response(query):
  a) busca por similitud en ChromaDB los fragmentos más relevantes,
  b) arma un prompt que incluye SOLO ese contexto (grounding),
  c) llama al LLM de forma asíncrona,
  d) parsea la salida a un modelo Pydantic (respuesta + fuentes).

El prompt actúa como "filtro de veracidad": si la respuesta no está en el
contexto, el modelo debe decir que no tiene esa información (no alucinar).
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from ingest import get_vectorstore
from schemas import RAGResponse

TOP_K = 3  # entre 3 y 5: evita el "contexto infinito" y el Lost in the Middle.

_parser = PydanticOutputParser(pydantic_object=RAGResponse)

_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Sos un asistente técnico. Respondé la pregunta usando EXCLUSIVAMENTE el "
            "CONTEXTO proporcionado. Si la respuesta no está en el contexto, indicá que "
            "no tenés acceso a esa información, poné respuesta_encontrada en false y dejá "
            "fuentes vacío. No inventes nada fuera del contexto.\n\n{format_instructions}",
        ),
        ("human", "CONTEXTO:\n{context}\n\nPREGUNTA: {question}"),
    ]
).partial(format_instructions=_parser.get_format_instructions())


def _build_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        temperature=0,
        max_output_tokens=1024,
        google_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
    )


def _format_docs(docs: list[Document]) -> str:
    """Concatena los fragmentos recuperados, etiquetando su fuente."""
    return "\n\n".join(
        f"[Fuente: {d.metadata.get('source', '?')}]\n{d.page_content}" for d in docs
    )


async def get_rag_response(query: str, k: int = TOP_K) -> RAGResponse:
    load_dotenv()
    vectorstore = get_vectorstore()

    # a) Recuperación: búsqueda de similitud (asíncrona).
    docs = await vectorstore.asimilarity_search(query, k=k)
    context = _format_docs(docs)

    # b + c + d) Cadena LCEL: prompt (con contexto) -> LLM -> parser Pydantic.
    chain = _prompt | _build_llm() | _parser
    try:
        return await chain.ainvoke({"context": context, "question": query})
    except Exception as e:  # noqa: BLE001 - error de LLM o de parseo tras el intento
        return RAGResponse(
            respuesta=f"[error controlado] {type(e).__name__}: {e}",
            fuentes=[],
            respuesta_encontrada=False,
        )


async def main() -> None:
    pruebas = [
        (
            "PREGUNTA EN CONTEXTO",
            "¿Qué es el chunking y por qué conviene medirlo en tokens?",
        ),
        ("PREGUNTA TRAMPA", "¿Cuál es la capital de Francia?"),
    ]
    for etiqueta, pregunta in pruebas:
        print("\n" + "=" * 70)
        print(f"[{etiqueta}] {pregunta}")
        print("=" * 70)
        respuesta = await get_rag_response(pregunta)
        print(respuesta.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
