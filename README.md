# Sistema de Recuperación Semántica Local (RAG)

Pre-entrega 3 — Módulo 3, AI Engineering.

Sistema **RAG (Retrieval-Augmented Generation)** local end-to-end: ingesta un
conjunto de documentos en una base vectorial (ChromaDB) y responde preguntas
**fundamentadas exclusivamente en esos documentos**, citando las fuentes y
negándose a alucinar cuando la respuesta no está en el contexto.

## 🧩 Arquitectura

```
data/*.md ──► [ingest.py] chunking (500 tok / 50 overlap) ──► embeddings (Gemini) ──► ChromaDB (./vectorstore)

pregunta ──► [rag.py] búsqueda por similitud (top-3) ──► prompt con contexto ──► LLM ──► RAGResponse (Pydantic)
```

| Archivo | Rol |
|---|---|
| `data/` | Dataset de ejemplo (apuntes sobre embeddings, chunking y vector DBs). |
| `schemas.py` | Modelo Pydantic `RAGResponse` (respuesta + fuentes + flag). |
| `ingest.py` | Lee `data/`, fragmenta en tokens, persiste en ChromaDB. |
| `rag.py` | `get_rag_response()` async: retriever + cadena LCEL + parser Pydantic. |

## ✨ Puntos clave

- **Mismo modelo de embeddings** para indexar y consultar (`gemini-embedding-001`) — evita el error #1 de RAG.
- **Chunking en tokens** (no caracteres) con `tiktoken`: 500 tokens, 50 de overlap.
- **Grounding**: el prompt obliga a responder solo con el contexto y a decir que no sabe si la info no está.
- **top-k = 3**: evita el "contexto infinito" y el efecto *Lost in the Middle*.
- **Salida validada con Pydantic** (`PydanticOutputParser`), con trazabilidad de fuentes.
- **Persistencia**: si la base ya existe, no se reindexa.

## 🚀 Cómo ejecutarlo

### 1. Instalar dependencias

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
```

O con pip:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate  |  Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configurar la API key

Copiar `.env.example` a `.env` y completar la `GEMINI_API_KEY`
(gratis en [aistudio.google.com/apikey](https://aistudio.google.com/apikey)):

```bash
cp .env.example .env     # Windows: copy .env.example .env
```

### 3. Ingestar los documentos (una vez)

```bash
python ingest.py
```

Lee los archivos de `data/`, los fragmenta y crea la base en `./vectorstore`.

### 4. Ejecutar el RAG

```bash
python rag.py
```

Corre dos pruebas: una pregunta **en contexto** y una **pregunta trampa** (sin
respuesta en los documentos), verificando que el sistema no alucine.

## 📤 Ejemplo de salida

Pregunta en contexto:

```json
{
  "respuesta": "El chunking es el proceso de dividir un documento largo en fragmentos más pequeños...",
  "fuentes": ["chunking.md"],
  "respuesta_encontrada": true
}
```

Pregunta trampa (no está en los documentos):

```json
{
  "respuesta": "No tengo acceso a esa información en el contexto proporcionado.",
  "fuentes": [],
  "respuesta_encontrada": false
}
```

## 🔑 Variables de entorno

| Variable | Descripción | Requerida |
|---|---|---|
| `GEMINI_API_KEY` | API key de Google AI Studio (embeddings + generación). | Sí |
| `GEMINI_MODEL` | Modelo de generación (default `gemini-2.5-flash`). | No |

