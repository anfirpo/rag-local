# Bases de datos vectoriales

Una base de datos vectorial almacena embeddings y permite buscarlos por
similitud. A diferencia de una base SQL, la búsqueda principal no es por
coincidencia exacta sino por cercanía semántica en el espacio vectorial.

## ChromaDB

ChromaDB es una base de datos vectorial de código abierto y simple. En su modo
local persistente guarda los datos en archivos en una carpeta del disco, sin
necesidad de Docker, y permite operaciones CRUD sobre el conocimiento del agente.

## Operaciones principales

El método upsert inserta o actualiza documentos: si el ID ya existe, lo
actualiza; si no, lo crea. Es más robusto que add, que falla ante IDs duplicados.
El método query busca por similitud semántica y devuelve los documentos más
cercanos a una consulta; es el corazón de un sistema RAG.

## Metadatos

Los metadatos son etiquetas que se adjuntan a cada vector, como el autor, la
fecha o la fuente. Sirven para filtrar resultados antes de la búsqueda y para
rastrear el origen de la información que usa el modelo de lenguaje.
