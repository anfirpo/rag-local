# Chunking

El chunking es el proceso de dividir un documento largo en fragmentos más
pequeños antes de convertirlos en embeddings. Es necesario porque los modelos de
embedding tienen un límite de ventana de contexto: convertir un documento entero
en un solo vector diluye su significado y lo vuelve inútil para la búsqueda.

## Estrategia recursiva

La estrategia recomendada es el Recursive Character Splitting. Divide el texto
usando una jerarquía de separadores: primero por párrafos, luego por saltos de
línea, luego por espacios y, como último recurso, por caracteres. Así mantiene
juntas las oraciones y los párrafos en la medida de lo posible.

## Overlap

El overlap (solapamiento) hace que el final de un fragmento se repita al inicio
del siguiente. Esto preserva el contexto cuando un concepto clave cae justo en
el límite entre dos fragmentos. Un valor típico es de 50 tokens de overlap.

## Medir en tokens

Es un error medir el tamaño de los chunks en caracteres. Los modelos limitan y
cobran por tokens, así que conviene medir con un tokenizador real como tiktoken.
