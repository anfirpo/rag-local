# Embeddings

Un embedding es una representación numérica densa de un texto: una lista de
números (un vector) que captura su significado. Los modelos modernos generan
vectores de entre 1536 y 3072 dimensiones.

## Similitud

Para medir qué tan parecidos son dos textos se usa la similitud coseno, que
mide el ángulo entre sus vectores. Su valor va de 0 a 1: cuanto más cercano a 1,
más parecidos son en significado. La similitud coseno es la métrica estándar en
sistemas RAG porque ignora la longitud del texto y se enfoca en el tema.

## Limitación con la negación

Los embeddings capturan el tema, no la lógica. Por eso pueden confundir la
negación: "hay stock" y "no hay stock" quedan cerca en el espacio vectorial
aunque signifiquen lo contrario. La negación la resuelve el modelo de lenguaje
que lee los documentos recuperados, no el embedding.
