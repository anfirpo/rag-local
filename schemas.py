"""Contrato de datos de la respuesta del sistema RAG.

La salida del pipeline no es texto libre: es un objeto validado con la respuesta
y las fuentes usadas, para garantizar la trazabilidad (de dónde salió la info).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RAGResponse(BaseModel):
    """Respuesta fundamentada (grounded) del sistema RAG."""

    respuesta: str = Field(
        description="Respuesta a la pregunta basada EXCLUSIVAMENTE en el contexto. "
        "Si la respuesta no está en el contexto, indicar que no se tiene esa información.",
    )
    fuentes: list[str] = Field(
        default_factory=list,
        description="Nombres de los archivos fuente citados. Vacío si la respuesta "
        "no estaba en el contexto.",
    )
    respuesta_encontrada: bool = Field(
        description="True si la respuesta estaba en el contexto; False si no se encontró.",
    )
