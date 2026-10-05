import os
from typing import List, Dict, Any, Generator, Tuple
from openai import OpenAI
import chromadb

CHROMA_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
COLLECTION_NAME = "esan_reglamentos"

SYSTEM_PROMPT = """Eres el Asistente Virtual Oficial de Normativa y Reglamentos de Pregrado de la Universidad ESAN.
Tu misión es orientar con máxima precisión a estudiantes, docentes y postulantes resolviendo sus consultas con base EXCLUSIVA en los fragmentos de reglamentos oficiales provistos a continuación.

REGLAS FUNDAMENTALES:
1. CITAS OBLIGATORIAS: Cada vez que des información sobre un requisito, trámite, porcentaje o sanción, debes citar explícitamente el documento oficial de referencia y, cuando sea identificable en el texto, el número de Artículo o Capítulo (ejemplo: *"Según el Reglamento de Estudios de Pregrado (Art. 45)..."*).
2. RIGOR Y FIDELIDAD: Responde únicamente con lo que esté respaldado por los documentos. No inventes procedimientos ni especules sobre excepciones.
3. CASOS NO CONTEMPLADOS: Si la consulta del usuario no se encuentra detallada en los documentos provistos, dilo amablemente sin rodeos y recomienda consultar formalmente con la Secretaría Académica de la Universidad ESAN o la Dirección de Pregrado.
4. FORMATO: Emplea formato Markdown elegante: listas numeradas para pasos consecutivos, viñetas para requisitos y negritas en los conceptos clave para facilitar la lectura rápida del estudiante.
5. VIGENCIA: Si encuentras información en normas de distintas fechas, prioriza siempre el documento con la fecha más reciente (ej. normativas de 2025/2026 frente a versiones anteriores).
"""

class RAGEngine:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client_ai = OpenAI(api_key=api_key)
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
        try:
            self.collection = self.chroma_client.get_collection(name=COLLECTION_NAME)
        except Exception:
            self.collection = None

    def is_indexed(self) -> bool:
        if self.collection is None:
            try:
                self.collection = self.chroma_client.get_collection(name=COLLECTION_NAME)
            except Exception:
                return False
        return self.collection.count() > 0

    def get_document_count(self) -> int:
        if not self.is_indexed():
            return 0
        return self.collection.count()

    def search_relevant_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Busca los fragmentos más pertinentes en ChromaDB."""
        if not self.is_indexed():
            return []

        # Generar embedding de la consulta
        embed_resp = self.client_ai.embeddings.create(
            model="text-embedding-3-small",
            input=[query]
        )
        query_embedding = embed_resp.data[0].embedding

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        sources = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0] if "distances" in results else [0]*len(docs)

            for doc_text, meta, dist in zip(docs, metas, distances):
                sources.append({
                    "text": doc_text,
                    "source": meta.get("source", "Documento ESAN"),
                    "page": meta.get("page", 1),
                    "score": round(1 - dist, 3) if dist is not None else None
                })
        return sources

    def answer_question(self, query: str, chat_history: List[Dict[str, str]] = None, top_k: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
        """Genera una respuesta utilizando RAG con gpt-4o-mini."""
        sources = self.search_relevant_chunks(query, top_k=top_k)

        if not sources:
            return (
                "Aún no se han indexado los reglamentos o no se encontró información relevante en la base de datos.",
                []
            )

        # Construir contexto
        context_parts = []
        for i, s in enumerate(sources, 1):
            context_parts.append(
                f"[Fragmento #{i} | Fuente: {s['source']} | Página {s['page']}]:\n{s['text']}\n"
            )
        context_str = "\n---\n".join(context_parts)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

        # Agregar historial previo si existe (hasta 4 mensajes para optimizar tokens)
        if chat_history:
            for msg in chat_history[-4:]:
                messages.append({"role": msg["role"], "content": msg["content"]})

        user_content = f"""DOCUMENTOS DE REFERENCIA DE LA UNIVERSIDAD ESAN:
{context_str}

PREGUNTA DEL ESTUDIANTE:
{query}

Responde a la pregunta basándote en los documentos de referencia anteriores, indicando de forma visible el documento y artículo o página de donde proviene la respuesta."""

        messages.append({"role": "user", "content": user_content})

        response = self.client_ai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.2,
            max_tokens=1000
        )

        answer = response.choices[0].message.content
        return answer, sources
