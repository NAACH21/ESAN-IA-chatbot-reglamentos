import os
import sys
from typing import List, Dict, Any, Tuple
from openai import OpenAI

# Compatibilidad SQLite para despliegues en la nube (Linux / Streamlit Cloud)
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import chromadb
from ingest import detect_section_or_article

CHROMA_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
COLLECTION_NAME = "esan_reglamentos"

SYSTEM_PROMPT = """Eres el Sistema de Soporte a la Decisión Normativa para la Dirección Ejecutiva y Coordinación Académica de Pregrado de la Universidad ESAN.
Tu propósito es brindar dictámenes ejecutivos, análisis reglamentarios precisos y sustento normativo con base EXCLUSIVA en los documentos oficiales de la institución provistos en el contexto.

DIRECTRICES PARA DIRECCIÓN EJECUTIVA Y COORDINACIÓN:
1. ESTRUCTURA EJECUTIVA: Organiza tus respuestas en tres bloques claros:
   - **Dictamen Ejecutivo:** Respuesta directa, concisa y accionable para la toma de decisiones.
   - **Sustento Normativo (Citas Múltiples):** Desglose detallado de las reglas, requisitos, plazos o sanciones aplicables.
   - **Implicancias / Procedimiento Administrativo:** Instancias responsables (Dirección de Pregrado, Secretaría Académica, Comité de Disciplina, etc.) o pasos operativos que debe validar la coordinación.
2. CITAS MÚLTIPLES OBLIGATORIAS (SECCIÓN + DOCUMENTO FUENTE): Cada punto normativo debe incluir explícitamente entre corchetes o negritas la sección/artículo referenciado, el documento fuente y la página (ejemplo: **[Reglamento de Estudios de Pregrado | Capítulo V, Art. 42° | Pág. 14]**). Si varios reglamentos abordan el tema, integra y cita todas las fuentes pertinentes.
3. RIGOR CERO ALUCINACIONES: Bajo ninguna circunstancia inventes normas, porcentajes o excepciones. Si un caso vacío legal o excepcional no está contemplado en los textos, indícalo expresamente señalando qué órgano directivo tiene la facultad de resolver casos no previstos según el reglamento.
4. PRELACIÓN Y VIGENCIA: Ante discrepancias entre documentos de diferentes años, prioriza la normativa de fecha más reciente (2025/2026) e infórmalo en el dictamen.
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

    def search_relevant_chunks(self, query: str, top_k: int = 6) -> List[Dict[str, Any]]:
        """Busca los fragmentos más pertinentes en ChromaDB con sección y documento fuente."""
        if not self.is_indexed():
            return []

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
                section = meta.get("section") or detect_section_or_article(doc_text)
                sources.append({
                    "text": doc_text,
                    "source": meta.get("source", "Documento Normativo ESAN"),
                    "page": meta.get("page", 1),
                    "section": section,
                    "score": round((1 - dist) * 100, 1) if dist is not None else None
                })
        return sources

    def answer_question(self, query: str, chat_history: List[Dict[str, str]] = None, top_k: int = 6) -> Tuple[str, List[Dict[str, Any]]]:
        """Genera un dictamen ejecutivo utilizando RAG con gpt-4o-mini y citas múltiples."""
        sources = self.search_relevant_chunks(query, top_k=top_k)

        if not sources:
            return (
                "La base documental aún no ha sido indexada o no se encontraron registros normativos asociados.",
                []
            )

        context_parts = []
        for i, s in enumerate(sources, 1):
            context_parts.append(
                f"[CITA #{i} | DOCUMENTO FUENTE: {s['source']} | SECCIÓN REFERENCIADA: {s['section']} | PÁGINA: {s['page']}]:\n{s['text']}\n"
            )
        context_str = "\n---\n".join(context_parts)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

        if chat_history:
            for msg in chat_history[-4:]:
                messages.append({"role": msg["role"], "content": msg["content"]})

        user_content = f"""REGLAMENTOS OFICIALES DE LA UNIVERSIDAD ESAN (FUENTES VERIFICADAS):
{context_str}

CONSULTA DE DIRECCIÓN EJECUTIVA / COORDINACIÓN ACADÉMICA:
{query}

Emite tu respuesta ejecutiva fundamentada estrictamente en los reglamentos anteriores, incluyendo citas múltiples que especifiquen la sección/artículo referenciado, el documento fuente y la página."""

        messages.append({"role": "user", "content": user_content})

        response = self.client_ai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.15,
            max_tokens=1100
        )

        answer = response.choices[0].message.content
        return answer, sources
