import os
import sys
import json
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

GOOGLE_DRIVE_FOLDER = "https://drive.google.com/drive/folders/1BG9Ud58D9-PioyqyNsYke7NwRMLiXoSi?usp=sharing"

# Modelos disponibles
MODELS = {
    "GPT-4o-mini  (Rápido · ~$0.0005/consulta)": {
        "provider": "openai",
        "model_id": "gpt-4o-mini",
    },
    "MiMo v2.6 Pro  (Razonamiento profundo · OpenRouter)": {
        "provider": "openrouter",
        "model_id": "xiaomi/mimo-v2.6-pro",
    },
}

SYSTEM_PROMPT = """Eres el Asistente de Coordinación Académica de la Universidad ESAN.
Ayudas a las coordinadoras a resolver consultas de estudiantes sobre reglamentos y normativas de pregrado, con total precisión y cero alucinaciones.

Responde SIEMPRE en formato JSON estricto con las siguientes claves (sin texto adicional fuera del JSON):
{
  "requiere_aprobacion": "string — quién debe aprobar o validar esto (ej: 'Profesor del curso', 'Secretaría Académica', 'Dirección de Pregrado', 'La coordinadora puede resolver directamente')",
  "borrador_alumno": "string — mensaje empático y claro listo para enviar al alumno. Incluye referencias numéricas entre corchetes [1], [2] para cada cita normativa usada. Usa un tono formal pero cercano. Firma con 'Coordinación Académica - Universidad ESAN'.",
  "acciones_coordinadora": ["lista de strings — acciones concretas que debe verificar o gestionar la coordinadora en el sistema, intranet, actas o expediente del alumno"],
  "razonamiento": "string — explicación técnica breve de qué reglas normativas aplicaste, de cuáles reglamentos, y por qué descartaste otras interpretaciones",
  "citas_normativas": [
    {
      "numero": 1,
      "documento": "nombre exacto del PDF",
      "seccion": "Artículo o Capítulo identificado",
      "pagina": "número de página",
      "extracto": "frase literal del reglamento que sustenta la respuesta (máx. 200 caracteres)"
    }
  ]
}

REGLAS ABSOLUTAS:
- No inventes artículos, plazos ni requisitos. Solo usa lo que está en los fragmentos provistos.
- Si la consulta no está cubierta por los documentos, coloca en 'borrador_alumno' que se consultará formalmente y en 'requiere_aprobacion' → 'Dirección de Pregrado (caso no previsto en normativa)'.
- Prioriza siempre la normativa más reciente (2025/2026) ante versiones anteriores.
- Usa referencias numéricas [1], [2] en el borrador_alumno que correspondan con los índices en citas_normativas.
"""


def _build_client(provider: str, api_key: str, openrouter_key: str = "") -> OpenAI:
    if provider == "openrouter":
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_key or api_key,
        )
    return OpenAI(api_key=api_key)


class RAGEngine:
    def __init__(self, api_key: str, openrouter_key: str = ""):
        self.api_key = api_key
        self.openrouter_key = openrouter_key
        self.embed_client = OpenAI(api_key=api_key)  # Embeddings siempre con OpenAI
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
        if not self.is_indexed():
            return []

        embed_resp = self.embed_client.embeddings.create(
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
            for doc_text, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0] if "distances" in results else [0] * top_k
            ):
                section = meta.get("section") or detect_section_or_article(doc_text)
                sources.append({
                    "text": doc_text,
                    "source": meta.get("source", "Documento ESAN"),
                    "page": meta.get("page", 1),
                    "section": section,
                    "score": round((1 - dist) * 100, 1) if dist is not None else None
                })
        return sources

    def answer_question(
        self,
        query: str,
        context_extra: str = "",
        model_label: str = "GPT-4o-mini  (Rápido · ~$0.0005/consulta)",
        chat_history: List[Dict[str, str]] = None,
        top_k: int = 6,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Devuelve (parsed_dict, sources).
        parsed_dict incluye: requiere_aprobacion, borrador_alumno,
        acciones_coordinadora, razonamiento, citas_normativas.
        """
        sources = self.search_relevant_chunks(query, top_k=top_k)

        if not sources:
            empty = {
                "requiere_aprobacion": "Sin información",
                "borrador_alumno": "No se encontraron documentos indexados. Por favor indexe la base documental primero.",
                "acciones_coordinadora": ["Verificar que la base documental esté indexada."],
                "razonamiento": "La base vectorial está vacía.",
                "citas_normativas": []
            }
            return empty, []

        context_parts = []
        for i, s in enumerate(sources, 1):
            context_parts.append(
                f"[FRAGMENTO #{i} | {s['source']} | {s['section']} | Pág. {s['page']}]:\n{s['text']}"
            )
        context_str = "\n---\n".join(context_parts)

        user_content = f"""FRAGMENTOS NORMATIVOS VERIFICADOS:
{context_str}

CONSULTA DE LA COORDINADORA:
Mensaje del alumno: {query}
{f'Contexto adicional: {context_extra}' if context_extra else ''}

Genera la respuesta estructurada en JSON estricto como se indicó en el system prompt."""

        model_cfg = MODELS.get(model_label, list(MODELS.values())[0])
        client = _build_client(
            provider=model_cfg["provider"],
            api_key=self.api_key,
            openrouter_key=self.openrouter_key
        )

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if chat_history:
            for m in chat_history[-4:]:
                messages.append({"role": m["role"], "content": m["content"]})
        messages.append({"role": "user", "content": user_content})

        response = client.chat.completions.create(
            model=model_cfg["model_id"],
            messages=messages,
            temperature=0.15,
            max_tokens=1400,
        )

        raw = response.choices[0].message.content.strip()
        # Limpiar bloque markdown si el modelo lo devuelve con ```json
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {
                "requiere_aprobacion": "Ver respuesta completa",
                "borrador_alumno": raw,
                "acciones_coordinadora": [],
                "razonamiento": "El modelo no devolvió JSON válido. Se muestra la respuesta directa.",
                "citas_normativas": []
            }

        return parsed, sources
