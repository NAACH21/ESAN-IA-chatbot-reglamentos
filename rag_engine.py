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

GOOGLE_DRIVE_FOLDER = "https://drive.google.com/drive/folders/1lVWWDXml31HMsoGxF4OTZMbJE2KLBJ0m"

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

SYSTEM_PROMPT = """Eres el Asistente Institucional de Coordinación Académica de la Universidad ESAN y ESAN Graduate School of Business.
Ayudas a las coordinadoras académicas a resolver consultas y trámites de estudiantes sobre reglamentos, directivas y normativas institucionales (Posgrado, Educación Ejecutiva, Maestrías y programas académicos), con rigurosa precisión y cero alucinaciones.

Responde SIEMPRE en formato JSON estricto con las siguientes claves (sin texto adicional fuera del JSON):
{
  "requiere_aprobacion": "string — instancia competente que debe resolver o validar (ej: 'Dirección del Programa', 'Secretaría Académica', 'Comité Académico', 'Dirección de Posgrado / Educación Ejecutiva', 'La coordinadora puede resolver directamente')",
  "borrador_alumno": "string — mensaje empático, formal y claro listo para enviar al estudiante. Incluye referencias numéricas entre corchetes [1], [2] para cada cita normativa usada. Firma con 'Coordinación Académica - Universidad ESAN'.",
  "acciones_coordinadora": ["lista de strings — acciones concretas de validación, registro en intranet, verificación de actas o gestión del expediente del alumno"],
  "razonamiento": "string — fundamentación técnica de las reglas normativas aplicadas y reglamentos considerados",
  "citas_normativas": [
    {
      "numero": 1,
      "documento": "nombre exacto del archivo PDF o documento",
      "seccion": "Artículo, Capítulo o Directiva identificada",
      "pagina": "número de página o sección",
      "extracto": "frase textual literal del documento que fundamenta la decisión (máx. 200 caracteres)"
    }
  ]
}

REGLAS ABSOLUTAS:
- No inventes artículos, plazos ni requisitos. Solo usa lo que esté en los fragmentos normativos provistos.
- Identifica correctamente el tipo de programa o instancia institucional (Posgrado, Educación Ejecutiva, PEE, Maestrías o según corresponda al reglamento citado). Nunca asumas 'Pregrado' a menos que el documento o la consulta lo especifiquen expresamente.
- Si la consulta requiere una validación excepcional no prevista en la normativa, indica en 'borrador_alumno' que el caso será elevado en consulta formal y en 'requiere_aprobacion' asigna 'Dirección Académica / Dirección del Programa (caso no previsto en normativa)'.
- Prioriza siempre la normativa y tarifarios más recientes ante versiones anteriores.
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
                if isinstance(m, dict):
                    role = m.get("role", "user")
                    content = m.get("content", "")
                else:
                    role = "user"
                    content = str(m)
                if content:
                    messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_content})

        raw = ""
        try:
            kwargs = dict(
                model=model_cfg["model_id"],
                messages=messages,
                temperature=0.15,
                max_tokens=1400,
                response_format={"type": "json_object"},
            )
            try:
                response = client.chat.completions.create(**kwargs)
            except Exception:
                # Si algún modelo no soporta response_format, intentar sin él
                kwargs.pop("response_format", None)
                response = client.chat.completions.create(**kwargs)

            raw = response.choices[0].message.content.strip()
        except Exception as api_err:
            return {
                "requiere_aprobacion": "Error de comunicación con la API",
                "borrador_alumno": f"No se pudo conectar con el modelo: {api_err}",
                "acciones_coordinadora": ["Verificar que la API Key sea válida y tenga saldo disponible."],
                "razonamiento": str(api_err),
                "citas_normativas": []
            }, sources

        # Parseo multi-capa ultra robusto del JSON
        import re as _re
        text = raw.strip()
        # 1. Quitar etiquetas de razonamiento <think>...</think>
        text = _re.sub(r'<think>[\s\S]*?</think>', '', text).strip()

        # 2. Extraer el contenido del bloque ```json ... ``` si existe
        codeblock = _re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', text)
        if codeblock:
            candidate = codeblock.group(1).strip()
        else:
            start_b = text.find('{')
            end_b = text.rfind('}')
            if start_b != -1 and end_b != -1 and end_b > start_b:
                candidate = text[start_b : end_b + 1].strip()
            else:
                candidate = text

        parsed = None
        # 3. Intento 1: json.loads con strict=False (permite saltos de línea literales)
        try:
            p = json.loads(candidate, strict=False)
            if isinstance(p, dict):
                parsed = p
        except Exception:
            pass

        # 4. Intento 2: Limpieza de comas finales (trailing commas)
        if not parsed:
            try:
                candidate_no_commas = _re.sub(r',\s*([\}\]])', r'\1', candidate)
                p = json.loads(candidate_no_commas, strict=False)
                if isinstance(p, dict):
                    parsed = p
            except Exception:
                pass

        def _clean_str(s: str) -> str:
            """Decodifica escapes sin alterar caracteres UTF-8 ni tildes."""
            return s.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"').replace('\\\\', '\\')

        # 5. Intento 3: Extracción regex de campos clave conservando tildes y caracteres UTF-8
        if not parsed:
            parsed = {}
            m_draft = _re.search(r'"borrador_alumno"\s*:\s*"((?:\\.|[^"\\])*)"', candidate, _re.DOTALL)
            if m_draft:
                parsed["borrador_alumno"] = _clean_str(m_draft.group(1))
            else:
                clean_raw = _re.sub(r'^```(?:json)?\s*', '', text)
                clean_raw = _re.sub(r'\s*```$', '', clean_raw).strip()
                parsed["borrador_alumno"] = clean_raw

            m_appr = _re.search(r'"requiere_aprobacion"\s*:\s*"((?:\\.|[^"\\])*)"', candidate)
            parsed["requiere_aprobacion"] = _clean_str(m_appr.group(1)) if m_appr else "La coordinadora puede resolver directamente"

            m_reas = _re.search(r'"razonamiento"\s*:\s*"((?:\\.|[^"\\])*)"', candidate)
            parsed["razonamiento"] = _clean_str(m_reas.group(1)) if m_reas else ""

            parsed["acciones_coordinadora"] = []
            parsed["citas_normativas"] = []

        # Normalizar citas_normativas: garantizar que cada elemento sea un dict
        raw_citas = parsed.get("citas_normativas", [])
        clean_citas = []
        for idx_c, c in enumerate(raw_citas if isinstance(raw_citas, list) else [], 1):
            if isinstance(c, dict):
                clean_citas.append(c)
            else:
                # Si por algún error el modelo devolvió una cadena, convertirla
                clean_citas.append({
                    "numero": idx_c,
                    "documento": str(c),
                    "seccion": "",
                    "pagina": "",
                    "extracto": ""
                })
        parsed["citas_normativas"] = clean_citas

        # Normalizar acciones_coordinadora como lista de strings
        raw_acc = parsed.get("acciones_coordinadora", [])
        if isinstance(raw_acc, str):
            parsed["acciones_coordinadora"] = [raw_acc]
        elif not isinstance(raw_acc, list):
            parsed["acciones_coordinadora"] = []

        return parsed, sources

