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
from citation_utils import normalize, validate_citations, infer_program_scope, source_applies_to_scope, link_claims_to_citations

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
Tu usuaria principal es la COORDINADORA ACADÉMICA. Tu objetivo primordial es proporcionarle un análisis normativo seguro y un dictamen técnico claro para que ella valide la situación del estudiante, y adicionalmente un borrador de correo formal, humano y empático para el alumno.

Responde SIEMPRE en formato JSON estricto con las siguientes claves:
{
  "dictamen_coordinacion": {
    "procedencia": "string — 'Procede conforme a reglamento' | 'No procede conforme a reglamento' | 'Caso no contemplado expresamente — Requiere elevación'",
    "resumen_ejecutivo": "string — síntesis ejecutiva breve de la decisión; no repitas las afirmaciones detalladas cuando ya estén en afirmaciones_citadas."
  },
  "requiere_aprobacion": "string — instancia competente que debe resolver o validar formalmente (ej: 'Profesor del curso', 'Dirección del Programa', 'Secretaría Académica', 'Comité Académico', 'La coordinadora puede resolver directamente')",
  "acciones_coordinadora": [
    "lista de strings — protocolo de verificación para la coordinadora: acciones concretas a verificar en sistema, intranet, expediente del alumno o actas"
  ],
  "razonamiento": "string — fundamentación técnica detallada de las reglas normativas aplicadas y reglamentos considerados",
  "borrador_alumno": "string — mensaje institucional empático, claro y profesional listo para enviar al estudiante como correo formal. Debe seguir estas reglas:
     - Saludo institucional formal.
     - Explicación fluida y respetuosa de la respuesta en párrafos naturales.
     - Menciona los reglamentos por su nombre de manera redactada (ej: 'conforme a lo dispuesto en el Reglamento General de Maestrías...'), SIN incluir corchetes numéricos como [1] o [2].
     - Si hay requisitos o pasos específicos para el estudiante, preséntalos con viñetas limpias.
     - Cierre formal firmado por 'Coordinación Académica - Universidad ESAN'.
     - REGLA CRÍTICA DE ESTILO: NO uses encabezados robóticos numerados (NO escribas '1. Diagnóstico de su consulta', '2. Fundamentación normativa', '3. Procedimiento', '2.1, 2.2', etc.). Redacta un correo institucional auténtico.",
  "afirmaciones_citadas": [
    {"texto": "afirmación normativa principal, breve, autosuficiente y directamente relevante para responder la consulta", "cita_indices": [1]}
  ],
  "citas_normativas": [
    {
      "fragmento_id": "F1",
      "extracto": "copia LITERAL de la disposición relevante del fragmento indicado (máx. 200 caracteres)"
    }
  ]
}

REGLAS ABSOLUTAS:
1. CERO INVENCIÓN DE REQUISITOS EN CASOS NO REGULADOS:
   Si la consulta plantea una situación que NO está expresamente prevista ni regulada en los reglamentos provistos (por ejemplo: justificación de inasistencias por descanso médico en programas donde la norma no lo contempla):
   - Indica con total transparencia que la normativa vigente no contempla un procedimiento ni trámite específico para ese supuesto.
   - Señala que por tratarse de un caso no previsto, corresponde a la Coordinación elevar la solicitud a la Dirección del Programa / Dirección Académica para su evaluación excepcional.
   - QUEDA TERMINANTEMENTE PROHIBIDO inventar requisitos ficticios (como exigir 'historia clínica resumida', recetas médicas o certificados notariales) o inventar procedimientos arbitrarios de 5 pasos que no existan en los documentos.

2. ENFOQUE PRINCIPAL HACIA LA COORDINADORA:
   La destinataria principal de tu análisis es la Coordinadora Académica. Dale a ella un dictamen riguroso, seguro y con la lista de verificación de lo que debe controlar antes de emitir cualquier pronunciamiento.

3. ESTILO DEL BORRADOR PARA EL ESTUDIANTE:
   - Debe ser un correo institucional formal, comprensible y empático.
   - NO incluyas corchetes numéricos como [1] o [2] en el borrador para el alumno (las citas técnicas van en 'citas_normativas' para la coordinadora).
   - NO uses numeraciones rígidas de plantilla como '1. Diagnóstico', '2. Fundamentación normativa', '2.1', 'Paso 1, Paso 2'. Redáctalo con fluidez profesional.

4. CITAS TRAZABLES Y MÍNIMAS:
   - Cada cita DEBE usar el identificador F1, F2, etc., de un fragmento entregado.
   - Copia literalmente la oración o cláusula que fundamenta una afirmación concreta.
   - No inventes documento, página, artículo, sección ni extracto: esos metadatos se asignan por código.
   - Una misma regla = una sola cita, aunque se repita en varios fragmentos.
   - Pueden existir DOS citas de la misma página si sustentan reglas realmente distintas.
   - No rellenes una cuota de citas; elige solo las indispensables. Devuelve [] si no hay sustento literal.
   - Nunca cites un reglamento de Doctorado para acreditar el procedimiento de un Diploma de Educación Ejecutiva, ni viceversa.
   - En afirmaciones_citadas incluye SOLO los fundamentos imprescindibles para responder la pregunta; evita requisitos secundarios no solicitados.
   - Estas afirmaciones conformarán el criterio resolutivo mostrado a la coordinadora: deben ser comprensibles sin un segundo párrafo y no debes repetirlas en otro bloque.
   - cita_indices se refiere a las posiciones 1, 2, 3... de citas_normativas PROPUESTAS, NO a F1/F2.
   - Cada afirmación debe referir a una o más citas que realmente la respalden.
   - Si no existe sustento textual para una afirmación, no le inventes referencias.
   - No insertes [1] ni [2] directamente en el resumen; el servidor los incorpora.


5. ALCANCE NORMATIVO Y PRUDENCIA:
   - Distingue: (A) prohibición normativa expresa = 'No procede';
     (B) ausencia de regulación suficiente = 'Caso no contemplado expresamente — Requiere elevación'.
     La ausencia de evidencia NO constituye una prohibición normativa.
   - Si la pregunta no identifica el programa, NO extrapoles reglas de programas distintos:
     explica que falta identificar Diploma, Doctorado, Maestría, etc., antes del dictamen definitivo.
   - Un extracto que menciona 'tardanza' NO demuestra por sí solo que exista o no un trámite
     de 'justificación de inasistencias'. La pertinencia material es obligatoria.
   - Distingue retiro de un curso y retiro total del programa; no los confundas.
   - En el dictamen, distingue requisitos normativos de recomendaciones operativas.
   - La declaración de procedencia siempre está sujeta a comprobar la versión vigente
     y aplicabilidad del reglamento para el programa del alumno.
   - No declares que la coordinadora puede resolver directamente si indicas que debe
     decidir o aprobar la Dirección del Programa.
   - No trates correos, manuales operativos o documentos administrativos como reglamentos
     de autoridad normativa sin verificar su carácter, alcance y aprobación.
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

    def search_relevant_chunks(self, query: str, top_k: int = 6,
                               program_scope: str = None) -> List[Dict[str, Any]]:
        if not self.is_indexed():
            return []

        embed_resp = self.embed_client.embeddings.create(
            model="text-embedding-3-small",
            input=[query]
        )
        query_embedding = embed_resp.data[0].embedding

        fetch_k = min(top_k * 3, 24)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_k,
            include=["documents", "metadatas", "distances"]
        )

        raw_candidates = []
        if results and "documents" in results and results["documents"]:
            for doc_text, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0] if "distances" in results else [0] * fetch_k
            ):
                if not source_applies_to_scope(meta.get("source", ""), program_scope):
                    continue
                section = meta.get("section") or detect_section_or_article(doc_text)
                raw_candidates.append({
                    "text": doc_text.strip(),
                    "source": meta.get("source", "Documento ESAN"),
                    "page": meta.get("page", 1),
                    "section": section,
                    "score": round((1 - dist) * 100, 1) if dist is not None else None
                })

        # Desduplicación de los fragmentos recuperados: se evita repetir el
        # mismo texto, sin confundir "misma página" con "misma norma".
        candidates = []
        for cand in raw_candidates:
            candidate_text = normalize(cand["text"])
            if not candidate_text:
                continue
            duplicate = False
            for prior in candidates:
                old_text = normalize(prior["text"])
                if normalize(cand["source"]) == normalize(prior["source"]):
                    if candidate_text == old_text or (
                        min(len(candidate_text), len(old_text)) >= 120
                        and (candidate_text in old_text or old_text in candidate_text)
                    ):
                        duplicate = True
                        break
            if not duplicate:
                candidates.append(cand)

        # Diversificar primero por documento; completar luego con otros pasajes
        # relevantes del documento dominante cuando no haya suficiente diversidad.
        unique_sources = []
        counts = {}
        for max_per_document in (2, top_k):
            for cand in candidates:
                if len(unique_sources) >= top_k:
                    break
                if cand in unique_sources:
                    continue
                name = normalize(cand["source"])
                if counts.get(name, 0) >= max_per_document:
                    continue
                unique_sources.append(cand)
                counts[name] = counts.get(name, 0) + 1
            if len(unique_sources) >= top_k:
                break

        return unique_sources

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
        # El ámbito indicado en "Contexto extra" también influye en la búsqueda.
        program_scope = infer_program_scope(query, context_extra)
        retrieval_query = f"{query}\nPrograma / contexto: {context_extra}" if context_extra else query
        sources = self.search_relevant_chunks(
            retrieval_query, top_k=top_k, program_scope=program_scope
        )

        if not sources:
            indexed = self.is_indexed()
            message = (
                "No se recuperaron fragmentos aplicables al ámbito del programa. "
                "Verifique el programa y la documentación indexada."
                if indexed else "No se encontraron documentos indexados. Indexe la base documental."
            )
            empty = {
                "dictamen_coordinacion": {
                    "procedencia": "Sustento normativo insuficiente — Requiere revisión",
                    "resumen_ejecutivo": message,
                },
                "requiere_aprobacion": "Revisión de normativa por Coordinación/Dirección",
                "borrador_alumno": "No se dispone de sustento normativo verificable para emitir una respuesta al estudiante.",
                "acciones_coordinadora": ["Verificar el programa académico y los reglamentos indexados."],
                "razonamiento": message,
                "citas_normativas": [],
                "auditoria_citas": {"propuestas": 0, "duplicadas": 0, "no_verificables": 0, "aceptadas": 0},
                "validacion_pendiente": True,
            }
            return empty, []

        context_parts = []
        for i, s in enumerate(sources, 1):
            context_parts.append(
                f"[F{i} | documento: {s['source']} | sección: {s['section']} | página: {s['page']}]:\n{s['text']}"
            )
        context_str = "\n---\n".join(context_parts)

        user_content = f"""FRAGMENTOS DOCUMENTALES RECUPERADOS (pueden incluir documentos no normativos;
la búsqueda por similitud NO acredita vigencia ni pertinencia):
{context_str}

ALCANCE DEL PROGRAMA:
{program_scope or 'NO IDENTIFICADO: evita una decisión categórica si depende de saber el programa.'}

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
            max_output_tokens = (
                6000 if model_cfg["provider"] == "openrouter"
                else 2500
            )

            kwargs = dict(
                model=model_cfg["model_id"],
                messages=messages,
                temperature=0.15,
                max_tokens=max_output_tokens,
                response_format={"type": "json_object"},
            )
            try:
                response = client.chat.completions.create(**kwargs)
            except Exception:
                # Si algún modelo no soporta response_format, intentar sin él
                kwargs.pop("response_format", None)
                response = client.chat.completions.create(**kwargs)

            raw = response.choices[0].message.content.strip()
            if response.choices[0].finish_reason == "length":
                print(
                    "[RAG] ERROR: el modelo alcanzó el límite "
                    "de generación antes de completar su respuesta."
                )

                return {
                    "dictamen_coordinacion": {
                        "procedencia": "Error técnico — Respuesta incompleta",
                        "resumen_ejecutivo": (
                            "El modelo no completó la generación "
                            "del dictamen. Esto no demuestra "
                            "insuficiencia normativa."
                        )
                    },
                    "requiere_aprobacion": "No determinado",
                    "acciones_coordinadora": [
                        "Reintentar la consulta tras ajustar "
                        "el límite de generación."
                    ],
                    "razonamiento": (
                        "La generación fue interrumpida "
                        "por el límite de tokens."
                    ),
                    "borrador_alumno": "",
                    "citas_normativas": [],
                    "auditoria_citas": {
                        "propuestas": 0,
                        "duplicadas": 0,
                        "no_verificables": 0,
                        "aceptadas": 0
                    },
                    "validacion_pendiente": True,
                    "error_tecnico": "generation_length"
                }, sources
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
        except json.JSONDecodeError as e:
            print(f"[RAG] JSON inválido: {e.msg} (línea {e.lineno}, columna {e.colno})")
        except Exception as e:
            print(f"[RAG] Error inesperado en JSON: {type(e).__name__}: {e}")

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

        # 5. Rescate de campos cuando la respuesta completa no es JSON válido.
        # Se conservan las citas estructuradas: no se aceptan por defecto;
        # validate_citations comprueba posteriormente su literalidad y origen.
        if not parsed:
            parsed = {"citas_normativas": []}
            match_citas = _re.search(r'"citas_normativas"\s*:\s*\[', candidate)
            if match_citas:
                decoder = json.JSONDecoder(strict=False)
                pos = match_citas.end()
                while pos < len(candidate):
                    while pos < len(candidate) and candidate[pos] in " \t\r\n,":
                        pos += 1
                    if pos >= len(candidate) or candidate[pos] == "]":
                        break
                    try:
                        cita, consumed = decoder.raw_decode(candidate[pos:])
                    except json.JSONDecodeError as e:
                        print(f"[RAG] Cita no recuperable en posición {pos}: {e.msg}")
                        break
                    if (isinstance(cita, dict)
                            and isinstance(cita.get("fragmento_id"), str)
                            and isinstance(cita.get("extracto"), str)):
                        parsed["citas_normativas"].append(cita)
                    pos += consumed

            # Extracción robusta de borrador_alumno (tolera comillas internas sin escapar de MiMo)
            m_draft = _re.search(r'"borrador_alumno"\s*:\s*"(.*?)(?:"\s*,\s*"[a-zA-Z_]+"|"\s*\}\s*$)', candidate, _re.DOTALL)
            if not m_draft:
                m_draft = _re.search(r'"borrador_alumno"\s*:\s*"((?:\\.|[^"\\])*)"', candidate, _re.DOTALL)

            if m_draft:
                parsed["borrador_alumno"] = _clean_str(m_draft.group(1))
            else:
                # Buscar si hay un cuerpo de mensaje institucional que comience con Estimado
                m_est = _re.search(r'(Estimad[oa][\s\S]*?)(?:"\s*,\s*"[a-zA-Z_]+"|"\s*\}\s*$|\Z)', candidate)
                if m_est:
                    parsed["borrador_alumno"] = _clean_str(m_est.group(1))
                else:
                    parsed["borrador_alumno"] = ""

            m_appr = _re.search(r'"requiere_aprobacion"\s*:\s*"((?:\\.|[^"\\])*)"', candidate)
            parsed["requiere_aprobacion"] = _clean_str(m_appr.group(1)) if m_appr else "La coordinadora puede resolver directamente"

            m_reas = _re.search(r'"razonamiento"\s*:\s*"((?:\\.|[^"\\])*)"', candidate)
            parsed["razonamiento"] = _clean_str(m_reas.group(1)) if m_reas else ""

            # Extracción robusta de acciones_coordinadora para MiMo
            m_acc = _re.search(r'"acciones_coordinadora"\s*:\s*\[(.*?)\]', candidate, _re.DOTALL)
            if m_acc:
                items = _re.findall(r'"((?:\\.|[^"\\])*)"', m_acc.group(1))
                parsed["acciones_coordinadora"] = [_clean_str(it) for it in items if it.strip()]
            else:
                parsed["acciones_coordinadora"] = []

            # Extracción de dictamen_coordinacion si existe
            m_dict = _re.search(r'"dictamen_coordinacion"\s*:\s*\{([^}]*)\}', candidate, _re.DOTALL)
            if m_dict:
                m_proc = _re.search(r'"procedencia"\s*:\s*"((?:\\.|[^"\\])*)"', m_dict.group(1))
                m_sint = _re.search(r'"resumen_ejecutivo"\s*:\s*"((?:\\.|[^"\\])*)"', m_dict.group(1))
                parsed["dictamen_coordinacion"] = {
                    "procedencia": _clean_str(m_proc.group(1)) if m_proc else "Evaluación de Coordinación",
                    "resumen_ejecutivo": _clean_str(m_sint.group(1)) if m_sint else ""
                }

        # 6. Normalización de dictamen_coordinacion
        dictam = parsed.get("dictamen_coordinacion")
        if not isinstance(dictam, dict):
            # Derivar procedencia y síntesis a partir de los campos existentes
            req_appr = parsed.get("requiere_aprobacion", "")
            if "no previsto" in req_appr.lower() or "elevación" in req_appr.lower() or "no contemplado" in req_appr.lower():
                proc_label = "Caso no contemplado expresamente — Requiere elevación"
            elif "no procede" in parsed.get("borrador_alumno", "").lower()[:200]:
                proc_label = "No procede conforme a reglamento"
            else:
                proc_label = "Procede con verificación reglamentaria"
            parsed["dictamen_coordinacion"] = {
                "procedencia": proc_label,
                "resumen_ejecutivo": parsed.get("razonamiento", "")[:280] or "Dictamen institucional emitido con base en los reglamentos consultados."
            }

        # 7. Normalizar acciones_coordinadora (Garantiza que MiMo nunca quede en blanco)
        raw_acc = parsed.get("acciones_coordinadora", [])
        clean_acc = []
        if isinstance(raw_acc, str):
            clean_acc = [raw_acc]
        elif isinstance(raw_acc, list):
            clean_acc = [str(a).strip() for a in raw_acc if str(a).strip()]

        if not clean_acc:
            clean_acc = [
                "Verificar la condición académica y matrícula activa del estudiante en el sistema institucional.",
                "Revisar el expediente del alumno para constatar antecedentes o solicitudes previas vinculadas.",
                "Validar la aplicación del reglamento con la Dirección del Programa antes de emitir comunicación formal."
            ]
        parsed["acciones_coordinadora"] = clean_acc

        # 8. Limpieza del borrador del alumno (Eliminar corchetes [1], [2], encabezados robóticos y JSON crudo)
        d_text = parsed.get("borrador_alumno", "")
        if isinstance(d_text, str):
            # Si por algún motivo contiene JSON crudo (empieza con { o contiene "dictamen_coordinacion")
            if d_text.strip().startswith("{") or '"dictamen_coordinacion"' in d_text:
                m_inner = _re.search(r'"borrador_alumno"\s*:\s*"(.*?)(?:"\s*,\s*"[a-zA-Z_]+"|"\s*\}\s*$)', d_text, _re.DOTALL)
                if m_inner:
                    d_text = _clean_str(m_inner.group(1))
                else:
                    m_est = _re.search(r'(Estimad[oa][\s\S]*?)(?:"\s*,\s*"[a-zA-Z_]+"|"\s*\}\s*$|\Z)', d_text)
                    if m_est:
                        d_text = _clean_str(m_est.group(1))
                    else:
                        d_text = ""

            # Si sigue vacío o no se encontró mensaje, generar un borrador institucional limpio a partir del dictamen
            if not d_text.strip():
                res_ej = parsed.get("dictamen_coordinacion", {}).get("resumen_ejecutivo", "") or parsed.get("razonamiento", "")
                d_text = f"Estimado(a) estudiante,\n\nCon relación a su consulta sobre la aplicación de la normativa académica institucional, le informamos lo siguiente:\n\n{res_ej}\n\nQuedamos a su disposición para cualquier orientación adicional en su trámite.\n\nAtentamente,\nCoordinación Académica - Universidad ESAN"

            # Quitar corchetes numéricos tipo [1], [2], [1][2]
            d_text = _re.sub(r'\s*\[\d+\](?:\s*\[\d+\])*', '', d_text)
            # Quitar encabezados robóticos de plantilla como "1. Diagnóstico de su consulta", "2. Fundamentación"
            d_text = _re.sub(r'(?m)^\s*(?:\d+\.|\b\d+\.\d+\b)\s*(?:Diagnóstico|Fundamentación|Consecuencias|Procedimiento|Documentación|Requisitos)[^\n]*\n+', '', d_text)
            parsed["borrador_alumno"] = d_text.strip()
        else:
            parsed["borrador_alumno"] = ""


        # 9. Citas trazables: el servidor verifica extracto y asigna página/documento.
        citas, auditoria = validate_citations(parsed.get("citas_normativas", []), sources)






        # Resolver relaciones usando índices de citas propuestas antes de reemplazarlas.
        parsed["afirmaciones_citadas_validadas"] = link_claims_to_citations(
            parsed.get("afirmaciones_citadas", []),
            parsed.get("citas_normativas", []), citas, sources
        )
        parsed["citas_normativas"] = citas
        parsed["auditoria_citas"] = auditoria
        parsed["validacion_pendiente"] = not bool(citas)
        # Preguntas generales sobre faltas/retiros sin programa identificado
        # no se convierten automáticamente en prohibiciones universales.
        consulta_norm = normalize(query)
        falta_ambito = (not program_scope and any(
            palabra in consulta_norm for palabra in ("inasistencia", "justific", "tardanza", "retiro")
        ))
        if falta_ambito:
            parsed["validacion_pendiente"] = True
            dictamen = parsed.get("dictamen_coordinacion", {})
            if isinstance(dictamen, dict):
                dictamen["procedencia"] = "Alcance del programa no identificado — Requiere validación"
                dictamen["resumen_ejecutivo"] = (
                    "Antes de emitir un dictamen definitivo debe identificarse el programa "
                    "del estudiante y el reglamento aplicable. "
                    + str(dictamen.get("resumen_ejecutivo", ""))
                )
            parsed["requiere_aprobacion"] = "Identificar el programa y validar la norma con su Dirección"
            # No mostrar al alumno un rechazo universal generado para un caso ambiguo.
            parsed["borrador_alumno"] = (
                "Estimado(a) estudiante,\n\n"
                "Para orientarle correctamente, necesitamos confirmar a qué programa "
                "académico pertenece su consulta. Las disposiciones aplicables pueden "
                "variar según el reglamento del programa. Una vez confirmado, "
                "podremos revisar el procedimiento pertinente.\n\n"
                "Atentamente,\nCoordinación Académica - Universidad ESAN"
            )
        if not citas:
            parsed["afirmaciones_citadas_validadas"] = []
            # Un dictamen categórico sin evidencia verificable no se muestra como
            # decisión concluida. El borrador queda sujeto a revisión manual.
            dictamen = parsed.get("dictamen_coordinacion", {})
            if isinstance(dictamen, dict):
                dictamen["procedencia"] = "Sustento normativo insuficiente — Requiere revisión"
            parsed["requiere_aprobacion"] = "Validación normativa de la Dirección del Programa"
            if not falta_ambito:
                parsed["borrador_alumno"] = (
                    "Estimado(a) estudiante,\n\n"
                    "Su consulta requiere una verificación adicional de la normativa "
                    "aplicable. No podemos confirmar todavía el procedimiento o la "
                    "decisión institucional correspondiente. La Coordinación "
                    "Académica revisará los documentos y el programa pertinente "
                    "antes de emitir una respuesta formal.\n\n"
                    "Atentamente,\nCoordinación Académica - Universidad ESAN"
                )
        return parsed, sources

