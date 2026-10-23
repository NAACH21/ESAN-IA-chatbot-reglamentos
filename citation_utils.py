"""Validación determinista de citas normativas generadas por el modelo.

El modelo propone fragmento_id + extracto, pero NUNCA determina el documento,
la página ni el texto final mostrado. Esos datos provienen de ChromaDB.
"""

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple


def normalize(text: Any) -> str:
    """Normaliza espacios, puntuación y diacríticos para cotejar OCR/PDF."""
    decomposed = unicodedata.normalize("NFKD", str(text or "").casefold())
    clean = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", clean)).strip()


def _normalized_with_positions(text: str) -> Tuple[str, List[int]]:
    """Cadena normalizada y mapeo a índices del texto ORIGINAL del fragmento."""
    out: List[str] = []
    offsets: List[int] = []
    for pos, character in enumerate(text):
        folded = unicodedata.normalize("NFKD", character.casefold())
        folded = "".join(c for c in folded if not unicodedata.combining(c))
        for c in folded:
            if c in "abcdefghijklmnopqrstuvwxyz0123456789":
                out.append(c)
                offsets.append(pos)
            elif out and out[-1] != " ":
                out.append(" ")
                offsets.append(pos)
    while out and out[-1] == " ":
        out.pop()
        offsets.pop()
    return "".join(out), offsets


def locate_literal_quote(quote: str, source_text: str, max_chars: int = 230) -> Optional[Tuple[str, int]]:
    """Devuelve cita ORIGINAL y posición si la cita aparece realmente en el fragmento.

    No se emplea similitud semántica para aceptar una cita: una paráfrasis no
    constituye un extracto literal verificable.
    """
    normalized_quote = normalize(quote)
    if len(normalized_quote) < 24 or len(normalized_quote.split()) < 5:
        return None
    normalized_source, positions = _normalized_with_positions(source_text)
    start = normalized_source.find(normalized_quote)
    if start < 0:
        return None
    original_start = positions[start]
    original_end = positions[start + len(normalized_quote) - 1] + 1
    literal = source_text[original_start:original_end].strip()
    if len(literal) > max_chars:
        # En la interfaz se muestran extractos cortos, siempre literales.
        cut = literal[:max_chars]
        last_space = cut.rfind(" ")
        literal = cut[:last_space].rstrip() if last_space > max_chars // 2 else cut
    return literal, original_start


def _similar_evidence(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """Misma regla textual; nunca agrupa solo por coincidencia de página."""
    if normalize(a.get("documento")) != normalize(b.get("documento")):
        return False
    a_text, b_text = normalize(a.get("extracto")), normalize(b.get("extracto"))
    if not a_text or not b_text:
        return False
    if a_text in b_text or b_text in a_text:
        return True
    # Mismo pasaje copiado desde desplazamientos diferentes de un PDF.
    if str(a.get("pagina")) != str(b.get("pagina")):
        return False
    overlap = SequenceMatcher(None, a_text, b_text, autojunk=False).find_longest_match()
    return overlap.size / min(len(a_text), len(b_text)) >= 0.64 and overlap.size >= 42


def _section_near_quote(source_text: str, offset: int, default: str) -> str:
    """Detecta epígrafe inmediatamente anterior, si está dentro del chunk."""
    preceding = source_text[:offset]
    headings = list(re.finditer(
        r"(?m)^\s*(\d+(?:\.\d+){1,4})\.?\s+([^\n]{5,85})\s*$",
        preceding,
    ))
    if headings:
        match = headings[-1]
        return f"{match.group(1)}. {match.group(2).strip()}"
    return default or "Sección sin identificar"


def validate_citations(
    proposed: Any, sources: List[Dict[str, Any]], max_citations: int = 8
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Valida citas, asigna metadatos verificados, elimina evidencia repetida.

    El orden de sources define los identificadores F1, F2, ... que ve el LLM.
    La comparación NO elimina citas distintas de la misma página/documento.
    """
    stats = {"propuestas": 0, "duplicadas": 0, "no_verificables": 0, "aceptadas": 0}
    if not isinstance(proposed, list):
        return [], stats

    source_by_id = {f"F{i}": src for i, src in enumerate(sources, 1)}
    accepted: List[Dict[str, Any]] = []
    for c in proposed:
        if not isinstance(c, dict):
            continue
        stats["propuestas"] += 1
        fragment_id = str(c.get("fragmento_id", "")).strip().upper()
        quote = str(c.get("extracto") or "").strip()
        source = source_by_id.get(fragment_id)
        if source is None:
            # Compatibilidad con modelos que todavía devuelvan el formato viejo:
            # solo resuelve documento/página si el extracto coincide literalmente.
            doc_name, page = normalize(c.get("documento")), str(c.get("pagina", ""))
            legacy = [s for s in sources if normalize(s.get("source")) == doc_name
                      and str(s.get("page")) == page
                      and locate_literal_quote(quote, str(s.get("text", ""))) is not None]
            source = legacy[0] if legacy else None

        matched = locate_literal_quote(quote, str(source.get("text", ""))) if source else None
        if matched is None:
            stats["no_verificables"] += 1
            continue

        literal_quote, position = matched
        validated = {
            "numero": len(accepted) + 1,
            "documento": source.get("source", ""),
            "pagina": str(source.get("page", "")),
            "seccion": _section_near_quote(
                str(source.get("text", "")), position, str(source.get("section", ""))
            ),
            "extracto": literal_quote,
            "fragmento_id": fragment_id or "compatibilidad",
        }
        if any(_similar_evidence(validated, existing) for existing in accepted):
            stats["duplicadas"] += 1
            continue
        accepted.append(validated)
        if len(accepted) >= max_citations:
            break

    stats["aceptadas"] = len(accepted)
    return accepted, stats




def link_claims_to_citations(claims: Any, proposed: Any,
                             accepted: List[Dict[str, Any]],
                             sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Relaciona afirmaciones con citas *ya verificadas* sin inventar referencias.

    `cita_indices` son índices 1-based de `citas_normativas` PROPUESTAS.
    La numeración final pertenece a `accepted`, incluso si hubo deduplicación.
    No asume que un fragmento completo constituye una única cita.
    """
    if not isinstance(claims, list) or not isinstance(proposed, list):
        return []
    source_by_id = {f"F{i}": src for i, src in enumerate(sources, 1)}
    proposal_to_number = {}
    for i, proposal in enumerate(proposed, 1):
        if not isinstance(proposal, dict):
            continue
        frag = str(proposal.get("fragmento_id", "")).strip().upper()
        src = source_by_id.get(frag)
        if src is None:
            continue
        match = locate_literal_quote(str(proposal.get("extracto") or ""), str(src.get("text", "")))
        if match is None:
            continue
        candidate = {"documento": src.get("source", ""),
                     "pagina": str(src.get("page", "")), "extracto": match[0]}
        for citation in accepted:
            if _similar_evidence(candidate, citation):
                proposal_to_number[i] = citation["numero"]
                break

    linked = []
    for claim in claims[:12]:
        if not isinstance(claim, dict):
            continue
        statement = str(claim.get("texto") or "").strip()
        indices = claim.get("cita_indices", [])
        if not statement or len(statement) > 650 or not isinstance(indices, list):
            continue
        nums = []
        for idx in indices:
            if isinstance(idx, bool):
                continue
            try:
                index = int(idx)
            except (TypeError, ValueError):
                continue
            num = proposal_to_number.get(index)
            if num and num not in nums:
                nums.append(num)
        if nums:
            linked.append({"texto": statement, "numeros": nums})
    return linked


def infer_program_scope(question: str, context_extra: str = "") -> Optional[str]:
    """Detecta ámbitos inequívocos. No intenta adivinar el programa ausente."""
    text = normalize(f"{question} {context_extra}")
    categories = set()
    if re.search(r"\bdoctorad[oa]\b|\bdoctoral(?:es)?\b", text):
        categories.add("doctorado")
    if re.search(r"\bdiploma(?:do)?s?\b|\beducacion ejecutiva\b", text):
        categories.add("diploma")
    if re.search(r"\bmaestria(?:s)?\b", text):
        categories.add("maestria")
    return next(iter(categories)) if len(categories) == 1 else None


def source_applies_to_scope(filename: str, scope: Optional[str]) -> bool:
    """Filtro conservador: descarta reglamentos explícitamente de OTRO programa.

    La verificación definitiva de aplicabilidad y vigencia requiere revisión humana.
    """
    if scope is None:
        return True
    file = normalize(filename)
    doctoral = "doctorales" in file or "doctorado" in file
    executive = "educacion ejecutiva" in file or "educacion ejecutivos" in file
    masters = "maestrias" in file or "maestria" in file
    if scope == "diploma":
        return not doctoral and not masters
    if scope == "doctorado":
        return not executive and not masters
    if scope == "maestria":
        return not executive and not doctoral
    return True
