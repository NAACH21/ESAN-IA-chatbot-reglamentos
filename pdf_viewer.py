"""Vista PDF local bajo demanda, con resaltado amarillo en memoria."""
from pathlib import Path
import unicodedata
import re

from ingest import DOCS_FOLDER


def _normalized_name(value: str) -> str:
    return unicodedata.normalize('NFKC', value).strip().casefold()


def locate_local_pdf(document_name: str):
    """Resuelve solo nombres simples de PDF bajo DOCS, nunca rutas del usuario."""
    raw = str(document_name or '').strip()
    if not raw or Path(raw).name != raw or not raw.lower().endswith('.pdf'):
        return None
    root = Path(DOCS_FOLDER).resolve()
    if not root.is_dir():
        return None
    target = _normalized_name(raw)
    for candidate in root.rglob('*.pdf'):
        if any(p.startswith(('_', '.')) for p in candidate.relative_to(root).parts[:-1]):
            continue
        if candidate.is_file() and _normalized_name(candidate.name) == target:
            resolved = candidate.resolve()
            if root in resolved.parents:
                return resolved
    return None


def _tokens(s):
    s = unicodedata.normalize('NFKD', s).casefold()
    s = ''.join(c for c in s if not unicodedata.combining(c))
    return re.findall(r'[a-z0-9]+', s)


def _find_word_span(page, excerpt):
    """Localiza SOLO secuencias contiguas exactas de palabras; nunca coincidencia difusa."""
    target = _tokens(excerpt)
    if len(target) < 3:
        return []
    words = page.get_text('words', sort=True)
    expanded = []
    for idx, word in enumerate(words):
        for token in _tokens(word[4]):
            expanded.append((token, idx))
    terms = [w[0] for w in expanded]
    for i in range(len(terms) - len(target) + 1):
        if terms[i:i + len(target)] == target:
            indices = sorted({expanded[j][1] for j in range(i, i + len(target))})
            return [words[j] for j in indices]
    return []


def render_pdf_page(pdf_path, page_number: int, excerpt: str = '', scale: float = 1.35):
    """PNG de página solicitada, total de páginas, si se resaltó el extracto.

    Las anotaciones solo afectan el documento abierto en memoria; no guarda el PDF.
    """
    import fitz
    if not isinstance(page_number, int) or isinstance(page_number, bool) or page_number < 1:
        raise ValueError('Número de página inválido.')
    with fitz.open(str(pdf_path)) as pdf:
        total = len(pdf)
        if page_number > total:
            raise ValueError(f'Página {page_number} fuera de rango (1–{total}).')
        page = pdf.load_page(page_number - 1)
        marked = False
        if excerpt and excerpt.strip():
            # Primero coincidencia textual directa (admite saltos de línea de PDF).
            quads = page.search_for(excerpt.strip(), quads=True)
            if quads:
                annot = page.add_highlight_annot(quads)
                annot.set_colors(stroke=(1.0, 0.84, 0.05))
                annot.update()
                marked = True
            else:
                # Alternativa determinista para espacios y puntuación del texto extraído.
                words = _find_word_span(page, excerpt)
                if words:
                    for w in words:
                        rect = fitz.Rect(w[:4])
                        annot = page.add_highlight_annot(rect)
                        annot.set_colors(stroke=(1.0, 0.84, 0.05))
                        annot.update()
                    marked = True
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False, annots=True)
        return pix.tobytes('png'), total, marked
