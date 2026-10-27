import os
import sys
import re
import base64
from typing import Any
from html import escape
import importlib
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import ingest as _ingest_mod
importlib.reload(_ingest_mod)
import rag_engine as _rag_mod
importlib.reload(_rag_mod)
import drive_sync as _drive_mod
importlib.reload(_drive_mod)
import pdf_viewer as _pdf_mod
importlib.reload(_pdf_mod)
from pdf_viewer import resolve_local_doc_path, generate_pdf_viewer_html, parse_page_number

from ingest import (
    build_vector_store,
    index_missing_documents,
    remove_document_from_store,
    index_single_pdf,
    restore_archived_documents,
    is_supported_file,
    get_all_supported_files,
    DOCS_FOLDER,
    ARCHIVE_FOLDER,
)
from rag_engine import RAGEngine, MODELS, GOOGLE_DRIVE_FOLDER
from pdf_viewer import locate_local_pdf, render_pdf_page
from drive_sync import (
    sync_from_drive,
    download_full_folder_gdown,
    list_drive_files,
    get_indexed_sources,
    get_file_drive_url,
)

load_dotenv()


@st.cache_data(show_spinner=False, max_entries=24)
def _cached_pdf_page(pdf_path: str, pdf_mtime_ns: int, page: int, excerpt: str):
    # pdf_mtime_ns invalida caché si cambia el archivo.
    return render_pdf_page(pdf_path, page, excerpt=excerpt)


@st.dialog("Página citada del reglamento", width="large")
def show_pdf_dialog(document_name: str, page_number: int, excerpt: str):
    """Vista de una única página del PDF; no mantiene widgets de navegación."""
    local_pdf = locate_local_pdf(document_name)
    if local_pdf is None:
        st.warning("No se encontró este documento en DOCS. Utiliza el enlace a Google Drive.")
        return

    try:
        png, total, marked = _cached_pdf_page(
            str(local_pdf), local_pdf.stat().st_mtime_ns, page_number, excerpt
        )
        st.caption(f"{document_name} · Página {page_number} de {total}")
        if excerpt:
            if marked:
                st.caption("Pasaje citado resaltado en amarillo.")
            else:
                st.info("No se pudo localizar literalmente el extracto en esta página. Comprueba el pasaje original.")
        # La imagen se ajusta al espacio visible del modal, manteniendo la
        # proporción de la hoja y sin exigir desplazarse por una página ampliada.
        import base64
        encoded = base64.b64encode(png).decode("ascii")
        st.markdown(
            '<div style="display:flex;justify-content:center;width:100%;">'
            '<img alt="Página citada del documento normativo" '
            'style="display:block;max-width:100%;max-height:65vh;object-fit:contain;'
            'width:auto;height:auto;" '
            f'src="data:image/png;base64,{encoded}">'
            '</div>',
            unsafe_allow_html=True,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        st.error(f"No se pudo visualizar esta página: {exc}")

# ── CARGA SILENCIOSA DE CREDENCIALES (Sin mostrar en la interfaz) ───────────────
api_key = os.getenv("OPENAI_API_KEY", "") or os.getenv("openai_api_key", "")
try:
    if not api_key and hasattr(st, "secrets"):
        api_key = st.secrets.get("OPENAI_API_KEY", "") or st.secrets.get("openai_api_key", "") or ""
except Exception:
    pass

or_key = os.getenv("OPENROUTER_API_KEY", "") or os.getenv("openrouter_api_key", "") or os.getenv("MIMO_API_KEY", "") or ""
try:
    if not or_key and hasattr(st, "secrets"):
        or_key = st.secrets.get("OPENROUTER_API_KEY", "") or st.secrets.get("openrouter_api_key", "") or st.secrets.get("MIMO_API_KEY", "") or ""
except Exception:
    pass


# ── CARGA DE LOGO OFICIAL SVG EN BASE64 ───────────────────────────────────────
def get_esan_logo_base64():
    try:
        p = os.path.join(os.path.dirname(__file__), "assets", "logo_esan.svg")
        if os.path.exists(p):
            with open(p, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        pass
    return ""

ESAN_LOGO_B64 = get_esan_logo_base64()

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sistema Normativo · Universidad ESAN",
    page_icon="https://cdn.worldvectorlogo.com/logos/universidad-esan.svg",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── SVG ICONS INSTITUCIONALES (CERO EMOJIS) ───────────────────────────────────
SVG = {
    "scale": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#D31418" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:6px"><path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="M7 21h10"/><path d="M12 3v18"/><path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"/></svg>""",
    "db": """<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#D31418" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:6px"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>""",
    "book": """<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#D31418" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:6px"><path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1-2.5-2.5Z"/><path d="M6 6h10"/><path d="M6 10h10"/></svg>""",
    "file": """<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#D31418" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:4px"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>""",
    "check": """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#16a34a" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:4px"><polyline points="20 6 9 17 4 12"/></svg>""",
    "shield_check": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#D97706" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-3px;margin-right:6px"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10"/><path d="m9 12 2 2 4-4"/></svg>""",
    "drive": """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#1967D2" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:4px"><path d="M21 15a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V9a2 2 0 0 1 2-2h1l2-3h6l2 3h1a2 2 0 0 1 2 2z"/></svg>""",
    "pin": """<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#64748B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-1px;margin-right:3px"><path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/><circle cx="12" cy="10" r="3"/></svg>""",
    "chip": """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#475569" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:4px"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M9 1v3"/><path d="M15 1v3"/><path d="M9 20v3"/><path d="M15 20v3"/><path d="M20 9h3"/><path d="M20 14h3"/><path d="M1 9h3"/><path d="M1 14h3"/></svg>""",
    "alert": """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#D97706" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:5px"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>""",
    "search": """<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#0F172A" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:5px"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>""",
    "clipboard_check": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#D31418" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:6px"><rect width="8" height="4" x="8" y="2" rx="1" ry="1"/><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/><path d="m9 14 2 2 4-4"/></svg>""",
    "grad_cap": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#D31418" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:6px"><path d="M22 10v6M2 10l10-5 10 5-10 5z"/><path d="M6 12v5c3 3 9 3 12 0v-5"/></svg>""",
    "sparkles": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#D31418" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:6px"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/></svg>""",
}

# ── STYLING INSTITUCIONAL ESAN ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* V3.2: desplazamiento suave y centrado visual de citas vinculadas. */
html,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    scroll-behavior: smooth;
}

/* La tarjeta se sitúa aproximadamente en el centro del área visible. */
.cite-card[id^="cita-"] {
    scroll-margin-top: calc(50vh - 75px);
}

/* El destino del enlace se ilumina brevemente y vuelve a su estilo normal. */
@keyframes espan-cita-destacada {
    0%   { background-color: rgba(250, 204, 21, 0.25); box-shadow: 0 0 0 3px rgba(234, 179, 8, 0.4); }
    65%  { background-color: rgba(250, 204, 21, 0.16); box-shadow: 0 0 0 2px rgba(234, 179, 8, 0.2); }
    100% { background-color: transparent; box-shadow: none; }
}
.cite-card[id^="cita-"]:target {
    animation: espan-cita-destacada 1.6s ease-out 1;
}
@media (prefers-reduced-motion: reduce) {
    html, [data-testid="stAppViewContainer"], [data-testid="stMain"] { scroll-behavior: auto; }
    .cite-card[id^="cita-"]:target { animation: none; outline: 2px solid #D4A017; }
}

/* Navbar Institucional Superior */
.esan-navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 22px;
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-top: 3.5px solid #D31418;
    border-radius: 10px;
    margin-bottom: 22px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.03);
}
.esan-nav-left {
    display: flex;
    align-items: center;
    gap: 16px;
}
.esan-nav-logo {
    height: 38px;
    width: auto;
    display: block;
    object-fit: contain;
}
.esan-nav-divider {
    width: 1px;
    height: 32px;
    background: #E2E8F0;
}
.esan-nav-title {
    font-size: 1.05rem;
    font-weight: 800;
    color: #0F172A;
    letter-spacing: -0.01em;
    line-height: 1.25;
}
.esan-nav-subtitle {
    font-size: 0.78rem;
    color: #64748B;
    margin-top: 2px;
    font-weight: 500;
}
.esan-nav-right {
    display: flex;
    align-items: center;
    gap: 8px;
}
.esan-status-indicator {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 6px;
    padding: 6px 12px;
}
.pulse-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #16A34A;
    display: inline-block;
}
.status-text {
    font-size: 0.76rem;
    font-weight: 600;
    color: #334155;
}

/* Tarjetas y Contenedores */
.esan-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 18px 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.02);
    margin-bottom: 16px;
}
.esan-card-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #0F172A;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 1px solid #F1F5F9;
}

/* Badge de Aprobación */
.appr-badge {
    display: flex;
    align-items: center;
    gap: 12px;
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 14px;
}
.appr-icon {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: #FEF3C7;
    border: 1px solid #F59E0B;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

/* Borrador del Alumno */
.draft-box {
    background: #FAFAFA;
    border: 1px solid #E2E8F0;
    border-left: 3.5px solid #D31418;
    border-radius: 8px;
    padding: 16px;
    font-size: 0.92rem;
    line-height: 1.65;
    color: #1E293B;
    white-space: pre-wrap;
    margin-bottom: 12px;
}

/* Tarjeta de Cita Normativa */
.cite-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-left: 3.5px solid #D31418;
    border-radius: 8px;
    padding: 12px 14px;
    margin-bottom: 12px;
    transition: all 0.2s ease;
    box-shadow: 0 1px 3px rgba(0,0,0,0.02);
}
.cite-card:hover {
    border-color: #FECACA;
    border-left-color: #D31418;
    box-shadow: 0 3px 10px rgba(211,20,24,0.05);
}
.cite-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    background: #D31418;
    color: #FFFFFF;
    font-size: 0.72rem;
    font-weight: 800;
    margin-right: 8px;
    flex-shrink: 0;
}
.cite-source {
    font-weight: 700;
    font-size: 0.88rem;
    color: #0F172A;
}
.cite-section {
    color: #64748B;
    font-size: 0.8rem;
    margin-top: 2px;
}
.cite-excerpt {
    background: #FFFBEB;
    border-radius: 4px;
    padding: 6px 10px;
    margin-top: 8px;
    font-size: 0.82rem;
    font-style: italic;
    color: #475569;
    border-left: 2px solid #F59E0B;
    line-height: 1.45;
}
.drive-link {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    color: #1967D2;
    text-decoration: none;
    font-size: 0.8rem;
    font-weight: 600;
    padding: 4px 8px;
    background: #F0F7FF;
    border-radius: 6px;
    border: 1px solid #DBEAFE;
    transition: all 0.15s ease;
}
.drive-link:hover {
    background: #E0E7FF;
    text-decoration: none;
}

/* Botón Ver página: estilo secundario armonizado con el enlace de Drive. */
.st-key-open_pdf_modal_1 button,
[class*="st-key-open_pdf_modal_"] button {
    color: #1967D2 !important;
    background: #F0F7FF !important;
    border: 1px solid #DBEAFE !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    min-height: 34px !important;
    transition: background 0.15s ease !important;
}
[class*="st-key-open_pdf_modal_"] button:hover {
    color: #174EA6 !important;
    background: #E0E7FF !important;
    border-color: #BFDBFE !important;
}
/* Enlace original de Drive: el estilo drive-link no se modifica. */

/* Razonamiento */
.reason-box {
    background: #F8FAFC;
    border-left: 3px solid #64748B;
    padding: 10px 14px;
    border-radius: 4px;
    font-size: 0.84rem;
    color: #334155;
    line-height: 1.55;
}

/* Welcome Hero Cards */
.hero-box {
    text-align: center;
    padding: 24px 20px 18px 20px;
}
.hero-title {
    font-size: 1.45rem;
    font-weight: 800;
    color: #0F172A;
    margin-bottom: 6px;
}
.hero-desc {
    font-size: 0.9rem;
    color: #64748B;
    max-width: 680px;
    margin: 0 auto;
    line-height: 1.5;
}

/* Botón primario en Rojo ESAN */
button[kind="primary"] {
    background-color: #D31418 !important;
    border-color: #D31418 !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}
button[kind="primary"]:hover {
    background-color: #B20D10 !important;
    border-color: #B20D10 !important;
    box-shadow: 0 4px 12px rgba(211,20,24,0.22) !important;
}

/* Status pills */
.status-pill-ok {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #F0FDF4;
    border: 1px solid #BBF7D0;
    color: #166534;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 0.78rem;
    font-weight: 600;
}
.status-pill-model {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #FEF2F2;
    border: 1px solid #FECACA;
    color: #991B1B;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 0.78rem;
    font-weight: 700;
}
.sb-section {
    font-size: 0.85rem;
    font-weight: 700;
    color: #0F172A;
    margin: 14px 0 8px 0;
    display: flex;
    align-items: center;
}
</style>
""", unsafe_allow_html=True)

# ── REDISEÑO INTERFAZ: CAPA DE ESTILOS OPCIONAL ─────────────────────────────
from pathlib import Path as _esan_Path
_esan_css_path = _esan_Path(__file__).resolve().parent / "styles" / "redisenio.css"
if _esan_css_path.is_file():
    st.markdown("<style>" + _esan_css_path.read_text(encoding="utf-8") + "</style>", unsafe_allow_html=True)

# ── ESAN F3: SISTEMA VISUAL ESTRUCTURAL ─────────────────────────────────────
_esan_v3_css_path = _esan_Path(__file__).resolve().parent / "styles" / "esan_v3.css"
if _esan_v3_css_path.is_file():
    st.markdown("<style>" + _esan_v3_css_path.read_text(encoding="utf-8") + "</style>", unsafe_allow_html=True)

# ESAN-F4-STYLE-LOADER
_esan_v4_path = _esan_Path(__file__).resolve().parent / "styles" / "esan_v4.css"
if _esan_v4_path.is_file():
    st.markdown("<style>" + _esan_v4_path.read_text(encoding="utf-8") + "</style>", unsafe_allow_html=True)

# ESAN-F5-STYLE-LOADER
_esan_v5_path = _esan_Path(__file__).resolve().parent / "styles" / "esan_v5.css"
if _esan_v5_path.is_file():
    st.markdown("<style>" + _esan_v5_path.read_text(encoding="utf-8") + "</style>", unsafe_allow_html=True)

# ESAN-F6-STYLE-LOADER
_esan_v6_path = _esan_Path(__file__).resolve().parent / "styles" / "esan_v6.css"
if _esan_v6_path.is_file():
    st.markdown("<style>" + _esan_v6_path.read_text(encoding="utf-8") + "</style>", unsafe_allow_html=True)

# ── SESSION STATE ─────────────────────────────────────────────────────────────
for key, default in [
    ("history", []),
    ("last_result", None),
    ("last_sources", []),
    ("last_program", None),
    ("last_query", None),
    ("last_extra_ctx", None),
    ("quick_q", None),
    ("form_id", 0),
    ("scroll_to_top", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── SIDEBAR (ELECCIÓN DE MODELOS, BASE DOCUMENTAL, SYNC DRIVE) ─────────────────
with st.sidebar:
    if ESAN_LOGO_B64:
        st.markdown(f'<img src="data:image/svg+xml;base64,{ESAN_LOGO_B64}" width="180" style="margin-bottom:8px" />', unsafe_allow_html=True)
    else:
        st.markdown('<h3 style="color:#D31418;margin:0">UNIVERSIDAD ESAN</h3>', unsafe_allow_html=True)

    st.caption("Coordinación Académica · Marco Normativo")

    if api_key:
        st.markdown(f'<div class="status-pill-ok">{SVG["check"]} Conexión reglamentaria activa</div>', unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:#FFFBEB;border:1px solid #FDE68A;border-radius:8px;padding:10px 12px;margin:8px 0">
          <div style="font-weight:700;font-size:0.82rem;color:#92400E;display:flex;align-items:center;gap:6px">
            {SVG['alert']} Claves no configuradas
          </div>
          <div style="font-size:0.75rem;color:#78350F;margin-top:4px;line-height:1.4">
            En Streamlit Cloud ve a <b>Settings &gt; Secrets</b> y agrega:<br>
            <code style="display:block;margin-top:4px;padding:3px;background:#FEF3C7;border-radius:4px;font-size:0.72rem">OPENAI_API_KEY = "tu_key"</code>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 1. SELECCIÓN DE MODELO
    st.markdown(f'<div class="sb-section">{SVG["chip"]} Motor de Dictamen</div>', unsafe_allow_html=True)
    model_label = st.radio(
        "Seleccionar Modelo",
        options=list(MODELS.keys()),
        index=0,
        label_visibility="collapsed",
        help="Seleccione el modelo según la complejidad del dictamen normativo."
    )
    model_cfg = MODELS[model_label]
    if model_cfg["provider"] == "openai":
        st.markdown(f'<div class="status-pill-model">{SVG["check"]} OpenAI · GPT-4o-mini (Estándar)</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="status-pill-model">{SVG["check"]} OpenRouter · MiMo v2.6 Pro (Razonamiento)</div>', unsafe_allow_html=True)

    st.markdown("---")

    # 2. BASE DOCUMENTAL
    st.markdown(f'<div class="sb-section">{SVG["db"]} Repositorio Normativo</div>', unsafe_allow_html=True)
    rag = RAGEngine(api_key=api_key, openrouter_key=or_key) if api_key else None
    is_indexed = rag.is_indexed() if rag else False

    if is_indexed:
        cnt = rag.get_document_count()
        st.markdown(f'<div style="font-size:0.8rem;color:#475569;display:flex;align-items:center;gap:6px;margin-bottom:8px">{SVG["book"]} <b>{cnt:,}</b> secciones normativas indexadas</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="font-size:0.8rem;color:#B45309;display:flex;align-items:center;gap:6px;margin-bottom:8px">{SVG["alert"]} Repositorio pendiente de inicialización</div>', unsafe_allow_html=True)
        if api_key and os.path.exists(DOCS_FOLDER):
            if st.button("Indexar los 43 reglamentos ahora", type="primary", use_container_width=True, key="btn_init_index"):
                with st.spinner("Indexando los 43 reglamentos..."):
                    res = index_missing_documents(api_key=api_key)
                    st.success("¡Reglamentos indexados exitosamente!")
                    st.rerun()


    # 3. SINCRONIZACIÓN CON GOOGLE DRIVE
    with st.expander("Sincronizar Google Drive"):
        drive_url_input = st.text_input(
            "Carpeta Drive",
            value=GOOGLE_DRIVE_FOLDER,
            label_visibility="collapsed",
            placeholder="https://drive.google.com/drive/folders/..."
        )
        c_sync1, c_sync2 = st.columns(2)
        with c_sync1:
            if st.button("Ver en Drive", use_container_width=True, key="btn_sb_drive_preview"):
                with st.spinner("Consultando Drive..."):
                    try:
                        dfiles = list_drive_files(drive_url_input)
                        idx_set = set(get_indexed_sources())
                        nuevos = [f for f in dfiles if f["name"] not in idx_set]
                        st.info(f"{len(dfiles)} docs en Drive ({len(nuevos)} nuevos)")
                    except Exception as e:
                        st.error(str(e))
        with c_sync2:
            if st.button("Sincronizar", use_container_width=True, disabled=not bool(api_key), key="btn_sb_drive_sync"):
                with st.spinner("Descargando e indexando nuevos..."):
                    try:
                        s_res = sync_from_drive(drive_url_input, docs_dir=DOCS_FOLDER, progress_callback=lambda m: None)
                        n_added = s_res.get("new_files", [])
                        for nf in n_added:
                            p = os.path.join(DOCS_FOLDER, nf)
                            if os.path.exists(p):
                                index_single_pdf(api_key, p)
                        st.success(f"{len(n_added)} nuevo(s) sincronizado(s).")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

    # 4. GESTIÓN DE ARCHIVOS LOCALES
    if os.path.exists(DOCS_FOLDER):
        all_docs = get_all_supported_files(DOCS_FOLDER)
        with st.expander(f"Documentos locales ({len(all_docs)})"):
            for i_d, p in enumerate(all_docs):
                d = os.path.basename(p)
                sub_rel = os.path.relpath(p, DOCS_FOLDER).replace("\\", "/")
                c1, c2 = st.columns([4, 1.2])
                with c1:
                    st.markdown(f"{SVG['file']}<small style='word-break:break-all'>{sub_rel}</small>", unsafe_allow_html=True)
                with c2:
                    if st.button("Quitar", key=f"del_{i_d}", help=f"Quitar {d}"):
                        n_rm = remove_document_from_store(d)
                        st.toast(f"'{d}' eliminado ({n_rm} secciones)")
                        st.rerun()

            if st.button("Indexar documentos nuevos", use_container_width=True, disabled=not bool(api_key)):
                with st.spinner("Indexando..."):
                    res = index_missing_documents(api_key=api_key)
                    st.info(res["message"])
                    st.rerun()

    # 5. LIMPIAR SESIÓN
    st.markdown("---")
    if st.button("Nueva consulta / Limpiar", use_container_width=True):
        st.session_state.history = []
        st.session_state.last_result = None
        st.session_state.last_sources = []
        st.session_state.last_program = None
        st.session_state.last_query = None
        st.session_state.last_extra_ctx = None
        st.session_state.quick_q = None
        st.session_state.form_id += 1
        st.rerun()

# ── MODAL DIALOG: VISOR NORMATIVO CON PDF.JS Y RESALTADO ─────────────────────
@st.dialog("Visor Normativo Oficial · Verificación de Cita", width="large")
def show_pdf_viewer_modal(
    doc_name: str,
    page_val: Any,
    extracto: str,
    seccion: str,
    fallback_source: str = "",
    drive_url: str = "",
):
    local_path = resolve_local_doc_path(doc_name, fallback_source=fallback_source)
    page_int = parse_page_number(page_val)

    if local_path and os.path.exists(local_path):
        ext = os.path.splitext(local_path)[1].lower()
        if ext == ".pdf":
            try:
                with open(local_path, "rb") as f:
                    pdf_b64 = base64.b64encode(f.read()).decode("utf-8")

                viewer_html = generate_pdf_viewer_html(
                    pdf_b64=pdf_b64,
                    initial_page=page_int,
                    search_text=extracto,
                    doc_title=os.path.basename(local_path),
                    section_name=seccion,
                )
                components.html(viewer_html, height=720, scrolling=False)
            except Exception as e:
                st.error(f"Error al cargar el documento en el visor: {e}")
        else:
            st.info(f"El documento normativo **{os.path.basename(local_path)}** está en formato {ext.upper()}.")
            if extracto:
                st.markdown(f"""
                <div style="background:#FFFBEB;border:1px solid #FDE68A;border-radius:8px;padding:12px;margin:10px 0">
                  <div style="font-size:0.78rem;font-weight:700;color:#92400E;margin-bottom:4px">FRAGMENTO TEXTUAL CITADO:</div>
                  <div style="font-size:0.88rem;color:#1E293B;line-height:1.4">"{extracto}"</div>
                </div>
                """, unsafe_allow_html=True)
            with open(local_path, "rb") as f:
                st.download_button(
                    label=f"Descargar archivo local ({os.path.basename(local_path)})",
                    data=f.read(),
                    file_name=os.path.basename(local_path),
                    use_container_width=True
                )
    else:
        st.warning(f"No se encontró copia local de '{doc_name}' en DOCS/. Puedes acceder al archivo en Google Drive:")
        if drive_url:
            st.markdown(f'<a href="{drive_url}" target="_blank" class="drive-link">{SVG["drive"]} Abrir documento oficial en Drive</a>', unsafe_allow_html=True)


# ── TOP NAVBAR INSTITUCIONAL REIMAGINADO (SIN FORMATO IA DE JUGUETE) ─────────
# Ancla invisible para scroll-to-top via JS
st.markdown('<div id="top-anchor"></div>', unsafe_allow_html=True)

# Auto-scroll al top cuando hay resultado nuevo
if st.session_state.get("scroll_to_top"):
    components.html(
        """
        <script>
        setTimeout(function() {
            try {
                const parentDoc = window.parent.document;
                const main = parentDoc.querySelector("section.main") || parentDoc.querySelector(".main") || parentDoc.documentElement;
                if (main) {
                    main.scrollTo({top: 0, behavior: "smooth"});
                }
            } catch (e) {
                window.parent.scrollTo(0, 0);
            }
        }, 150);
        </script>
        """,
        height=0,
        width=0,
    )
    st.session_state.scroll_to_top = False

logo_html = f'<img src="data:image/svg+xml;base64,{ESAN_LOGO_B64}" alt="Universidad ESAN" class="esan-nav-logo" />' if ESAN_LOGO_B64 else '<div style="font-weight:900;color:#D31418;font-size:1.15rem">ESAN</div>'

st.markdown(f"""
<div class="esan-navbar">
  <div class="esan-nav-left">
    {logo_html}
    <div class="esan-nav-divider"></div>
    <div>
      <div class="esan-nav-title">SISTEMA INSTITUCIONAL DE CONSULTA Y GESTIÓN NORMATIVA</div>
      <div class="esan-nav-subtitle">Coordinación Académica · Documentación institucional indexada</div>
    </div>
  </div>
  <div class="esan-nav-right">
    <div class="esan-status-indicator">
      <span class="pulse-dot"></span>
      <span class="status-text">Repositorio documental · vigencia por confirmar</span>
    </div>
    <div class="status-pill-model">
      {SVG["shield_check"]} Asistencia para revisión normativa
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── ÁREA PRINCIPAL: RESULTADOS O PANTALLA DE BIENVENIDA ───────────────────────
result = st.session_state.last_result
sources = st.session_state.last_sources

# ── ESAN-F4-WORKSPACE: FORMULARIO IZQUIERDA, RESPUESTA DERECHA ─────────────
esan_workspace_left, esan_workspace_right = st.columns([0.36, 0.64], gap="medium")
with esan_workspace_left:
    with st.container():
        st.markdown('<p class="esan4-panel-eyebrow esan4-panel-left">01 / CONSULTA Y CONTEXTO</p>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("""<div class="esan3-form-marker"></div>""", unsafe_allow_html=True)
            st.markdown("""
            <div class="esan3-form-head">
              <span class="esan3-number">02</span>
              <div class="esan3-form-head-text">
                <span>CONSULTA PERSONALIZADA</span>
                <h2>Consulta o Caso del Estudiante</h2>
                <p>Selecciona el programa, describe el caso y revisa los parámetros antes de emitir un dictamen.</p>
              </div>
              <span class="esan3-form-head-note">ANÁLISIS SUJETO A VALIDACIÓN</span>
            </div>
            """, unsafe_allow_html=True)

            current_form_id = st.session_state.form_id

            # ── SELECTOR DE PROGRAMA (OBLIGATORIO) ────────────────────────────────
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:6px;font-size:0.82rem;font-weight:700;color:#0F172A;margin-bottom:4px">'
                f'{SVG["grad_cap"]} Tipo de programa del participante '
                '<span style="color:#D31418;font-weight:900">*</span>'
                '<span style="font-weight:400;color:#64748B;font-size:0.75rem;margin-left:4px">(obligatorio)</span>'
                '</div>',
                unsafe_allow_html=True
            )
            programa_options = ["PEE", "PADE", "PAE", "Diploma", "CIE", "Maestría", "Doctorado"]
            # ESAN-F5-POLISH · selector adaptativo (mismo key y valor)
            if hasattr(st, "segmented_control"):
                programa_sel = st.segmented_control(
                    "Programa",
                    options=programa_options,
                    selection_mode="single",
                    default=None,
                    label_visibility="collapsed",
                    key=f"prog_choice_{current_form_id}",
                )
            else:
                if hasattr(st, "pills"):
                    programa_sel = st.pills(
                        "Programa",
                        options=programa_options,
                        selection_mode="single",
                        default=None,
                        label_visibility="collapsed",
                        key=f"prog_choice_{current_form_id}",
                    )
                else:
                    programa_sel = st.radio(
                        "Programa", options=programa_options, index=None,
                        horizontal=False, label_visibility="collapsed",
                        key=f"prog_choice_{current_form_id}",
                    )

            if not programa_sel:
                st.markdown(
                    '<div style="color:#D31418;font-size:0.78rem;margin-bottom:6px;display:flex;align-items:center;gap:4px">'
                    f'{SVG["alert"]} Debe seleccionar el tipo de programa antes de emitir el dictamen.'
                    '</div>',
                    unsafe_allow_html=True
                )
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

            default_prompt = st.session_state.quick_q or ""
            student_msg = st.text_area(
                "Consulta del estudiante",
                value=default_prompt,
                height=90,
                placeholder="Pegue o redacte aquí la consulta del estudiante (ej: Alumno solicita rendir evaluación extemporánea por descanso médico)...",
                label_visibility="collapsed",
                max_chars=4000,
                key=f"query_text_{current_form_id}",
            )

            # Si se cargo desde quick_q, limpiar para que no persista en futuros reruns
            if st.session_state.quick_q:
                st.session_state.quick_q = None

            with st.expander("Parámetros opcionales del caso (ciclo, antecedentes del alumno)"):
                extra_ctx = st.text_input(
                    "Contexto extra",
                    placeholder="Ej: ciclo V, alumno presenta solicitud fuera de plazo de 48 horas...",
                    label_visibility="collapsed",
                    key=f"extra_ctx_{current_form_id}",
                )

            can_run = bool(api_key and student_msg.strip() and programa_sel)

            # Botón principal prominente con feedback visual
            button_help = "Haga clic para consultar la base reglamentaria y generar el dictamen." if can_run else "Seleccione el tipo de programa y escriba la consulta para activar este botón."
            run_query = st.button(
                "▶  Consultar reglamentos y emitir dictamen",
                use_container_width=True,
                type="primary",
                disabled=not can_run,
                help=button_help,
            )
            if st.session_state.history:
                st.markdown(
                    f'<div style="font-size:0.75rem;color:#64748B;text-align:center;padding-top:10px">'
                    f'{len(st.session_state.history)} consulta(s) previas</div>',
                    unsafe_allow_html=True
                )
with esan_workspace_right:
    esan_response_slot = st.empty()  # ESAN-F6-EXPERIENCE: respuesta sustituible
    with esan_response_slot.container():
        with st.container():
            st.markdown('<p class="esan4-panel-eyebrow esan4-panel-right">02 / RESULTADO Y EVIDENCIAS</p>', unsafe_allow_html=True)
            if result is None:
                # ESAN-F3-COMPOSITION · Tarjetas unificadas con botones funcionales nativos.
                st.markdown("""
                <div class="esan3-hero">
                  <div class="esan3-hero-main">
                    <div class="esan3-eyebrow"><span class="esan3-mark"></span>CENTRO DE CONSULTAS ACADÉMICAS</div>
                    <h1>Consultas Reglamentarias y Casuística Estudiantil</h1>
                    <p>Consulta procedimientos institucionales o analiza un caso estudiantil con respaldo documental.
                       Una experiencia de trabajo preparada para Coordinación Académica.</p>
                  </div>
                  <div class="esan3-hero-aside">
                    <div class="esan3-hero-aside-symbol"></div>
                    <span>ORIENTACIÓN INSTITUCIONAL</span>
                    <strong>Asistencia normativa con IA</strong>
                    <small>Todo dictamen requiere verificación de vigencia y aplicabilidad.</small>
                  </div>
                </div>
                <div class="esan3-section-intro">
                   <div><span class="esan3-number">01</span><div><h2>Elige el tipo de consulta</h2><p>Accesos rápidos a procedimientos frecuentes</p></div></div>
                   <span class="esan3-section-meta">4 categorías disponibles</span>
                </div>
                """, unsafe_allow_html=True)
                c_c1, c_c2 = st.columns(2, gap="medium")
                with c_c1:
                    with st.container(border=True):
                        st.markdown(f"""
                        <div class="esan3-card-head">
                            <span class="esan3-icon">{SVG['scale']}</span>
                            <div class="esan3-card-content">
                              <span class="esan3-card-category">PROCEDIMIENTOS ACADÉMICOS</span>
                              <h3>Reconsideración de Evaluaciones</h3>
                              <p>Plazos reglamentarios, requisitos formales e instancia resolutiva ante reclamos de calificación.</p>
                            </div>
                            <span class="esan3-card-index">01</span>
                        </div>
                        """, unsafe_allow_html=True)
                        if st.button("Consultar plazos y trámite de reconsideración →", key="q_recons", use_container_width=True):
                            st.session_state.quick_q = "¿Cuándo puedo pedir reconsideración de nota y ante qué autoridad corresponde presentarla?"
                            st.session_state.form_id += 1
                            st.rerun()
                    with st.container(border=True):
                        st.markdown(f"""
                        <div class="esan3-card-head">
                            <span class="esan3-icon esan3-icon-blue">{SVG['chip']}</span>
                            <div class="esan3-card-content">
                              <span class="esan3-card-category">DIRECTIVAS INSTITUCIONALES</span>
                              <h3>Lineamientos de Inteligencia Artificial</h3>
                              <p>Disposiciones institucionales para docentes y alumnos sobre el uso de herramientas IA en evaluaciones.</p>
                            </div>
                            <span class="esan3-card-index">03</span>
                        </div>
                        """, unsafe_allow_html=True)
                        if st.button("Consultar directiva sobre uso de IA →", key="q_ia", use_container_width=True):
                            st.session_state.quick_q = "¿Cuáles son los lineamientos sobre el uso de Inteligencia Artificial para docentes y alumnos en ESAN?"
                            st.session_state.form_id += 1
                            st.rerun()
                with c_c2:
                    with st.container(border=True):
                        st.markdown(f"""
                        <div class="esan3-card-head">
                            <span class="esan3-icon esan3-icon-blue">{SVG['grad_cap']}</span>
                            <div class="esan3-card-content">
                              <span class="esan3-card-category">GRADOS ACADÉMICOS</span>
                              <h3>Requisitos para Grado de Bachiller</h3>
                              <p>Cómputo de créditos académicos, acreditación obligatoria de idiomas y condiciones de graduación.</p>
                            </div>
                            <span class="esan3-card-index">02</span>
                        </div>
                        """, unsafe_allow_html=True)
                        if st.button("Consultar requisitos de bachillerato →", key="q_bach", use_container_width=True):
                            st.session_state.quick_q = "¿Cuáles son los requisitos y créditos necesarios para obtener el Grado de Bachiller?"
                            st.session_state.form_id += 1
                            st.rerun()
                    with st.container(border=True):
                        st.markdown(f"""
                        <div class="esan3-card-head">
                            <span class="esan3-icon esan3-icon-gold">{SVG['book']}</span>
                            <div class="esan3-card-content">
                              <span class="esan3-card-category">PROCESOS ADMINISTRATIVOS</span>
                              <h3>Procedimiento de Devolución y Retiro</h3>
                              <p>Plazos de desistimiento, cronograma de retenciones y trámite formal de retiro de programa.</p>
                            </div>
                            <span class="esan3-card-index">04</span>
                        </div>
                        """, unsafe_allow_html=True)
                        if st.button("Consultar política de devoluciones y retiros →", key="q_ret", use_container_width=True):
                            st.session_state.quick_q = "¿Cuál es el procedimiento formal y plazos para solicitar devolución de dinero o retiro de programa?"
                            st.session_state.form_id += 1
                            st.rerun()
            else:
                # ── PANEL DE CONSULTA EVALUADA (CONSULTA DEL ESTUDIANTE + PROGRAMA) ────────
                last_q = st.session_state.get("last_query", "")
                last_prog = st.session_state.get("last_program", "")
                last_extra = st.session_state.get("last_extra_ctx", "")

                if last_q:
                    badge_prog_html = (
                        f'<span style="background:#FEF2F2;border:1.5px solid #FECACA;color:#991B1B;'
                        f'padding:3px 12px;border-radius:6px;font-size:0.8rem;font-weight:800;letter-spacing:0.03em;'
                        f'display:inline-flex;align-items:center;gap:6px">'
                        f'{SVG["grad_cap"]} Programa: {last_prog}</span>'
                    ) if last_prog else ''

                    extra_html = (
                        f'<div style="font-size:0.8rem;color:#475569;margin-top:8px;padding-top:8px;'
                        f'border-top:1px dashed #CBD5E1;display:flex;align-items:center;gap:6px">'
                        f'<span style="font-weight:700;color:#0F172A">Contexto del caso ingresado:</span> {last_extra}</div>'
                    ) if last_extra else ''

                    st.markdown(f"""
                    <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-left:5px solid #D31418;border-radius:10px;padding:14px 18px;margin-bottom:18px;box-shadow:0 1px 4px rgba(0,0,0,0.04)">
                      <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;margin-bottom:8px">
                        <div style="font-size:0.75rem;font-weight:800;color:#64748B;text-transform:uppercase;letter-spacing:0.05em;display:flex;align-items:center;gap:6px">
                          {SVG["search"]} Consulta del Estudiante Evaluada
                        </div>
                        {badge_prog_html}
                      </div>
                      <div style="font-size:0.95rem;color:#0F172A;font-weight:600;line-height:1.45">
                        "{last_q}"
                      </div>
                      {extra_html}
                    </div>
                    """, unsafe_allow_html=True)

                # ── VISTA DE RESULTADOS (2 COLUMNAS: RESPUESTA + RESPALDO NORMATIVO) ───────
                st.markdown("""
                <div class="esan3-results-intro">
                   <span class="esan3-results-kicker">EXPEDIENTE DE CONSULTA</span>
                   <h2>Resultado del análisis normativo</h2>
                   <p>Consulta las conclusiones preliminares y su respaldo documental. La validación institucional prevalece sobre la asistencia automática.</p>
                </div>
                """, unsafe_allow_html=True)
                col_resp, col_norm = st.tabs(["Dictamen y validación", f"Citas y documentos ({len(result.get('citas_normativas', []))})"])

                # ═════════════════════════════════════════ COLUMNA 1: RESPUESTA PROPUESTA ═
                with col_resp:
                    st.markdown('<span class="esan3-colmark esan3-col-resp"></span>', unsafe_allow_html=True)
                    _prog_sel = st.session_state.get('last_program', '') or st.session_state.get('programa_sel', '')
                    _prog_badge = (
                        f'<span style="background:#FEF2F2;border:1px solid #FECACA;color:#991B1B;'
                        f'border-radius:6px;padding:2px 10px;font-size:0.75rem;font-weight:700;margin-left:8px"'
                        f'>{_prog_sel}</span>'
                    ) if _prog_sel else ''
                    st.markdown(
                        f'<div class="esan-card-title">{SVG["scale"]} Dictamen Técnico y Validación para Coordinación{_prog_badge}</div>',
                        unsafe_allow_html=True
                    )

                    # 1. DICTAMEN EJECUTIVO Y PROCEDENCIA (RESPUESTA PRINCIPAL PARA LA COORDINADORA)
                    dictamen = result.get("dictamen_coordinacion", {})
                    procedencia = dictamen.get("procedencia", "Evaluación de Coordinación Académica")
                    resumen_ejecutivo = dictamen.get("resumen_ejecutivo", "") or result.get("razonamiento", "")
                    appr = result.get("requiere_aprobacion", "La coordinadora puede resolver directamente")

                    # Estilo de badge según procedencia
                    proc_lower = procedencia.lower()
                    if "no procede" in proc_lower:
                        badge_bg, badge_border, badge_text = "#FEF2F2", "#FECACA", "#991B1B"
                        icon_badge = SVG["alert"]
                    elif "no contemplado" in proc_lower or "elevación" in proc_lower or "insuficiente" in proc_lower or "revisión" in proc_lower or "validación" in proc_lower:
                        badge_bg, badge_border, badge_text = "#FFFBEB", "#FDE68A", "#92400E"
                        icon_badge = SVG["alert"]
                    else:
                        badge_bg, badge_border, badge_text = "#F0FDF4", "#BBF7D0", "#166534"
                        icon_badge = SVG["check"]

                    # V3.1: las afirmaciones referenciadas sustituyen al resumen libre
                    # dentro del recuadro. Nunca se duplica el criterio en otro apartado.
                    # Los números proceden exclusivamente de citas ya validadas por Python.
                    claims_linked = result.get("afirmaciones_citadas_validadas", [])
                    citas_aceptadas = result.get("citas_normativas", [])
                    partes_criterio = []
                    if not result.get("validacion_pendiente") and isinstance(claims_linked, list):
                        for claim in claims_linked:
                            if not isinstance(claim, dict):
                                continue
                            texto = str(claim.get("texto", "")).strip()
                            numeros = []
                            for numero in claim.get("numeros", []):
                                if isinstance(numero, int) and not isinstance(numero, bool) and 1 <= numero <= len(citas_aceptadas) and numero not in numeros:
                                    numeros.append(numero)
                            if texto and numeros:
                                refs = " ".join(
                                    f'<a href="#cita-{n}" style="color:#1D4ED8;font-weight:700;text-decoration:none">[{n}]</a>'
                                    for n in numeros
                                )
                                partes_criterio.append(f"{escape(texto)} {refs}")

                    if partes_criterio:
                        criterio_html = " ".join(partes_criterio)
                    else:
                        # Sin relaciones verificadas, se conserva el resumen sin insertar
                        # referencias arbitrarias. La interfaz conserva su estado de revisión.
                        criterio_html = escape(resumen_ejecutivo)

                    st.markdown(f"""
                    <div style="background:{badge_bg};border:1.5px solid {badge_border};border-radius:8px;padding:12px 16px;margin-bottom:12px">
                      <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
                        <span style="font-weight:800;font-size:0.88rem;color:{badge_text};text-transform:uppercase;letter-spacing:0.02em">
                          {icon_badge} {escape(procedencia)}
                        </span>
                      </div>
                      <div style="font-size:0.86rem;color:#1E293B;line-height:1.5;margin-bottom:8px">
                        <b>Criterio resolutivo:</b> {criterio_html}
                      </div>
                      <div style="font-size:0.8rem;color:#475569;display:flex;align-items:center;gap:6px;border-top:1px solid rgba(0,0,0,0.06);padding-top:6px">
                        {SVG['shield_check']} <span><b>Instancia competente:</b> {escape(appr)}</span>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # 2. PROTOCOLO DE VERIFICACIÓN PARA LA COORDINADORA (PROMINENTE)
                    actions = result.get("acciones_coordinadora", [])
                    if actions:
                        st.markdown(f"<div style='margin:14px 0 6px 0;display:flex;align-items:center;gap:6px;'><b style='font-size:0.88rem;color:#0F172A'>{SVG['clipboard_check']} Protocolo de verificación en sistema y expediente</b></div>", unsafe_allow_html=True)
                        for act in actions:
                            st.checkbox(act, value=False, key=f"chk_{hash(act)}")

                    # 3. FUNDAMENTACIÓN TÉCNICA
                    reason = result.get("razonamiento", "")
                    if reason:
                        with st.expander("Ver fundamentación y razonamiento normativo"):
                            st.markdown(f'<div class="reason-box">{escape(reason)}</div>', unsafe_allow_html=True)

                    st.markdown("<hr style='margin:16px 0'>", unsafe_allow_html=True)

                    # 4. BORRADOR DE COMUNICACIÓN PARA EL ESTUDIANTE (POST-VALIDACIÓN)
                    st.markdown("<div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:8px'><b style='font-size:0.88rem;color:#0F172A'>Borrador de comunicación para el estudiante (Post-validación)</b></div>", unsafe_allow_html=True)

                    draft = result.get("borrador_alumno", "")
                    # Salvaguarda final: si contiene JSON crudo, extraer solo el mensaje humano
                    if isinstance(draft, str) and (draft.strip().startswith("{") or '"dictamen_coordinacion"' in draft):
                        import re as _re_ui
                        m_inner = _re_ui.search(r'"borrador_alumno"\s*:\s*"(.*?)(?:"\s*,\s*"[a-zA-Z_]+"|"\s*\}\s*$)', draft, _re_ui.DOTALL)
                        if not m_inner:
                            m_inner = _re_ui.search(r'(Estimad[oa][\s\S]*?)(?:"\s*,\s*"[a-zA-Z_]+"|"\s*\}\s*$|\Z)', draft)
                        if m_inner:
                            draft = m_inner.group(1).replace('\\n', '\n').replace('\\"', '"').strip()
                        else:
                            draft = result.get("dictamen_coordinacion", {}).get("resumen_ejecutivo", "") or result.get("razonamiento", "")

                    st.markdown(f'<div class="draft-box">{escape(draft)}</div>', unsafe_allow_html=True)

                    # Botones de Copia Rápidos
                    b_c1, b_c2 = st.columns(2)
                    copia_bloqueada = bool(result.get("validacion_pendiente"))
                    if copia_bloqueada:
                        st.warning("El borrador requiere validación del ámbito o de su sustento antes de compartirse.")
                    wa_text = f"*Coordinación Académica ESAN*\n\n{draft}"
                    # El borrador ya incluye saludo; no repetirlo en el correo copiado.
                    mail_text = draft

                    with b_c1:
                        if st.button("Copiar formato WhatsApp", key="btn_cp_wa", use_container_width=True, disabled=copia_bloqueada):
                            st.code(wa_text, language=None)
                            st.toast("Texto preparado para WhatsApp — cópialo del recuadro")
                    with b_c2:
                        if st.button("Copiar formato Correo", key="btn_cp_mail", use_container_width=True, disabled=copia_bloqueada):
                            st.code(mail_text, language=None)
                            st.toast("Texto preparado para Correo — cópialo del recuadro")

                # ═════════════════════════════════════════ COLUMNA 2: RESPALDO NORMATIVO ═══
                with col_norm:
                    st.markdown('<span class="esan3-colmark esan3-col-norm"></span>', unsafe_allow_html=True)
                    citas = result.get("citas_normativas", [])
                    documentos = len({c.get("documento", "") for c in citas})
                    # ESAN-F6-EXPERIENCE · datos verificados vs. material contextual
                    referencias_relacionadas = not bool(citas)
                    fragmentos_recuperados = len(sources)
                    documentos_relacionados = len({str(s.get("source", "")) for s in sources if s.get("source")})
                    st.markdown(
                        f'<div class="esan6-evidence-header"><div class="esan6-eyebrow">TRAZABILIDAD DOCUMENTAL</div>'
                        f'<h3>Evidencia documental y referencias</h3>'
                        f'<div class="esan6-evidence-metrics">'
                        f'<div class="esan6-metric"><strong>{len(citas)}</strong><span>Citas textuales verificadas</span></div>'
                        f'<div class="esan6-metric"><strong>{documentos if citas else documentos_relacionados}</strong>'
                        f'<span>{"Documentos con citas verificadas" if citas else "Documentos relacionados"}</span></div>'
                        f'</div></div>', unsafe_allow_html=True,
                    )
                    st.caption(
                        f"Se recuperaron {len(sources)} fragmentos para el análisis. "
                        "Solo se muestran extractos que coinciden literalmente con un fragmento recuperado; "
                        "esto no certifica vigencia ni aplicabilidad jurídica."
                    )
                    auditoria = result.get("auditoria_citas", {})
                    if auditoria.get("duplicadas") or auditoria.get("no_verificables"):
                        st.caption(
                            f"Control automático: {auditoria.get('duplicadas', 0)} citas duplicadas "
                            f"y {auditoria.get('no_verificables', 0)} citas sin coincidencia literal descartadas."
                        )
                    if not citas:
                        st.warning(
                            "No hay citas textuales verificadas para respaldar este dictamen. "
                            "La búsqueda recuperó material contextual, pero no constituye por sí sola "
                            "prueba normativa. No emitir la comunicación sin revisión institucional."
                        )
                    elif result.get("validacion_pendiente"):
                        st.warning(
                            "Hay citas literales, pero todavía falta confirmar el programa y "
                            "la aplicabilidad normativa. No emitir una respuesta definitiva hasta verificarlo."
                        )

                    with st.container(height=520, border=False):
                        if referencias_relacionadas:
                            st.markdown(
                                '<div class="esan6-related-intro">'
                                '<span class="esan6-related-dot"></span>'
                                '<div><strong>Material relacionado con la consulta</strong>'
                                '<p>Estos fragmentos fueron recuperados por similitud temática. '
                                'No son citas verificadas ni acreditan por sí mismos la vigencia o aplicabilidad de una norma.</p>'
                                '</div></div>', unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                '<div class="esan6-verified-intro">Extractos con coincidencia textual verificada. '
                                'Su vigencia y aplicabilidad deben confirmarse.</div>', unsafe_allow_html=True,
                            )
                        raw_items = {}
                        for c in citas:
                            num = c.get("numero", 0)
                            raw_items[num] = c

                        if not raw_items:
                            for i_s, s in enumerate(sources[:6], 1):
                                raw_items[i_s] = {
                                    "numero": i_s,
                                    "documento": s["source"],
                                    "seccion": s["section"],
                                    "pagina": str(s["page"]),
                                    "extracto": s["text"][:220],
                                }

                        # ── Agrupar citas por documento (todas las citas y recuadros en una sola tarjeta por archivo) ──
                        import unicodedata as _ud

                        def _norm_docname(name):
                            n = _ud.normalize("NFD", name.lower())
                            n = "".join(c for c in n if _ud.category(c) != "Mn")
                            return n.strip()

                        grouped = {}   # key = norm_docname -> dict con info del doc y lista de citas
                        order = []
                        for num, c in sorted(raw_items.items()):
                            doc_name_raw = c.get("documento", "")
                            nkey = _norm_docname(doc_name_raw)
                            if nkey not in grouped:
                                fb_src = ""
                                fb_pg = ""
                                if isinstance(num, int) and 1 <= num <= len(sources):
                                    fb_src = sources[num - 1].get("source", "")
                                    fb_pg = str(sources[num - 1].get("page", ""))
                                grouped[nkey] = {
                                    "doc_name": doc_name_raw or fb_src,
                                    "fallback_source": fb_src,
                                    "citas": [],
                                }
                                order.append(nkey)

                            grp = grouped[nkey]
                            # Extraer solo el número de página limpio (sin duplicar "Pág.")
                            p_raw = c.get("pagina", "")
                            p_digits = re.findall(r"\d+", str(p_raw))
                            clean_p = p_digits[0] if p_digits else str(p_raw).strip()

                            grp["citas"].append({
                                "num": num,
                                "seccion": c.get("seccion", ""),
                                "pagina": clean_p,
                                "extracto": c.get("extracto", ""),
                            })

                        for card_idx, nkey in enumerate(order, 1):
                            grp = grouped[nkey]
                            doc_name = grp["doc_name"]
                            fallback_source = grp["fallback_source"]
                            all_citas = grp["citas"]
                            first_pg = all_citas[0]["pagina"] if all_citas else ""

                            doc_drive_url = get_file_drive_url(
                                filename=doc_name,
                                page=first_pg,
                                fallback_filename=fallback_source,
                                fallback_page=first_pg,
                            )

                            safe_doc_name = escape(doc_name)
                            safe_drive_url = escape(doc_drive_url, quote=True)

                            # Tarjeta contenedora única del Documento
                            st.markdown(f"""
                            <div class="cite-card {'esan6-related-card' if referencias_relacionadas else 'esan6-verified-card'}" style="margin-bottom:12px;">
                              <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:8px;margin-bottom:6px">
                                <div style="display:flex;align-items:flex-start;gap:8px">
                                  <span class="cite-num">{card_idx}</span>
                                  <div>
                                    <div class="cite-source">{safe_doc_name}</div>
                                    <div style="font-size:0.75rem;color:#64748B;margin-top:2px">{len(all_citas)} {"fragmento(s) relacionados" if referencias_relacionadas else "cita(s) textuales"} en este documento</div>
                                  </div>
                                </div>
                                <a href="{safe_drive_url}" target="_blank" rel="noopener noreferrer" class="drive-link" style="font-size:0.75rem;white-space:nowrap">
                                  {SVG["drive"]} Ver en Drive
                                </a>
                              </div>
                            """, unsafe_allow_html=True)

                            # Mostrar TODAS las citas / recuadros amarillos de este documento
                            for sub_i, cita in enumerate(all_citas, 1):
                                c_sec = cita["seccion"]
                                c_pg = cita["pagina"]
                                c_ext = cita["extracto"]

                                badge_pg = f'<span style="background:#FEF2F2;color:#991B1B;padding:2px 8px;border-radius:5px;font-size:0.72rem;font-weight:700;border:1px solid #FECACA">Pág. {escape(str(c_pg))}</span>' if c_pg else ""
                                sec_text = f'<span style="font-size:0.82rem;font-weight:700;color:#1E293B">{escape(str(c_sec))}</span>' if c_sec else ""

                                st.markdown(f"""
                                <div style="margin-top:10px;padding-top:8px;border-top:1px dashed #E2E8F0">
                                  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px">
                                    <div>{sec_text}</div>
                                    <div>{badge_pg}</div>
                                  </div>
                                  {f'<div class="cite-excerpt" style="margin-bottom:8px">"{escape(str(c_ext))}..."</div>' if c_ext else ''}
                                </div>
                                """, unsafe_allow_html=True)

                                btn_label = f"🔍 Ver y resaltar en PDF (Pág. {c_pg})" if c_pg else "🔍 Ver y resaltar en PDF"
                                if st.button(
                                    btn_label,
                                    key=f"btn_pdf_modal_{card_idx}_{sub_i}_{cita['num']}",
                                    use_container_width=True,
                                    type="primary",
                                    help=f"Abre el PDF oficial en la página {c_pg} y resalta este extracto exacto."
                                ):
                                    show_pdf_viewer_modal(
                                        doc_name=doc_name,
                                        page_val=c_pg,
                                        extracto=c_ext,
                                        seccion=c_sec,
                                        fallback_source=fallback_source,
                                        drive_url=doc_drive_url,
                                    )

                            st.markdown("</div>", unsafe_allow_html=True)
                            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ── PROCESAMIENTO DE CONSULTA ─────────────────────────────────────────────────
if run_query and api_key and student_msg.strip():
    # Banner de carga visual y prominente
    # ESAN-F6-EXPERIENCE · el estado de carga aparece en el panel de respuesta
    esan_response_slot.empty()
    with esan_response_slot.container():
        st.markdown(
            f'<section class="esan6-loading" role="status" aria-live="polite">'
            f'<span class="esan6-loading-spinner" aria-hidden="true"></span>'
            f'<div><span class="esan6-eyebrow">ANÁLISIS EN CURSO</span>'
            f'<h2>Preparando el dictamen · {escape(str(programa_sel))}</h2>'
            f'<p>Consultando reglamentos, contrastando referencias y elaborando una propuesta sujeta a validación institucional.</p>'
            f'</div></section>', unsafe_allow_html=True,
        )
    try:
        if not is_indexed and api_key:
            build_vector_store(api_key=api_key)
            rag = RAGEngine(api_key=api_key, openrouter_key=or_key)

        result_data, sources_data = rag.answer_question(
            query=student_msg.strip(),
            context_extra=extra_ctx.strip() if 'extra_ctx' in locals() and extra_ctx else "",
            model_label=model_label,
            # Consultas de otros alumnos no deben contaminar este dictamen
            chat_history=None,
            programa=programa_sel or "",
        )
        st.session_state.last_result = result_data
        st.session_state.last_sources = sources_data
        st.session_state.last_program = programa_sel
        st.session_state.last_query = student_msg.strip()
        st.session_state.last_extra_ctx = extra_ctx.strip() if 'extra_ctx' in locals() and extra_ctx else ""
        st.session_state.history.append(student_msg.strip()[:120])

        # Avanzar form_id: limpia automáticamente el textarea y deselecciona el programa sin tocar keys bloqueadas
        st.session_state.form_id += 1
        st.session_state.scroll_to_top = True
        esan_response_slot.empty()
        st.rerun()
    except Exception as e:
        esan_response_slot.empty()
        with esan_response_slot.container():
            st.error(f"Error al procesar la consulta: {e}")
