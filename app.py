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


# ── CARGA SILENCIOSA DE CREDENCIALES ───────────────────────────────────────────
api_key = os.getenv("OPENAI_API_KEY", "") or os.getenv("openai_api_key", "")
try:
    if not api_key and hasattr(st, "secrets"):
        api_key = st.secrets.get("OPENAI_API_KEY", "") or st.secrets.get("openai_api_key", "") or ""
except Exception:
    pass

or_key = os.getenv("OPENROUTER_API_KEY", "") or os.getenv("openrouter_api_key", "") or os.getenv("MIMO_API_KEY", "") or ""
try:
    if not or_key and hasattr(st, "secrets"):
        or_key = st.secrets.get("OPENROUTER_API_KEY", "") or st.secrets.get("OPENROUTER_API_KEY", "") or st.secrets.get("MIMO_API_KEY", "") or ""
except Exception:
    pass


# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ESAN · Asistente Normativo Académico",
    page_icon="https://cdn.worldvectorlogo.com/logos/universidad-esan.svg",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── SVG ICONS INSTITUCIONALES (CERO EMOJIS · GAMA ROJOS ESAN) ─────────────────
SVG = {
    "book_brand": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#6E0B1C" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>""",
    "sparkles_white": """<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/><path d="M5 3v4"/><path d="M19 17v4"/><path d="M3 5h4"/><path d="M17 19h4"/></svg>""",
    "sparkles_crimson": """<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#C8102E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/><path d="M5 3v4"/><path d="M3 5h4"/></svg>""",
    "sparkles_rose": """<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#FECDD3" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/><path d="M5 3v4"/><path d="M3 5h4"/></svg>""",
    "scale": """<svg xmlns="http://www.w3.org/2000/svg" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="#C8102E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="M7 21h10"/><path d="M12 3v18"/><path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"/></svg>""",
    "book": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#C8102E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>""",
    "book_sb": """<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.8)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>""",
    "check": """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#16A34A" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>""",
    "check_crimson": """<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#C8102E" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>""",
    "shield_check": """<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#C8102E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10"/><path d="m9 12 2 2 4-4"/></svg>""",
    "drive": """<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#C8102E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V9a2 2 0 0 1 2-2h1l2-3h6l2 3h1a2 2 0 0 1 2 2z"/></svg>""",
    "alert": """<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#C8102E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>""",
    "search": """<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#C8102E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>""",
    "clipboard_check": """<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#C8102E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="8" height="4" x="8" y="2" rx="1" ry="1"/><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/><path d="m9 14 2 2 4-4"/></svg>""",
    "grad_cap": """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#C8102E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 10v6M2 10l10-5 10 5-10 5z"/><path d="M6 12v5c3 3 9 3 12 0v-5"/></svg>""",
    "bookmark": """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#C8102E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m19 21-7-4-7 4V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v16z"/></svg>""",
}

# ── ESTILOS MINIMALISTAS Y LIMPIOS CON IDENTIDAD CROMÁTICA ESAN ───────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --esan-crimson: #C8102E;
    --esan-red: #D31418;
    --esan-dark: #5C0916;
    --esan-darker: #3D050E;
    --esan-rose-bg: #FFF1F2;
    --esan-rose-border: #FECDD3;
    --esan-rose-text: #9F1239;
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Lienzo principal ultra limpio */
[data-testid="stAppViewContainer"] {
    background-color: #F7F8FA;
}

.block-container {
    padding-top: 0.8rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1220px !important;
}

header[data-testid="stHeader"] {
    background: transparent !important;
    height: 1rem !important;
}

html, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    scroll-behavior: smooth;
}
.cite-card[id^="cita-"] {
    scroll-margin-top: calc(50vh - 75px);
}
@keyframes espan-cita-destacada {
    0%   { background-color: rgba(200, 16, 46, 0.14); box-shadow: 0 0 0 3px rgba(200, 16, 46, 0.25); }
    100% { background-color: transparent; box-shadow: none; }
}
.cite-card[id^="cita-"]:target {
    animation: espan-cita-destacada 1.6s ease-out 1;
}

/* ═══════════════════════════════════════════════════════════════════════════
   SIDEBAR MINIMALISTA EN GAMA VINO PROFUNDO ESAN
   ═══════════════════════════════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #5E0A17 0%, #460711 65%, #34040C 100%) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.07) !important;
}

[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
    padding-top: 1.2rem !important;
}

/* Brand Header estilo limpio de la maqueta */
.esan-sb-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 22px;
    padding: 0 2px;
}
.esan-sb-badge {
    width: 38px;
    height: 38px;
    border-radius: 10px;
    background: #FFF1F2;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.18);
    flex-shrink: 0;
}
.esan-sb-title {
    font-size: 1.05rem;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: 0.08em;
    line-height: 1.1;
}
.esan-sb-subtitle {
    font-size: 0.62rem;
    font-weight: 600;
    color: rgba(255, 255, 255, 0.72);
    letter-spacing: 0.14em;
    margin-top: 2px;
}
.esan-sb-bars {
    display: flex;
    gap: 4px;
    margin-top: 5px;
}
.esan-sb-bar-red {
    width: 18px;
    height: 2.5px;
    background: #FF4D5E;
    border-radius: 2px;
}
.esan-sb-bar-white {
    width: 18px;
    height: 2.5px;
    background: rgba(255, 255, 255, 0.75);
    border-radius: 2px;
}

/* Botón "+ Nueva conversación / consulta" en blanco puro arriba del sidebar */
.st-key-btn_sb_new button {
    background: #FFFFFF !important;
    color: #460711 !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 0.84rem !important;
    min-height: 44px !important;
    justify-content: flex-start !important;
    padding-left: 16px !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.16) !important;
    transition: all 0.18s ease !important;
}
.st-key-btn_sb_new button p::before {
    content: "+";
    font-size: 1.1rem;
    font-weight: 600;
    margin-right: 10px;
    color: #C8102E;
    vertical-align: -1px;
}
.st-key-btn_sb_new button:hover {
    background: #FFF1F2 !important;
    transform: translateY(-1px) !important;
}

.sb-group-title {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: rgba(255, 255, 255, 0.52);
    margin: 22px 0 10px 4px;
}

.sb-nav-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 10px;
    border-radius: 8px;
    color: rgba(255, 255, 255, 0.88);
    font-size: 0.82rem;
    font-weight: 500;
    margin-bottom: 4px;
}

/* Selector de modelo limpio en sidebar */
[data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"] {
    gap: 6px !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 10px !important;
    padding: 8px 12px !important;
    width: 100% !important;
    margin: 0 !important;
    transition: all 0.15s ease !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label p {
    color: rgba(255, 255, 255, 0.82) !important;
    font-size: 0.76rem !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
    background: rgba(255, 255, 255, 0.13) !important;
    border-color: rgba(255, 255, 255, 0.35) !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) p {
    color: #FFFFFF !important;
    font-weight: 700 !important;
}

/* Expanders discretos en el sidebar */
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: transparent !important;
    border: none !important;
    border-radius: 8px !important;
    margin-bottom: 2px !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    color: rgba(255, 255, 255, 0.85) !important;
    padding: 7px 8px !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {
    background: rgba(255, 255, 255, 0.06) !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary p {
    color: rgba(255, 255, 255, 0.88) !important;
    font-size: 0.81rem !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary svg {
    fill: rgba(255, 255, 255, 0.7) !important;
    color: rgba(255, 255, 255, 0.7) !important;
}
[data-testid="stSidebar"] [data-testid="stExpanderDetails"] {
    background: rgba(0, 0, 0, 0.2) !important;
    border-radius: 8px !important;
    padding: 10px !important;
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] [data-testid="stExpanderDetails"] small,
[data-testid="stSidebar"] [data-testid="stExpanderDetails"] p {
    color: rgba(255, 255, 255, 0.85) !important;
}

/* Tarjeta inferior del sidebar idéntica a la imagen */
.sb-bottom-card {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 12px;
    padding: 14px;
    margin-top: 28px;
}
.sb-bottom-card-title {
    display: flex;
    align-items: center;
    gap: 7px;
    font-size: 0.82rem;
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 5px;
}
.sb-bottom-card-desc {
    font-size: 0.73rem;
    color: rgba(255, 255, 255, 0.68);
    line-height: 1.45;
}
.sb-bottom-footer {
    border-top: 1px solid rgba(255, 255, 255, 0.1);
    margin-top: 16px;
    padding-top: 12px;
    font-size: 0.64rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: rgba(255, 255, 255, 0.45);
    text-transform: uppercase;
}

/* ═══════════════════════════════════════════════════════════════════════════
   TOPBAR MINIMALISTA (ESTILO IMAGEN 2)
   ═══════════════════════════════════════════════════════════════════════════ */
.clean-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 4px 16px 4px;
    border-bottom: 1px solid #E5E9F0;
    margin-bottom: 24px;
}
.topbar-kicker {
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    color: #64748B;
    text-transform: uppercase;
}
.topbar-title {
    font-size: 0.98rem;
    font-weight: 700;
    color: #0F172A;
    margin-top: 1px;
}
.topbar-model-pill {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 6px 14px 6px 8px;
    box-shadow: 0 2px 6px rgba(15, 23, 42, 0.03);
}
.topbar-model-icon {
    width: 30px;
    height: 30px;
    border-radius: 8px;
    background: #FFF1F2;
    display: flex;
    align-items: center;
    justify-content: center;
}
.topbar-model-label {
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: #64748B;
    text-transform: uppercase;
    line-height: 1.1;
}
.topbar-model-name {
    font-size: 0.8rem;
    font-weight: 700;
    color: #0F172A;
    line-height: 1.2;
}

/* ═══════════════════════════════════════════════════════════════════════════
   HERO CENTRAL Y TARJETAS DE PREGUNTAS FRECUENTES (2x2)
   ═══════════════════════════════════════════════════════════════════════════ */
.clean-hero {
    text-align: center;
    padding: 18px 16px 26px 16px;
    max-width: 720px;
    margin: 0 auto;
}
.clean-hero-badge {
    width: 48px;
    height: 48px;
    border-radius: 14px;
    background: linear-gradient(135deg, #C8102E 0%, #7C0D1E 100%);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 8px 20px rgba(200, 16, 46, 0.25);
    margin-bottom: 14px;
}
.clean-hero-kicker {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 7px;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    color: #64748B;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.clean-hero-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #C8102E;
    display: inline-block;
}
.clean-hero-h1 {
    font-size: 2.05rem;
    font-weight: 800;
    color: #0F172A;
    letter-spacing: -0.025em;
    line-height: 1.2;
    margin-bottom: 8px;
}
.clean-hero-sub {
    font-size: 0.92rem;
    color: #64748B;
    font-weight: 400;
    line-height: 1.5;
}

.faq-header-row {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    margin: 6px 0 12px 0;
}
.faq-header-title {
    font-size: 0.92rem;
    font-weight: 700;
    color: #0F172A;
}
.faq-header-sub {
    font-size: 0.78rem;
    color: #64748B;
    margin-top: 2px;
}
.faq-header-link {
    font-size: 0.78rem;
    font-weight: 700;
    color: #C8102E;
}

/* Tarjetas horizontales 2x2 de Preguntas Frecuentes (Botones nativos estilizados) */
[class*="st-key-q_"] button {
    background: #FFFFFF !important;
    border: 1px solid #E5E9F0 !important;
    border-radius: 16px !important;
    padding: 16px 42px 16px 68px !important;
    min-height: 78px !important;
    width: 100% !important;
    text-align: left !important;
    justify-content: flex-start !important;
    position: relative !important;
    box-shadow: 0 2px 6px rgba(15, 23, 42, 0.02) !important;
    transition: all 0.2s ease !important;
}
[class*="st-key-q_"] button:hover {
    border-color: #FECDD3 !important;
    box-shadow: 0 8px 22px rgba(200, 16, 46, 0.07) !important;
    transform: translateY(-2px) !important;
}
[class*="st-key-q_"] button p {
    font-size: 0.88rem !important;
    font-weight: 700 !important;
    color: #0F172A !important;
    line-height: 1.35 !important;
    margin: 0 !important;
}
/* Icono izquierdo en caja rosada suave dentro de cada tarjeta FAQ */
[class*="st-key-q_"] button::before {
    content: "";
    position: absolute;
    left: 16px;
    top: 50%;
    transform: translateY(-50%);
    width: 40px;
    height: 40px;
    border-radius: 10px;
    background-color: #FFF1F2;
    background-repeat: no-repeat;
    background-position: center;
    background-size: 18px 18px;
}
/* Flecha diagonal derecha ↗ */
[class*="st-key-q_"] button::after {
    content: "";
    position: absolute;
    right: 16px;
    top: 50%;
    transform: translateY(-50%);
    width: 15px;
    height: 15px;
    opacity: 0.55;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='15' height='15' viewBox='0 0 24 24' fill='none' stroke='%23475569' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M7 17 17 7'/%3E%3Cpath d='M7 7h10v10'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-size: contain;
    transition: all 0.18s ease;
}
[class*="st-key-q_"] button:hover::after {
    opacity: 1;
    transform: translate(2px, -50%);
}

/* Iconos y categorías específicas para cada una de las 4 tarjetas */
.st-key-q_recons button::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%23C8102E' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z'/%3E%3Cpath d='m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z'/%3E%3Cpath d='M7 21h10'/%3E%3Cpath d='M12 3v18'/%3E%3Cpath d='M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2'/%3E%3C/svg%3E");
}
.st-key-q_recons button p::before {
    content: "EVALUACIONES Y NOTAS\\A";
    white-space: pre;
    display: block;
    font-size: 0.64rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: #64748B;
    margin-bottom: 2px;
}

.st-key-q_bach button::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%23C8102E' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M22 10v6M2 10l10-5 10 5-10 5z'/%3E%3Cpath d='M6 12v5c3 3 9 3 12 0v-5'/%3E%3C/svg%3E");
}
.st-key-q_bach button p::before {
    content: "GRADOS Y TÍTULOS\\A";
    white-space: pre;
    display: block;
    font-size: 0.64rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: #64748B;
    margin-bottom: 2px;
}

.st-key-q_ia button::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%23C8102E' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='4' y='4' width='16' height='16' rx='2'/%3E%3Crect x='9' y='9' width='6' height='6'/%3E%3Cpath d='M9 1v3'/%3E%3Cpath d='M15 1v3'/%3E%3Cpath d='M9 20v3'/%3E%3Cpath d='M15 20v3'/%3E%3Cpath d='M20 9h3'/%3E%3Cpath d='M20 14h3'/%3E%3Cpath d='M1 9h3'/%3E%3Cpath d='M1 14h3'/%3E%3C/svg%3E");
}
.st-key-q_ia button p::before {
    content: "DIRECTIVA ACADÉMICA\\A";
    white-space: pre;
    display: block;
    font-size: 0.64rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: #64748B;
    margin-bottom: 2px;
}

.st-key-q_ret button::before {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='%23C8102E' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z'/%3E%3Cpolyline points='14 2 14 8 20 8'/%3E%3Cline x1='16' y1='13' x2='8' y2='13'/%3E%3Cline x1='16' y1='17' x2='8' y2='17'/%3E%3C/svg%3E");
}
.st-key-q_ret button p::before {
    content: "MATRÍCULA Y RETIROS\\A";
    white-space: pre;
    display: block;
    font-size: 0.64rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: #64748B;
    margin-bottom: 2px;
}

/* ═══════════════════════════════════════════════════════════════════════════
   COMPOSER INFERIOR FLOTANTE Y LIMPIO (ESTILO CHAT / ASISTENTE ESAN)
   ═══════════════════════════════════════════════════════════════════════════ */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF !important;
    border: 1px solid #DCE2EC !important;
    border-radius: 20px !important;
    padding: 16px 20px !important;
    box-shadow: 0 8px 28px rgba(15, 23, 42, 0.04) !important;
}

/* Píldoras de selección de programa minimalistas dentro del composer */
[data-testid="stMain"] [data-testid="stRadio"] > div[role="radiogroup"] {
    background: #F8FAFC !important;
    border: 1px solid #E5E9F0 !important;
    border-radius: 12px !important;
    padding: 3.5px !important;
    display: inline-flex !important;
    flex-wrap: wrap !important;
    gap: 3px !important;
}
[data-testid="stMain"] [data-testid="stRadio"] div[role="radiogroup"] > label {
    background: transparent !important;
    border-radius: 9px !important;
    padding: 4px 12px !important;
    margin: 0 !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
}
[data-testid="stMain"] [data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child {
    display: none !important;
}
[data-testid="stMain"] [data-testid="stRadio"] div[role="radiogroup"] > label p {
    font-size: 0.74rem !important;
    font-weight: 700 !important;
    color: #475569 !important;
}
[data-testid="stMain"] [data-testid="stRadio"] div[role="radiogroup"] > label:hover {
    background: #FFF1F2 !important;
    color: #C8102E !important;
}
[data-testid="stMain"] [data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
    background: #C8102E !important;
    box-shadow: 0 2px 6px rgba(200, 16, 46, 0.25) !important;
}
[data-testid="stMain"] [data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) p {
    color: #FFFFFF !important;
    font-weight: 800 !important;
}

/* Textarea limpio e integrado sin bordes pesados */
[data-testid="stTextArea"] textarea {
    background-color: #FFFFFF !important;
    border: none !important;
    border-bottom: 1px solid #F1F5F9 !important;
    border-radius: 0 !important;
    font-size: 0.93rem !important;
    color: #0F172A !important;
    padding: 8px 2px !important;
    box-shadow: none !important;
}
[data-testid="stTextArea"] textarea:focus {
    border-bottom: 1px solid #FECDD3 !important;
    box-shadow: none !important;
}

/* Botón de envío circular / píldora en Rojo ESAN */
button[kind="primary"] {
    background: #C8102E !important;
    border-color: #C8102E !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    border-radius: 999px !important;
    min-height: 38px !important;
    padding: 0 18px !important;
    box-shadow: 0 4px 12px rgba(200, 16, 46, 0.24) !important;
    transition: all 0.18s ease !important;
}
button[kind="primary"]:hover:not(:disabled) {
    background: #A30B24 !important;
    border-color: #A30B24 !important;
    transform: translateY(-1px) !important;
}
button[kind="primary"]:disabled {
    background: #CBD5E1 !important;
    border-color: #CBD5E1 !important;
    color: #FFFFFF !important;
    box-shadow: none !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   TARJETAS DE RESULTADOS (DICTAMEN Y CITAS)
   ═══════════════════════════════════════════════════════════════════════════ */
.esan-card-title {
    font-size: 0.94rem;
    font-weight: 800;
    color: #0F172A;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 1px solid #F1F5F9;
}
.draft-box {
    background: #FAFAFA;
    border: 1px solid #E2E8F0;
    border-left: 3.5px solid #C8102E;
    border-radius: 10px;
    padding: 16px;
    font-size: 0.9rem;
    line-height: 1.65;
    color: #1E293B;
    white-space: pre-wrap;
    margin-bottom: 12px;
}
.cite-card {
    background: #FFFFFF;
    border: 1px solid #E5E9F0;
    border-left: 3.5px solid #C8102E;
    border-radius: 12px;
    padding: 14px;
    margin-bottom: 12px;
    transition: all 0.2s ease;
}
.cite-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    background: #C8102E;
    color: #FFFFFF;
    font-size: 0.72rem;
    font-weight: 800;
    margin-right: 8px;
    flex-shrink: 0;
}
.cite-source {
    font-weight: 700;
    font-size: 0.86rem;
    color: #0F172A;
}
.cite-excerpt {
    background: #FFF1F2;
    border-radius: 8px;
    padding: 8px 12px;
    margin-top: 8px;
    font-size: 0.81rem;
    font-style: italic;
    color: #475569;
    border-left: 2.5px solid #C8102E;
    line-height: 1.45;
}
.drive-link {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    color: #C8102E;
    text-decoration: none;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 4px 10px;
    background: #FFF1F2;
    border-radius: 8px;
    border: 1px solid #FECDD3;
}
.drive-link:hover {
    background: #FFE4E6;
    text-decoration: none;
}
.reason-box {
    background: #F8FAFC;
    border-left: 3px solid #C8102E;
    padding: 10px 14px;
    border-radius: 6px;
    font-size: 0.83rem;
    color: #334155;
    line-height: 1.55;
}
</style>
""", unsafe_allow_html=True)

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

# ── SIDEBAR ULTRA LIMPIO (FIEL A LA SEGUNDA IMAGEN EN COLORES ESAN) ───────────
with st.sidebar:
    st.markdown(
        f"""
        <div class="esan-sb-brand">
          <div class="esan-sb-badge">{SVG["book_brand"]}</div>
          <div>
            <div class="esan-sb-title">ESAN</div>
            <div class="esan-sb-subtitle">UNIVERSIDAD</div>
            <div class="esan-sb-bars">
              <span class="esan-sb-bar-red"></span>
              <span class="esan-sb-bar-white"></span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Nueva conversación", key="btn_sb_new", use_container_width=True):
        st.session_state.history = []
        st.session_state.last_result = None
        st.session_state.last_sources = []
        st.session_state.last_program = None
        st.session_state.last_query = None
        st.session_state.last_extra_ctx = None
        st.session_state.quick_q = None
        st.session_state.form_id += 1
        st.rerun()

    st.markdown('<div class="sb-group-title">ESPACIO ESAN</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="sb-nav-item">{SVG["book_sb"]} <span>Preguntas frecuentes</span></div>',
        unsafe_allow_html=True,
    )

    with st.expander("Modelos disponibles", expanded=False):
        model_label = st.radio(
            "Seleccionar Modelo",
            options=list(MODELS.keys()),
            index=0,
            label_visibility="collapsed",
        )

    rag = RAGEngine(api_key=api_key, openrouter_key=or_key) if api_key else None
    is_indexed = rag.is_indexed() if rag else False
    doc_cnt = rag.get_document_count() if is_indexed else 0

    with st.expander(f"Repositorio normativo ({doc_cnt:,})", expanded=False):
        if not is_indexed and api_key and os.path.exists(DOCS_FOLDER):
            if st.button("Indexar reglamentos", type="primary", use_container_width=True, key="btn_init_index"):
                with st.spinner("Indexando reglamentos..."):
                    index_missing_documents(api_key=api_key)
                    st.rerun()

        drive_url_input = st.text_input(
            "Carpeta Drive",
            value=GOOGLE_DRIVE_FOLDER,
            label_visibility="collapsed",
            placeholder="https://drive.google.com/drive/folders/...",
        )
        c_s1, c_s2 = st.columns(2)
        with c_s1:
            if st.button("Ver Drive", use_container_width=True, key="btn_sb_drive_preview"):
                try:
                    dfiles = list_drive_files(drive_url_input)
                    st.info(f"{len(dfiles)} archivos")
                except Exception as e:
                    st.error(str(e))
        with c_s2:
            if st.button("Sincronizar", use_container_width=True, disabled=not bool(api_key), key="btn_sb_drive_sync"):
                try:
                    s_res = sync_from_drive(drive_url_input, docs_dir=DOCS_FOLDER, progress_callback=lambda m: None)
                    for nf in s_res.get("new_files", []):
                        p = os.path.join(DOCS_FOLDER, nf)
                        if os.path.exists(p):
                            index_single_pdf(api_key, p)
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

        if os.path.exists(DOCS_FOLDER):
            all_docs = get_all_supported_files(DOCS_FOLDER)
            st.caption(f"{len(all_docs)} documentos locales en DOCS/")
            if st.button("Reindexar faltantes", use_container_width=True, disabled=not bool(api_key)):
                with st.spinner("Indexando..."):
                    index_missing_documents(api_key=api_key)
                    st.rerun()

    st.markdown(
        f"""
        <div class="sb-bottom-card">
          <div class="sb-bottom-card-title">
            {SVG["sparkles_rose"]} <span>Asistente académico</span>
          </div>
          <div class="sb-bottom-card-desc">
            Un espacio para consultar reglamentos, validar criterios y redactar dictámenes oficiales.
          </div>
        </div>
        <div class="sb-bottom-footer">UNIVERSIDAD ESAN</div>
        """,
        unsafe_allow_html=True,
    )


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
                <div style="background:#FFF1F2;border:1px solid #FECDD3;border-radius:8px;padding:12px;margin:10px 0">
                  <div style="font-size:0.78rem;font-weight:700;color:#9F1239;margin-bottom:4px">FRAGMENTO TEXTUAL CITADO:</div>
                  <div style="font-size:0.88rem;color:#1E293B;line-height:1.4">"{escape(extracto)}"</div>
                </div>
                """, unsafe_allow_html=True)
            with open(local_path, "rb") as f:
                st.download_button(
                    label=f"Descargar archivo local ({os.path.basename(local_path)})",
                    data=f.read(),
                    file_name=os.path.basename(local_path),
                    use_container_width=True,
                )
    else:
        st.warning(f"No se encontró copia local de '{doc_name}' en DOCS/. Puedes acceder al archivo en Google Drive:")
        if drive_url:
            st.markdown(f'<a href="{drive_url}" target="_blank" class="drive-link">{SVG["drive"]} Abrir documento oficial en Drive</a>', unsafe_allow_html=True)


# ── TOPBAR ULTRA LIMPIO (ESTILO IMAGEN 2) ─────────────────────────────────────
st.markdown('<div id="top-anchor"></div>', unsafe_allow_html=True)

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

active_model_short = "ESAN Académico · GPT-4o" if "gpt-4o" in MODELS[model_label]["model_id"] else "ESAN Académico · MiMo Pro"

st.markdown(
    f"""
    <div class="clean-topbar">
      <div>
        <div class="topbar-kicker">UNIVERSIDAD ESAN</div>
        <div class="topbar-title">Asistente académico normativo</div>
      </div>
      <div class="topbar-model-pill">
        <div class="topbar-model-icon">{SVG["sparkles_crimson"]}</div>
        <div>
          <div class="topbar-model-label">MODELO ACTIVO</div>
          <div class="topbar-model-name">{active_model_short}</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── CONTENIDO PRINCIPAL: PANTALLA DE INICIO LIMPIA O RESULTADOS ───────────────
result = st.session_state.last_result
sources = st.session_state.last_sources

if result is None:
    _, center_col, _ = st.columns([0.12, 0.76, 0.12])
    with center_col:
        st.markdown(
            f"""
            <div class="clean-hero">
              <div class="clean-hero-badge">{SVG["sparkles_white"]}</div>
              <div class="clean-hero-kicker">
                <span class="clean-hero-dot"></span>
                <span>CONOCIMIENTO QUE TRANSFORMA</span>
              </div>
              <div class="clean-hero-h1">Hola, ¿qué necesitas resolver hoy?</div>
              <div class="clean-hero-sub">
                Consulta reglamentos oficiales, estructura dictámenes de procedencia y redacta comunicaciones académicas.
              </div>
            </div>
            <div class="faq-header-row">
              <div>
                <div class="faq-header-title">Preguntas frecuentes</div>
                <div class="faq-header-sub">Elige una para iniciar la conversación.</div>
              </div>
              <div class="faq-header-link">Casos frecuentes ▾</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        f_col1, f_col2 = st.columns(2, gap="medium")
        with f_col1:
            if st.button("¿Cómo solicitar reconsideración de una nota?", key="q_recons", use_container_width=True):
                st.session_state.quick_q = "¿Cuándo puedo pedir reconsideración de nota y ante qué autoridad corresponde presentarla?"
                st.session_state.form_id += 1
                st.rerun()
            if st.button("¿Qué lineamientos regulan el uso de IA en clases?", key="q_ia", use_container_width=True):
                st.session_state.quick_q = "¿Cuáles son los lineamientos sobre el uso de Inteligencia Artificial para docentes y alumnos en ESAN?"
                st.session_state.form_id += 1
                st.rerun()

        with f_col2:
            if st.button("¿Cuáles son los requisitos para el grado de Bachiller?", key="q_bach", use_container_width=True):
                st.session_state.quick_q = "¿Cuáles son los requisitos y créditos necesarios para obtener el Grado de Bachiller?"
                st.session_state.form_id += 1
                st.rerun()
            if st.button("¿Cuál es el trámite de retiro o devolución?", key="q_ret", use_container_width=True):
                st.session_state.quick_q = "¿Cuál es el procedimiento formal y plazos para solicitar devolución de dinero o retiro de programa?"
                st.session_state.form_id += 1
                st.rerun()

        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

else:
    # ── VISTA DE RESULTADOS (LIMPIA Y AIREADA) ────────────────────────────────
    last_q = st.session_state.get("last_query", "")
    last_prog = st.session_state.get("last_program", "")
    last_extra = st.session_state.get("last_extra_ctx", "")

    if last_q:
        badge_prog_html = (
            f'<span style="background:#FFF1F2;border:1px solid #FECDD3;color:#9F1239;'
            f'padding:3px 12px;border-radius:999px;font-size:0.75rem;font-weight:800;'
            f'display:inline-flex;align-items:center;gap:6px">'
            f'{SVG["grad_cap"]} Programa: {escape(last_prog)}</span>'
        ) if last_prog else ""

        extra_html = (
            f'<div style="font-size:0.78rem;color:#64748B;margin-top:6px">'
            f'<b>Contexto adicional:</b> {escape(last_extra)}</div>'
        ) if last_extra else ""

        st.markdown(
            f"""
            <div style="background:#FFFFFF;border:1px solid #E5E9F0;border-left:4px solid #C8102E;border-radius:14px;padding:14px 20px;margin-bottom:20px;box-shadow:0 2px 8px rgba(15,23,42,0.02)">
              <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;margin-bottom:4px">
                <span style="font-size:0.68rem;font-weight:800;color:#C8102E;text-transform:uppercase;letter-spacing:0.08em">
                  CONSULTA EVALUADA
                </span>
                {badge_prog_html}
              </div>
              <div style="font-size:0.98rem;color:#0F172A;font-weight:700;line-height:1.4">
                "{escape(last_q)}"
              </div>
              {extra_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    col_resp, col_norm = st.columns([1.3, 1.05], gap="large")

    with col_resp:
        with st.container(border=True):
            st.markdown(
                f'<div class="esan-card-title">{SVG["scale"]} Dictamen Técnico y Orientación para Coordinación</div>',
                unsafe_allow_html=True,
            )

            dictamen = result.get("dictamen_coordinacion", {})
            procedencia = dictamen.get("procedencia", "Evaluación de Coordinación Académica")
            resumen_ejecutivo = dictamen.get("resumen_ejecutivo", "") or result.get("razonamiento", "")
            appr = result.get("requiere_aprobacion", "La coordinadora puede resolver directamente")

            proc_lower = procedencia.lower()
            if "no procede" in proc_lower:
                badge_bg, badge_border, badge_text = "#FFF1F2", "#FECDD3", "#9F1239"
                icon_badge = SVG["alert"]
            elif any(k in proc_lower for k in ("no contemplado", "elevación", "insuficiente", "revisión", "validación")):
                badge_bg, badge_border, badge_text = "#FFFBEB", "#FDE68A", "#92400E"
                icon_badge = SVG["alert"]
            else:
                badge_bg, badge_border, badge_text = "#F0FDF4", "#BBF7D0", "#166534"
                icon_badge = SVG["check"]

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
                            f'<a href="#cita-{n}" style="color:#C8102E;font-weight:800;text-decoration:none">[{n}]</a>'
                            for n in numeros
                        )
                        partes_criterio.append(f"{escape(texto)} {refs}")

            resumen_fmt = escape(resumen_ejecutivo).replace("\n\n", "</p><p style='margin:6px 0'>").replace("\n", "<br>")

            fundamentos_html = ""
            if partes_criterio:
                items_li = "".join(f"<li style='margin-bottom:6px'>{p}</li>" for p in partes_criterio)
                fundamentos_html = f"""
                <div style="margin-top:12px;padding:10px 14px;background:#FFFFFF;border-radius:10px;border:1px solid rgba(200,16,46,0.14)">
                  <div style="font-size:0.72rem;font-weight:800;color:#9F1239;text-transform:uppercase;letter-spacing:0.04em;margin-bottom:6px;display:flex;align-items:center;gap:5px">
                    {SVG['bookmark']} Artículos y Fundamentos Verificados:
                  </div>
                  <ul style="margin:0;padding-left:18px;font-size:0.83rem;color:#1E293B;line-height:1.5">
                    {items_li}
                  </ul>
                </div>
                """

            st.markdown(
                f"""
                <div style="background:{badge_bg};border:1px solid {badge_border};border-radius:12px;padding:16px 18px;margin-bottom:14px">
                  <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
                    <span style="font-weight:800;font-size:0.86rem;color:{badge_text};text-transform:uppercase;letter-spacing:0.02em;display:flex;align-items:center;gap:6px">
                      {icon_badge} {escape(procedencia)}
                    </span>
                  </div>
                  <div style="font-size:0.87rem;color:#0F172A;line-height:1.6;margin-bottom:6px">
                    <p style="margin:4px 0">{resumen_fmt}</p>
                  </div>
                  {fundamentos_html}
                  <div style="font-size:0.78rem;color:#475569;display:flex;align-items:center;gap:6px;border-top:1px solid rgba(0,0,0,0.06);padding-top:8px;margin-top:10px">
                    {SVG['shield_check']} <span><b>Instancia competente:</b> {escape(appr)}</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            actions = result.get("acciones_coordinadora", [])
            if actions:
                st.markdown(
                    f"<div style='margin:12px 0 6px 0;display:flex;align-items:center;gap:6px;'><b style='font-size:0.84rem;color:#0F172A'>{SVG['clipboard_check']} Protocolo de verificación en sistema</b></div>",
                    unsafe_allow_html=True,
                )
                for act in actions:
                    st.checkbox(act, value=False, key=f"chk_{hash(act)}")

            reason = result.get("razonamiento", "")
            if reason:
                with st.expander("Ver fundamentación técnica detallada"):
                    st.markdown(f'<div class="reason-box">{escape(reason)}</div>', unsafe_allow_html=True)

            st.markdown("<hr style='margin:16px 0;border-color:#F1F5F9'>", unsafe_allow_html=True)

            st.markdown(
                "<div style='margin-bottom:8px'><b style='font-size:0.86rem;color:#0F172A'>Borrador de comunicación para el estudiante</b></div>",
                unsafe_allow_html=True,
            )

            draft = result.get("borrador_alumno", "")
            if isinstance(draft, str) and (draft.strip().startswith("{") or '"dictamen_coordinacion"' in draft):
                import re as _re_ui
                m_inner = _re_ui.search(r'"borrador_alumno"\s*:\s*"(.*?)(?:"\s*,\s*"[a-zA-Z_]+"|"\s*\}\s*$)', draft, _re_ui.DOTALL)
                if not m_inner:
                    m_inner = _re_ui.search(r'(Estimad[oa][\s\S]*?)(?:"\s*,\s*"[a-zA-Z_]+"|"\s*\}\s*$|\Z)', draft)
                if m_inner:
                    draft = m_inner.group(1).replace("\\n", "\n").replace('\\"', '"').strip()
                else:
                    draft = result.get("dictamen_coordinacion", {}).get("resumen_ejecutivo", "") or result.get("razonamiento", "")

            st.markdown(f'<div class="draft-box">{escape(draft)}</div>', unsafe_allow_html=True)

            b_c1, b_c2 = st.columns(2)
            copia_bloqueada = bool(result.get("validacion_pendiente"))
            if copia_bloqueada:
                st.warning("El borrador requiere validación previa antes de compartirse.")
            wa_text = f"*Coordinación Académica ESAN*\n\n{draft}"
            mail_text = draft

            with b_c1:
                if st.button("Copiar para WhatsApp", key="btn_cp_wa", use_container_width=True, disabled=copia_bloqueada):
                    st.code(wa_text, language=None)
            with b_c2:
                if st.button("Copiar para Correo", key="btn_cp_mail", use_container_width=True, disabled=copia_bloqueada):
                    st.code(mail_text, language=None)

    with col_norm:
        with st.container(border=True):
            citas = result.get("citas_normativas", [])
            documentos = len({c.get("documento", "") for c in citas})
            st.markdown(
                f'<div class="esan-card-title">{SVG["book"]} Citas verificadas ({len(citas)}) · Documentos ({documentos})</div>',
                unsafe_allow_html=True,
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

            import unicodedata as _ud

            def _norm_docname(name):
                n = _ud.normalize("NFD", name.lower())
                n = "".join(c for c in n if _ud.category(c) != "Mn")
                return n.strip()

            grouped = {}
            order = []
            for num, c in sorted(raw_items.items()):
                doc_name_raw = c.get("documento", "")
                nkey = _norm_docname(doc_name_raw)
                if nkey not in grouped:
                    fb_src = ""
                    if isinstance(num, int) and 1 <= num <= len(sources):
                        fb_src = sources[num - 1].get("source", "")
                    grouped[nkey] = {
                        "doc_name": doc_name_raw or fb_src,
                        "fallback_source": fb_src,
                        "citas": [],
                    }
                    order.append(nkey)

                grp = grouped[nkey]
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

                st.markdown(
                    f"""
                    <div class="cite-card" id="cita-{card_idx}">
                      <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:8px;margin-bottom:6px">
                        <div style="display:flex;align-items:flex-start;gap:8px">
                          <span class="cite-num">{card_idx}</span>
                          <div>
                            <div class="cite-source">{safe_doc_name}</div>
                            <div style="font-size:0.72rem;color:#64748B;margin-top:2px">{len(all_citas)} fragmento(s) verificado(s)</div>
                          </div>
                        </div>
                        <a href="{safe_drive_url}" target="_blank" rel="noopener noreferrer" class="drive-link" style="white-space:nowrap">
                          {SVG["drive"]} Drive
                        </a>
                      </div>
                    """,
                    unsafe_allow_html=True,
                )

                for sub_i, cita in enumerate(all_citas, 1):
                    c_sec = cita["seccion"]
                    c_pg = cita["pagina"]
                    c_ext = cita["extracto"]

                    badge_pg = f'<span style="background:#FFF1F2;color:#9F1239;padding:2px 8px;border-radius:5px;font-size:0.7rem;font-weight:700;border:1px solid #FECDD3">Pág. {escape(str(c_pg))}</span>' if c_pg else ""
                    sec_text = f'<span style="font-size:0.79rem;font-weight:700;color:#1E293B">{escape(str(c_sec))}</span>' if c_sec else ""

                    st.markdown(
                        f"""
                        <div style="margin-top:8px;padding-top:8px;border-top:1px dashed #E2E8F0">
                          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px">
                            <div>{sec_text}</div>
                            <div>{badge_pg}</div>
                          </div>
                          {f'<div class="cite-excerpt" style="margin-bottom:8px">"{escape(str(c_ext))}..."</div>' if c_ext else ''}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    btn_label = f"Ver y resaltar en PDF (Pág. {c_pg})" if c_pg else "Ver y resaltar en PDF"
                    if st.button(
                        btn_label,
                        key=f"btn_pdf_modal_{card_idx}_{sub_i}_{cita['num']}",
                        use_container_width=True,
                        type="primary",
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

# ═════════════════════════════════════════════════════════════════════════════
# COMPOSER INFERIOR CENTRALIZADO Y MINIMALISTA (FIEL A IMAGEN 2)
# ═════════════════════════════════════════════════════════════════════════════
_, comp_col, _ = st.columns([0.12, 0.76, 0.12])
with comp_col:
    current_form_id = st.session_state.form_id

    with st.container(border=True):
        # Fila superior compacta: Selector de programa por píldoras
        p_col_lbl, p_col_pills = st.columns([0.24, 0.76])
        with p_col_lbl:
            st.markdown(
                '<div style="font-size:0.72rem;font-weight:800;color:#64748B;text-transform:uppercase;letter-spacing:0.06em;padding-top:8px">'
                'Programa <span style="color:#C8102E">*</span></div>',
                unsafe_allow_html=True,
            )
        with p_col_pills:
            programa_options = ["PEE", "PADE", "PAE", "Diploma", "CIE", "Maestría", "Doctorado"]
            default_prog_idx = 0 if st.session_state.quick_q else None
            programa_sel = st.radio(
                "Programa",
                options=programa_options,
                index=default_prog_idx,
                horizontal=True,
                label_visibility="collapsed",
                key=f"prog_choice_{current_form_id}",
            )

        # Campo de texto limpio e integrado
        default_prompt = st.session_state.quick_q or ""
        student_msg = st.text_area(
            "Escribe tu pregunta",
            value=default_prompt,
            height=76,
            placeholder="Escribe tu pregunta normativa o caso del estudiante...",
            label_visibility="collapsed",
            max_chars=4000,
            key=f"query_text_{current_form_id}",
        )

        if st.session_state.quick_q:
            st.session_state.quick_q = None

        # Barra inferior del composer: marca izquierda + contexto opcional + botón enviar
        b_left, b_mid, b_right = st.columns([0.48, 0.27, 0.25])
        with b_left:
            prog_hint = f" · Programa: <b>{programa_sel}</b>" if programa_sel else " · <span style='color:#C8102E'>Selecciona un programa</span>"
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:6px;font-size:0.76rem;color:#64748B;padding-top:8px">'
                f'{SVG["sparkles_crimson"]} <span>Asistente académico ESAN{prog_hint}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with b_mid:
            extra_ctx = st.text_input(
                "Contexto",
                placeholder="+ Contexto opcional...",
                label_visibility="collapsed",
                key=f"extra_ctx_{current_form_id}",
            )
        with b_right:
            can_run = bool(api_key and student_msg.strip() and programa_sel)
            run_query = st.button(
                "Consultar  ↑",
                use_container_width=True,
                type="primary",
                disabled=not can_run,
                help="Selecciona el programa y escribe la consulta para enviar.",
            )

    st.markdown(
        '<div style="text-align:center;font-size:0.71rem;color:#94A3B8;margin-top:8px">'
        'Asistente normativo institucional · Todas las respuestas se fundamentan en reglamentos indexados de ESAN.'
        '</div>',
        unsafe_allow_html=True,
    )

# ── PROCESAMIENTO DE CONSULTA ─────────────────────────────────────────────────
if run_query and api_key and student_msg.strip():
    loading_slot = st.empty()
    loading_slot.markdown(
        f"""
        <div style="max-width:760px;margin:12px auto;background:#460711;border:1px solid #7C0D1E;color:#FFFFFF;border-radius:14px;padding:14px 20px;display:flex;align-items:center;gap:14px;box-shadow:0 6px 20px rgba(200,16,46,0.2)">
          <div style="width:22px;height:22px;border:2.5px solid #FF4D5E;border-top-color:transparent;border-radius:50%;animation:spin 0.8s linear infinite;flex-shrink:0"></div>
          <div style="font-size:0.84rem">
            <b>Analizando normativa para {escape(programa_sel or "")}...</b>
            <span style="color:#FECDD3;margin-left:6px">Verificando artículos y generando dictamen institucional.</span>
          </div>
        </div>
        <style>@keyframes spin {{ to {{ transform: rotate(360deg); }} }}</style>
        """,
        unsafe_allow_html=True,
    )
    try:
        if not is_indexed and api_key:
            build_vector_store(api_key=api_key)
            rag = RAGEngine(api_key=api_key, openrouter_key=or_key)

        result_data, sources_data = rag.answer_question(
            query=student_msg.strip(),
            context_extra=extra_ctx.strip() if "extra_ctx" in locals() and extra_ctx else "",
            model_label=model_label,
            chat_history=None,
            programa=programa_sel or "",
        )
        st.session_state.last_result = result_data
        st.session_state.last_sources = sources_data
        st.session_state.last_program = programa_sel
        st.session_state.last_query = student_msg.strip()
        st.session_state.last_extra_ctx = extra_ctx.strip() if "extra_ctx" in locals() and extra_ctx else ""
        st.session_state.history.append(student_msg.strip()[:120])

        st.session_state.form_id += 1
        st.session_state.scroll_to_top = True
        loading_slot.empty()
        st.rerun()
    except Exception as e:
        loading_slot.empty()
        st.error(f"Error al procesar la consulta: {e}")
