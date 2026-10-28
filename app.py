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


# ── IDENTIDAD DEL ASISTENTE: ACI ESAN (Asistente de Conocimiento Institucional) ─
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def get_asset_base64(filename: str) -> str:
    """Devuelve un recurso gráfico de assets/ en base64 ('' si no existe)."""
    try:
        with open(os.path.join(ASSETS_DIR, filename), "rb") as fh:
            return base64.b64encode(fh.read()).decode("utf-8")
    except OSError:
        return ""


ACI_ICON_B64 = get_asset_base64("aci_icon.png")
ACI_HORIZONTAL_B64 = get_asset_base64("aci_logo_blanco.png")
_ACI_FAVICON = os.path.join(ASSETS_DIR, "aci_favicon.png")

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ACI ESAN · Sistema Normativo · Universidad ESAN",
    page_icon=_ACI_FAVICON if os.path.exists(_ACI_FAVICON) else "https://cdn.worldvectorlogo.com/logos/universidad-esan.svg",
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


# Iconos adicionales usados solo por la interfaz
def _ico(paths, size=16, color="#D31418", sw=2, extra=""):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
            f'fill="none" stroke="{color}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round" '
            f'style="flex-shrink:0;{extra}">{paths}</svg>')

_P_SCALE = '<path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="M7 21h10"/><path d="M12 3v18"/><path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"/>'
_P_BOOK = '<path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>'
_P_FILE = '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><path d="M9 13h6"/><path d="M9 17h6"/>'
_P_CAP = '<path d="M22 10v6M2 10l10-5 10 5-10 5z"/><path d="M6 12v5c3 3 9 3 12 0v-5"/>'
_P_CHIP = '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 14h3M1 9h3M1 14h3"/>'
_P_SEARCH = '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>'
_P_MSG = '<path d="M7.9 20A9 9 0 1 0 4 16.1L2 22Z"/>'
_P_SHIELD = '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10"/>'
_P_CHECK = '<polyline points="20 6 9 17 4 12"/>'
_P_ALERT = '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/>'

ICON = {
    "scale_w": _ico(_P_SCALE, 22, "#FFFFFF"),
    "scale": _ico(_P_SCALE, 18),
    "book": _ico(_P_BOOK, 18),
    "book_w": _ico(_P_BOOK, 16, "#FFFFFF"),
    "file": _ico(_P_FILE, 18),
    "file_sm": _ico(_P_FILE, 16),
    "cap": _ico(_P_CAP, 18),
    "chip": _ico(_P_CHIP, 18),
    "search": _ico(_P_SEARCH, 18),
    "search_w": _ico(_P_SEARCH, 16, "#FFFFFF"),
    "msg": _ico(_P_MSG, 16),
    "shield": _ico(_P_SHIELD, 14),
    "check_g": _ico(_P_CHECK, 14, "#15803D", 2.5),
    "check_r": _ico(_P_CHECK, 13, "#C8102E", 2.5),
    "alert": _ico(_P_ALERT, 16, "currentColor"),
    # Iconos de presentación añadidos en el rediseño
    "search_ph": _ico(_P_SEARCH, 18, "#FFFFFF"),
    "layers_ph": _ico('<path d="m12 2 10 5-10 5L2 7z"/><path d="m2 17 10 5 10-5"/><path d="m2 12 10 5 10-5"/>', 18, "#FFFFFF"),
    "scale_ph": _ico(_P_SCALE, 18, "#FFFFFF"),
    "mail": _ico('<rect width="20" height="16" x="2" y="4" rx="2"/><path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7"/>', 17),
    "book_ink": _ico(_P_BOOK, 17, "#1E1517"),
    "file_ctx": _ico(_P_FILE, 15, "#7B6F71"),
    "check_ok": _ico(_P_CHECK, 11, "#17663A", 3),
    "info": _ico('<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>', 15, "#9C0E14"),
}

# ── ESTILOS INSTITUCIONALES ESAN ──────────────────────────────────────────────
# Toda la presentación visual está centralizada en styles/esan.css.
# Si el archivo no existiera, la aplicación sigue funcionando sin estilos propios.
_CSS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "styles", "esan.css")


def _load_css(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return ""


st.markdown(f"<style>\n{_load_css(_CSS_PATH)}\n</style>", unsafe_allow_html=True)

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
        st.markdown(f'<div class="sb-logo"><img src="data:image/svg+xml;base64,{ESAN_LOGO_B64}" alt="Universidad ESAN" /></div>', unsafe_allow_html=True)
    else:
        st.markdown('<h3 style="color:#D31418;margin:0">UNIVERSIDAD ESAN</h3>', unsafe_allow_html=True)

    _aci_icon_html = (
        f'<img class="sb-aci-icon" src="data:image/png;base64,{ACI_ICON_B64}" alt="ACI ESAN" />'
        if ACI_ICON_B64 else ''
    )
    st.markdown(
        f'<div class="sb-aci">{_aci_icon_html}<div>'
        f'<div class="sb-aci-name">ACI <span>ESAN</span></div>'
        f'<div class="sb-aci-desc">Asistente de Conocimiento Institucional</div>'
        f'</div></div>'
        '<div class="sb-eyebrow">Coordinación Académica</div><div class="sb-brand">Gestión y consulta normativa</div>',
        unsafe_allow_html=True,
    )

    if api_key:
        st.markdown(f'<div class="status-pill-ok">Conexión reglamentaria activa</div>', unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="keys-warning">
          <div class="keys-warning-title">{SVG['alert']} Claves no configuradas</div>
          <div class="keys-warning-body">
            En Streamlit Cloud ve a <b>Settings &gt; Secrets</b> y agrega:
            <code>OPENAI_API_KEY = "tu_key"</code>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 1. SELECCIÓN DE MODELO
    st.markdown('<div class="sb-section">Motor de dictámenes</div>', unsafe_allow_html=True)
    model_label = st.radio(
        "Seleccionar Modelo",
        options=list(MODELS.keys()),
        index=0,
        label_visibility="collapsed",
        help="Seleccione el modelo según la complejidad del dictamen normativo."
    )
    model_cfg = MODELS[model_label]
    if model_cfg["provider"] == "openai":
        st.markdown(f'<div class="status-pill-model">OpenAI · GPT-4o-mini (Estándar)</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="status-pill-model">OpenRouter · MiMo v2.6 Pro (Razonamiento)</div>', unsafe_allow_html=True)

    st.markdown("---")

    # 2. BASE DOCUMENTAL
    st.markdown('<div class="sb-section">Repositorio normativo</div>', unsafe_allow_html=True)
    rag = RAGEngine(api_key=api_key, openrouter_key=or_key) if api_key else None
    is_indexed = rag.is_indexed() if rag else False

    if is_indexed:
        cnt = rag.get_document_count()
        st.markdown(f'<div class="sb-stat">{ICON["book_w"]}<div><div class="n">{cnt:,}</div><div class="t">secciones indexadas</div></div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="sb-stat warn">Repositorio pendiente de inicialización</div>', unsafe_allow_html=True)
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
                    st.markdown(f"<div class='sb-file'>{SVG['file']}{sub_rel}</div>", unsafe_allow_html=True)
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

    st.markdown('<div class="sb-footer"><b>ACI ESAN</b> · Conocimiento que impulsa decisiones<br>Herramienta de apoyo para revisión institucional.</div>', unsafe_allow_html=True)

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


# ── COMPONENTES DE PRESENTACIÓN (solo HTML; sin lógica de negocio) ────────────
def section_label(left: str, right: str = "") -> str:
    """Rótulo de sección sobre cada panel (01 · … / 02 · …), alineado en ambas columnas."""
    return f'<div class="section-label"><span class="l">{left}</span><span class="r">{right}</span></div>'


def panel_header(title: str, subtitle: str = "", icon: str = "", right: str = "", eyebrow: str = "") -> str:
    """Cabecera uniforme para los dos paneles de trabajo."""
    ico = f'<span class="ph-icon">{icon}</span>' if icon else ""
    eye = f'<div class="ph-step">{eyebrow}</div>' if eyebrow else ""
    sub = f'<div class="ph-sub">{subtitle}</div>' if subtitle else ""
    return (
        f'<div class="ph"><div class="ph-left">{ico}<div class="ph-text">'
        f'{eye}<div class="ph-title">{title}</div>{sub}'
        f'</div></div>{right}</div>'
    )


def block_header(icon: str, title: str, right: str = "", kicker: str = "") -> str:
    """Cabecera de tarjeta interna (dictamen, comunicación, evidencia)."""
    kick = f'<div class="block-kicker">{kicker}</div>' if kicker else ""
    return (
        f'<div class="block-head"><div>{kick}<div class="block-title">{icon}<span>{title}</span></div></div>'
        f'{right}</div>'
    )


def _topic_card(container_key: str, button_key: str, num: str, icon: str, title: str, desc: str, question: str):
    """Tarjeta de consulta frecuente. Conserva claves y comportamiento originales."""
    with st.container(key=container_key):
        st.markdown(
            f'<div class="topic-top"><span class="topic-icon">{icon}</span><span class="topic-num">{num}</span></div>'
            f'<div class="topic-title">{title}</div>'
            f'<div class="topic-desc">{desc}</div>',
            unsafe_allow_html=True,
        )
        if st.button("Iniciar consulta →", key=button_key, use_container_width=True):
            st.session_state.quick_q = question
            st.session_state.form_id += 1
            st.rerun()


# ── ENCABEZADO INSTITUCIONAL COMPACTO ─────────────────────────────────────────
_repo_chip = (
    '<span class="hchip"><span class="dot"></span>Repositorio conectado</span>'
    if is_indexed else
    '<span class="hchip"><span class="dot warn"></span>Repositorio pendiente</span>'
)
# Logotipo ACI ESAN en blanco sobre el rojo del encabezado
_aci_plate = (
    f'<div class="aci-plate"><img src="data:image/png;base64,{ACI_HORIZONTAL_B64}" alt="ACI ESAN" /></div>'
    if ACI_HORIZONTAL_B64 else f'<div class="nav-icon">{ICON["scale_w"]}</div>'
)

st.markdown(
    f'<div class="esan-navbar">'
    f'<div class="esan-nav-left">'
    f'{_aci_plate}'
    f'<div><div class="esan-nav-kicker">ACI ESAN · Asistente de Conocimiento Institucional</div>'
    f'<div class="esan-nav-title">Sistema Institucional de Consulta y Gestión Normativa</div>'
    f'<div class="esan-nav-sub">Coordinación Académica · Documentación institucional indexada</div></div>'
    f'</div>'
    f'<div class="esan-nav-right">{_repo_chip}'
    f'<span class="hchip solid">{ICON["shield"]}Revisión normativa asistida</span>'
    f'</div></div>',
    unsafe_allow_html=True,
)

# Distribución en dos paneles: consulta (≈36 %) y análisis (≈64 %)
col_form, col_main = st.columns([1, 1.55], gap="large")

# ── ÁREA PRINCIPAL: RESULTADOS O PANTALLA DE BIENVENIDA ───────────────────────
result = st.session_state.last_result
sources = st.session_state.last_sources

with col_main:
    st.markdown(
        section_label("02 · Consultas reglamentarias", "4 accesos rápidos")
        if result is None else
        section_label("02 · Análisis y evidencia", "Vista de trabajo"),
        unsafe_allow_html=True,
    )
    with st.container(key="panel_main"):
        if result is None:
            # ── Pantalla de consultas frecuentes ─────────────────────────────
            st.markdown(
                panel_header(
                    "¿Qué procedimiento necesitas revisar?",
                    "Elige un caso frecuente o redacta el tuyo en el panel de consulta.",
                    eyebrow="Consultas frecuentes",
                    right='<span class="chip red pill">Consultas académicas</span>',
                ),
                unsafe_allow_html=True,
            )
            # Espacio reservado para el indicador de carga dentro de este panel
            status_slot = st.empty()

            with st.container(key="welcome_body"):
                c_c1, c_c2 = st.columns(2, gap="small")
                with c_c1:
                    _topic_card(
                        "topic_1", "q_recons", "01", ICON["scale"],
                        "Reconsideración de evaluaciones",
                        "Plazos reglamentarios, requisitos formales e instancia resolutiva ante reclamos de calificación.",
                        "¿Cuándo puedo pedir reconsideración de nota y ante qué autoridad corresponde presentarla?",
                    )
                    _topic_card(
                        "topic_3", "q_ia", "03", ICON["chip"],
                        "Lineamientos de inteligencia artificial",
                        "Disposiciones institucionales para docentes y alumnos sobre el uso de herramientas IA en evaluaciones.",
                        "¿Cuáles son los lineamientos sobre el uso de Inteligencia Artificial para docentes y alumnos en ESAN?",
                    )
                with c_c2:
                    _topic_card(
                        "topic_2", "q_bach", "02", ICON["cap"],
                        "Requisitos para grado de bachiller",
                        "Cómputo de créditos académicos, acreditación obligatoria de idiomas y condiciones de graduación.",
                        "¿Cuáles son los requisitos y créditos necesarios para obtener el Grado de Bachiller?",
                    )
                    _topic_card(
                        "topic_4", "q_ret", "04", ICON["file"],
                        "Procedimiento de devolución y retiro",
                        "Plazos de desistimiento, cronograma de retenciones y trámite formal de retiro de programa.",
                        "¿Cuál es el procedimiento formal y plazos para solicitar devolución de dinero o retiro de programa?",
                    )

                st.markdown(
                    '<div class="steps-row">'
                    '<div class="step-box"><span class="step-n">1</span><div><div class="step-t">Selecciona</div><div class="step-d">Programa y consulta</div></div></div>'
                    '<div class="step-box"><span class="step-n">2</span><div><div class="step-t">Analiza</div><div class="step-d">Reglamentos relacionados</div></div></div>'
                    '<div class="step-box"><span class="step-n">3</span><div><div class="step-t">Verifica</div><div class="step-d">Dictamen y evidencia</div></div></div>'
                    '</div>',
                    unsafe_allow_html=True,
                )

        else:
            # ── Pantalla de dictamen ─────────────────────────────────────────
            st.markdown(
                panel_header(
                    "Resultado del análisis normativo",
                    "Dictamen preliminar y documentación recuperada",
                    eyebrow="Expediente de consulta",
                    right=f'<span class="chip green">{ICON["check_g"]}Análisis finalizado</span>',
                ),
                unsafe_allow_html=True,
            )
            # Espacio reservado para el indicador de carga dentro de este panel
            status_slot = st.empty()

            with st.container(key="res_body"):
                # ── CONSULTA EVALUADA (CONSULTA DEL ESTUDIANTE + PROGRAMA) ──────
                last_q = st.session_state.get("last_query", "")
                last_prog = st.session_state.get("last_program", "")
                last_extra = st.session_state.get("last_extra_ctx", "")

                if last_q:
                    badge_prog_html = (
                        f' · Programa {escape(str(last_prog))}'
                    ) if last_prog else ''

                    extra_html = (
                        f'<div class="query-extra"><b>Contexto del caso ingresado:</b> {escape(str(last_extra))}</div>'
                    ) if last_extra else ''

                    st.markdown(
                        f'<div class="query-card">{ICON["msg"]}'
                        f'<div style="min-width:0;flex:1">'
                        f'<div class="query-meta">Consulta evaluada{badge_prog_html}</div>'
                        f'<div class="query-text">{escape(str(last_q))}</div>'
                        f'{extra_html}'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )

                # ── VISTA DE RESULTADOS (2 COLUMNAS: DICTAMEN + EVIDENCIA) ──────
                col_resp, col_norm = st.columns([1.3, 1], gap="small")

                # ═══════════════════════════════ COLUMNA 1: DICTAMEN Y COMUNICACIÓN ═
                with col_resp:
                    with st.container(key="res_dictamen"):
                        _prog_sel = st.session_state.get('last_program', '') or st.session_state.get('programa_sel', '')

                        # 1. DICTAMEN EJECUTIVO Y PROCEDENCIA
                        dictamen = result.get("dictamen_coordinacion", {})
                        procedencia = dictamen.get("procedencia", "Evaluación de Coordinación Académica")
                        resumen_ejecutivo = dictamen.get("resumen_ejecutivo", "") or result.get("razonamiento", "")
                        appr = result.get("requiere_aprobacion", "La coordinadora puede resolver directamente")

                        # Estilo de badge según procedencia
                        proc_lower = procedencia.lower()
                        if "no procede" in proc_lower:
                            badge_bg, badge_border, badge_text = "#FDF0F0", "#F3C1C1", "#8F1116"
                            icon_badge = SVG["alert"]
                        elif "no contemplado" in proc_lower or "elevación" in proc_lower or "insuficiente" in proc_lower or "revisión" in proc_lower or "validación" in proc_lower:
                            badge_bg, badge_border, badge_text = "#FFF8EB", "#F6DDA8", "#8A5300"
                            icon_badge = SVG["alert"]
                        else:
                            badge_bg, badge_border, badge_text = "#F0F8F2", "#BFE3C9", "#17663A"
                            icon_badge = SVG["check"]

                        # Textos de apoyo de la interfaz según el estado ya calculado
                        _estado_ui = {
                            "#8F1116": ("red", "No procede", "La normativa recuperada no respalda la solicitud. Revise el sustento antes de comunicar una respuesta."),
                            "#8A5300": ("amber", "Requiere revisión", "Antes de comunicar una decisión, debe confirmarse la normativa vigente y su aplicabilidad al programa del participante."),
                            "#17663A": ("green", "Procede", "La solicitud encuentra respaldo en la normativa recuperada. Verifique el expediente antes de comunicar."),
                        }.get(badge_text, ("amber", "Requiere revisión", ""))

                        st.markdown(
                            block_header(
                                ICON["scale"], "Dictamen y validación",
                                f'<span class="state-badge {_estado_ui[0]}">{_estado_ui[1]}</span>',
                            ),
                            unsafe_allow_html=True,
                        )

                        # Las afirmaciones referenciadas sustituyen al resumen libre.
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
                                    refs = "".join(
                                        f'<a class="ref" href="#cita-{n}" title="Ver cita verificada {n}">{n}</a>'
                                        for n in numeros
                                    )
                                    partes_criterio.append(f"{escape(texto)} {refs}")

                        if partes_criterio:
                            criterio_html = " ".join(partes_criterio)
                        else:
                            # Sin relaciones verificadas, se conserva el resumen sin insertar
                            # referencias arbitrarias.
                            criterio_html = escape(resumen_ejecutivo)

                        st.markdown(
                            f'<div class="verdict" style="--v-bg:{badge_bg};--v-border:{badge_border};--v-text:{badge_text}">'
                            f'<div class="verdict-label">{icon_badge}{escape(procedencia)}</div>'
                            f'<div class="verdict-note">{_estado_ui[2]}</div>'
                            f'</div>'
                            f'<div class="mini-label" style="margin-top:12px">Criterio resolutivo preliminar</div>'
                            f'<div class="criterio" style="margin-top:6px">{criterio_html}</div>'
                            f'<div class="instance-row" style="margin-top:10px">{ICON["shield"]}<span><b>Instancia competente:</b> {escape(appr)}</span></div>',
                            unsafe_allow_html=True,
                        )

                        # 2. PROTOCOLO DE VERIFICACIÓN PARA LA COORDINADORA
                        actions = result.get("acciones_coordinadora", [])
                        if actions:
                            st.markdown(
                                f"<div class='mini-label'>Protocolo de verificación <span class='n'>· {len(actions)}</span></div>",
                                unsafe_allow_html=True,
                            )
                            with st.container(key="res_protocol"):
                                for act in actions:
                                    st.checkbox(act, value=False, key=f"chk_{hash(act)}")

                        # 3. FUNDAMENTACIÓN TÉCNICA
                        reason = result.get("razonamiento", "")
                        if reason:
                            with st.expander("Ver fundamentación y razonamiento normativo"):
                                st.markdown(f'<div class="reason-box">{escape(reason)}</div>', unsafe_allow_html=True)

                        # 4. BORRADOR DE COMUNICACIÓN PARA EL ESTUDIANTE (POST-VALIDACIÓN)
                        with st.container(key="res_draft"):
                            st.markdown(
                                f'<div class="draft-title">{ICON["mail"]}<span>Borrador para el estudiante</span>'
                                '<span class="draft-sub">· Post-validación</span></div>',
                                unsafe_allow_html=True,
                            )

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

                            # Botones de copia rápidos (la nota de bloqueo se muestra debajo)
                            b_c1, b_c2 = st.columns(2)
                            copia_bloqueada = bool(result.get("validacion_pendiente"))
                            if copia_bloqueada:
                                st.markdown('<div class="copy-note">Copia habilitada después de la validación del ámbito o de su sustento.</div>', unsafe_allow_html=True)
                            wa_text = f"*Coordinación Académica ESAN*\n\n{draft}"
                            # El borrador ya incluye saludo; no repetirlo en el correo copiado.
                            mail_text = draft

                            with b_c1:
                                if st.button("Copiar WhatsApp", key="btn_cp_wa", use_container_width=True, disabled=copia_bloqueada, icon=":material/chat:"):
                                    st.code(wa_text, language=None)
                                    st.toast("Texto preparado para WhatsApp — cópialo del recuadro")
                            with b_c2:
                                if st.button("Copiar correo", key="btn_cp_mail", use_container_width=True, disabled=copia_bloqueada, icon=":material/mail:"):
                                    st.code(mail_text, language=None)
                                    st.toast("Texto preparado para Correo — cópialo del recuadro")

                # ═══════════════════════════════════════ COLUMNA 2: EVIDENCIA DOCUMENTAL ═
                with col_norm:
                    with st.container(key="res_evidencia"):
                        citas = result.get("citas_normativas", [])
                        documentos = len({c.get("documento", "") for c in citas})
                        _docs_rel = documentos if citas else len({_s.get("source", "") for _s in sources[:6]})

                        st.markdown(
                            block_header(ICON["book"], "Evidencia documental")
                            + '<div class="stat-grid">'
                            f'<div class="stat {"ok" if citas else ""}"><div class="n">{len(citas)}</div><div class="t">Citas verificadas</div></div>'
                            f'<div class="stat hl"><div class="n">{_docs_rel}</div><div class="t">Documentos {"citados" if citas else "relacionados"}</div></div>'
                            '</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            '<div class="ev-disclaimer">'
                            f'Se recuperaron <b>{len(sources)}</b> fragmentos para el análisis. '
                            'Solo se muestran como verificados los extractos que coinciden literalmente con un fragmento recuperado; '
                            'esto no certifica vigencia ni aplicabilidad jurídica.'
                            '</div>',
                            unsafe_allow_html=True,
                        )
                        auditoria = result.get("auditoria_citas", {})
                        if auditoria.get("duplicadas") or auditoria.get("no_verificables"):
                            st.caption(
                                f"Control automático: {auditoria.get('duplicadas', 0)} citas duplicadas "
                                f"y {auditoria.get('no_verificables', 0)} citas sin coincidencia literal descartadas."
                            )
                        # Advertencias normativas (mismo texto y condiciones que antes)
                        if not citas:
                            st.markdown(
                                f'<div class="ev-alert" role="alert">{ICON["alert"]}<div>'
                                '<div class="t">Material contextual, no validado</div>'
                                '<div class="d">No hay citas textuales verificadas para respaldar este dictamen. '
                                'La búsqueda recuperó material contextual, pero no constituye por sí sola '
                                'prueba normativa. No emitir la comunicación sin revisión institucional.</div>'
                                '</div></div>',
                                unsafe_allow_html=True,
                            )
                        elif result.get("validacion_pendiente"):
                            st.markdown(
                                f'<div class="ev-alert" role="alert">{ICON["alert"]}<div>'
                                '<div class="t">Validación pendiente</div>'
                                '<div class="d">Hay citas literales, pero todavía falta confirmar el programa y '
                                'la aplicabilidad normativa. No emitir una respuesta definitiva hasta verificarlo.</div>'
                                '</div></div>',
                                unsafe_allow_html=True,
                            )

                        # ── Preparación de datos para mostrar (agrupación por documento) ──
                        import unicodedata as _ud

                        def _norm_docname(name):
                            n = _ud.normalize("NFD", name.lower())
                            n = "".join(c for c in n if _ud.category(c) != "Mn")
                            return n.strip()

                        def _group_items(raw_items):
                            """Agrupa citas/fragmentos por documento (una tarjeta por archivo)."""
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

                                # Extraer solo el número de página limpio (sin duplicar "Pág.")
                                p_raw = c.get("pagina", "")
                                p_digits = re.findall(r"\d+", str(p_raw))
                                clean_p = p_digits[0] if p_digits else str(p_raw).strip()

                                grouped[nkey]["citas"].append({
                                    "num": num,
                                    "seccion": c.get("seccion", ""),
                                    "pagina": clean_p,
                                    "extracto": c.get("extracto", ""),
                                })
                            return grouped, order

                        def _render_evidence(grouped, order, *, verified, card_prefix, btn_prefix, list_key):
                            """Tarjetas de documento. Verificadas y relacionadas se distinguen visualmente."""
                            tipo_ui = "Cita verificada" if verified else "Fragmento relacionado"
                            holder = (
                                st.container(height=540, key=list_key) if len(order) > 2
                                else st.container(key=list_key)
                            )
                            with holder:
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
                                    badge = (
                                        f'<span class="ev-badge ok">{ICON["check_ok"]}Verificada</span>' if verified
                                        else '<span class="ev-badge ctx">Contextual</span>'
                                    )

                                    with st.container(key=f"{card_prefix}_{card_idx}"):
                                        st.markdown(
                                            f'<div class="cite-head"><div class="cite-head-left">'
                                            f'<span class="doc-icon{"" if verified else " ctx"}">{ICON["file_sm"] if verified else ICON["file_ctx"]}</span>'
                                            f'<div style="min-width:0">'
                                            f'<div class="cite-source">{safe_doc_name}</div>'
                                            f'<div class="cite-meta">{badge}<span>{len(all_citas)} referencia(s)</span>'
                                            f'<a href="{safe_drive_url}" target="_blank" rel="noopener noreferrer" class="drive-link">Ver en Drive ↗</a></div>'
                                            f'</div></div></div>',
                                            unsafe_allow_html=True,
                                        )

                                        for sub_i, cita in enumerate(all_citas, 1):
                                            c_sec = cita["seccion"]
                                            c_pg = cita["pagina"]
                                            c_ext = cita["extracto"]

                                            num_html = f'<span class="cite-num">{cita["num"]}</span>' if verified and isinstance(cita["num"], int) else ""
                                            badge_pg = f'<span>Pág. {escape(str(c_pg))}</span>·' if c_pg else ""
                                            sec_text = f'·<b>{escape(str(c_sec))}</b>' if c_sec else ""
                                            anchor_id = f' id="cita-{cita["num"]}"' if verified and isinstance(cita["num"], int) else ""
                                            excerpt_html = (
                                                f'<div class="cite-excerpt{"" if verified else " ctx"}">“{escape(str(c_ext))}…”</div>'
                                                if c_ext else ""
                                            )

                                            st.markdown(
                                                f'<div class="cite-item"{anchor_id}>'
                                                f'<div class="cite-item-meta">{num_html}{badge_pg}<span>{tipo_ui}</span>{sec_text}</div>'
                                                f'{excerpt_html}'
                                                f'</div>',
                                                unsafe_allow_html=True,
                                            )

                                            btn_label = f"Abrir documento · pág. {c_pg} ↗" if c_pg else "Abrir documento ↗"
                                            btn_help = (
                                                f"Abre el PDF oficial en la página {c_pg} y resalta este extracto exacto."
                                                if verified else
                                                f"Abre el documento en la página {c_pg} para revisar el fragmento recuperado."
                                            )
                                            if st.button(
                                                btn_label,
                                                key=f"{btn_prefix}_{card_idx}_{sub_i}_{cita['num']}",
                                                use_container_width=True,
                                                type="primary" if verified else "secondary",
                                                help=btn_help,
                                            ):
                                                show_pdf_viewer_modal(
                                                    doc_name=doc_name,
                                                    page_val=c_pg,
                                                    extracto=c_ext,
                                                    seccion=c_sec,
                                                    fallback_source=fallback_source,
                                                    drive_url=doc_drive_url,
                                                )
                            if len(order) > 2:
                                st.markdown('<div class="scroll-hint">⌄ Desplaza para revisar más documentos</div>', unsafe_allow_html=True)

                        # Citas verificadas: exclusivamente las validadas por el motor
                        verified_items = {}
                        for c in citas:
                            num = c.get("numero", 0)
                            verified_items[num] = c

                        # Documentos relacionados: fragmentos recuperados (material contextual)
                        related_items = {}
                        for i_s, s in enumerate(sources[:6], 1):
                            related_items[i_s] = {
                                "numero": i_s,
                                "documento": s["source"],
                                "seccion": s["section"],
                                "pagina": str(s["page"]),
                                "extracto": s["text"][:220],
                            }

                        grouped_v, order_v = _group_items(verified_items)
                        grouped_r, order_r = _group_items(related_items)

                        tab_ver, tab_rel = st.tabs([
                            f"Verificadas ({len(citas)})",
                            f"Relacionadas ({len(order_r)})",
                        ])
                        with tab_ver:
                            if order_v:
                                st.markdown(
                                    f'<div class="ev-note ok">{ICON["check_ok"]}<span>Extractos validados literalmente por el sistema. '
                                    'Los números enlazan con el criterio resolutivo.</span></div>',
                                    unsafe_allow_html=True,
                                )
                                _render_evidence(
                                    grouped_v, order_v, verified=True,
                                    card_prefix="doc_card", btn_prefix="btn_pdf_modal", list_key="evid_list",
                                )
                            else:
                                st.markdown(
                                    '<div class="ev-empty"><b>Sin citas verificadas</b>'
                                    'El material recuperado está en la pestaña «Relacionadas» y es solo contextual.</div>',
                                    unsafe_allow_html=True,
                                )
                        with tab_rel:
                            st.markdown(
                                '<div class="ev-note ctx"><span>Fragmentos recuperados por relevancia. '
                                '<b>No constituyen por sí solos sustento normativo verificado.</b></span></div>',
                                unsafe_allow_html=True,
                            )
                            if order_r:
                                # Sin citas verificadas se conservan las claves originales de estos botones.
                                _render_evidence(
                                    grouped_r, order_r, verified=False,
                                    card_prefix="rel_card" if citas else "doc_card",
                                    btn_prefix="btn_rel_modal" if citas else "btn_pdf_modal",
                                    list_key="evid_list_rel",
                                )
                            else:
                                st.markdown(
                                    '<div class="ev-empty"><b>Sin fragmentos recuperados</b>'
                                    'La búsqueda no devolvió material para esta consulta.</div>',
                                    unsafe_allow_html=True,
                                )

# ── PANEL DE CONSULTA (IZQUIERDA) ─────────────────────────────────────────────
with col_form:
    st.markdown(section_label("01 · Consulta y contexto", "Panel de trabajo"), unsafe_allow_html=True)
    with st.container(key="panel_form"):
        st.markdown(
            panel_header(
                "Nueva consulta",
                "Programa, caso y contexto académico",
                ICON["search"],
            ),
            unsafe_allow_html=True,
        )

        current_form_id = st.session_state.form_id

        # ── SELECTOR DE PROGRAMA (OBLIGATORIO) ────────────────────────────────
        st.markdown(
            '<div class="field-row"><span class="lbl">Programa del participante<span class="req">*</span></span>'
            '<span class="aux">Obligatorio</span></div>',
            unsafe_allow_html=True
        )
        programa_options = ["PEE", "PADE", "PAE", "Diploma", "CIE", "Maestría", "Doctorado"]
        programa_sel = st.segmented_control(
            "Programa",
            options=programa_options,
            selection_mode="single",
            default=None,
            label_visibility="collapsed",
            key=f"prog_choice_{current_form_id}",
        )

        if not programa_sel:
            st.markdown(
                '<div class="field-hint warn">'
                f'{ICON["alert"]}Seleccione el programa antes de emitir el dictamen.'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="field-hint ok">{ICON["check_r"]}Programa seleccionado: {programa_sel}</div>',
                unsafe_allow_html=True
            )

        st.markdown(
            '<div class="field-row"><span class="lbl">Consulta del estudiante<span class="req">*</span></span>'
            '<span class="aux">Máx. 4,000 caracteres</span></div>',
            unsafe_allow_html=True
        )
        default_prompt = st.session_state.quick_q or ""
        student_msg = st.text_area(
            "Consulta del estudiante",
            value=default_prompt,
            height=170,
            placeholder="Pegue o redacte aquí la consulta del estudiante (ej: Alumno solicita rendir evaluación extemporánea por descanso médico)...",
            label_visibility="collapsed",
            max_chars=4000,
            key=f"query_text_{current_form_id}",
        )
        st.markdown('<div class="field-help">Incluya hechos, fechas y circunstancias relevantes.</div>', unsafe_allow_html=True)

        # Si se cargo desde quick_q, limpiar para que no persista en futuros reruns
        if st.session_state.quick_q:
            st.session_state.quick_q = None

        with st.expander("Contexto adicional del caso (opcional)"):
            extra_ctx = st.text_input(
                "Contexto extra",
                placeholder="Ej: ciclo V, alumno presenta solicitud fuera de plazo de 48 horas...",
                label_visibility="collapsed",
                key=f"extra_ctx_{current_form_id}",
            )

        can_run = bool(api_key and student_msg.strip() and programa_sel)

        # Botón principal prominente con feedback visual
        col_run, col_hist = st.columns([3, 1])
        with col_run:
            button_help = "Haga clic para consultar la base reglamentaria y generar el dictamen." if can_run else "Seleccione el tipo de programa y escriba la consulta para activar este botón."
            run_query = st.button(
                "Consultar reglamentos y emitir dictamen  →",
                use_container_width=True,
                type="primary",
                disabled=not can_run,
                help=button_help,
            )
        with col_hist:
            if st.session_state.history:
                st.markdown(
                    f'<div class="hist-note">{len(st.session_state.history)} consulta(s) previas en esta sesión</div>',
                    unsafe_allow_html=True
                )

        st.markdown(
            f'<div class="form-note">{ICON["info"]}<div>'
            '<div class="t">Toda respuesta requiere validación institucional</div>'
            '<div class="d">La IA recupera información del repositorio institucional. '
            'La evaluación definitiva corresponde al área competente.</div>'
            '</div></div>',
            unsafe_allow_html=True,
        )

    # ── PROCESAMIENTO DE CONSULTA ─────────────────────────────────────────────────
    if run_query and api_key and student_msg.strip():
        # Indicador de carga inmediato DENTRO del panel de resultados (derecha).
        # Mientras se procesa, el contenido anterior del panel se atenúa.
        loading_slot = status_slot
        _loading_icon = (
            f'<img src="data:image/png;base64,{ACI_ICON_B64}" alt="" />' if ACI_ICON_B64 else ''
        )
        loading_slot.markdown(
            '<style>'
            '.st-key-res_body, .st-key-welcome_body {'
            'opacity: .28; filter: grayscale(.6); pointer-events: none; transition: opacity .2s ease; }'
            '</style>'
            '<div class="loading-card" role="status" aria-live="polite">'
            f'<div class="loading-mark"><div class="spinner"></div>{_loading_icon}</div>'
            '<div class="loading-text">'
            f'<div class="loading-title">ACI ESAN está procesando el dictamen · {escape(str(programa_sel))}</div>'
            '<div class="loading-sub">Buscando artículos en reglamentos vigentes, evaluando procedencia y generando propuesta de correo…</div>'
            '<div class="loading-steps"><span>Recuperación normativa</span><span>Validación de citas</span><span>Redacción del borrador</span></div>'
            '</div>'
            '<div class="loading-bar"></div>'
            '</div>',
            unsafe_allow_html=True
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
            loading_slot.empty()
            st.rerun()
        except Exception as e:
            # El error se muestra en el mismo lugar donde estaba el indicador de carga
            loading_slot.error(f"Error al procesar la consulta: {e}")
