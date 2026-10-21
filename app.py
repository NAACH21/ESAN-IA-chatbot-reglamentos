import os
import sys
import base64
import importlib
import streamlit as st
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
from drive_sync import (
    sync_from_drive,
    download_full_folder_gdown,
    list_drive_files,
    get_indexed_sources,
    get_file_drive_url,
)

load_dotenv()

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

# ── SESSION STATE ─────────────────────────────────────────────────────────────
for key, default in [
    ("history", []),
    ("last_result", None),
    ("last_sources", []),
    ("quick_q", None),
    ("programa_sel", None),
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
        st.rerun()

# ── TOP NAVBAR INSTITUCIONAL REIMAGINADO (SIN FORMATO IA DE JUGUETE) ─────────
logo_html = f'<img src="data:image/svg+xml;base64,{ESAN_LOGO_B64}" alt="Universidad ESAN" class="esan-nav-logo" />' if ESAN_LOGO_B64 else '<div style="font-weight:900;color:#D31418;font-size:1.15rem">ESAN</div>'

st.markdown(f"""
<div class="esan-navbar">
  <div class="esan-nav-left">
    {logo_html}
    <div class="esan-nav-divider"></div>
    <div>
      <div class="esan-nav-title">SISTEMA INSTITUCIONAL DE CONSULTA Y GESTIÓN NORMATIVA</div>
      <div class="esan-nav-subtitle">Coordinación Académica · Marco Reglamentario Vigente</div>
    </div>
  </div>
  <div class="esan-nav-right">
    <div class="esan-status-indicator">
      <span class="pulse-dot"></span>
      <span class="status-text">Repositorio Vigente · 43 Reglamentos</span>
    </div>
    <div class="status-pill-model">
      {SVG["shield_check"]} Verificación Oficial
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── ÁREA PRINCIPAL: RESULTADOS O PANTALLA DE BIENVENIDA ───────────────────────
result = st.session_state.last_result
sources = st.session_state.last_sources

if result is None:
    # Pantalla Institucional de Casos Frecuentes
    st.markdown("""
    <div class="hero-box">
      <div class="hero-title">Consultas Reglamentarias y Casuística Estudiantil</div>
      <div class="hero-desc">
        Seleccione un tipo de consulta frecuente o ingrese la solicitud del estudiante en el panel inferior para obtener la fundamentación reglamentaria, instancia resolutiva y propuesta de respuesta.
      </div>
    </div>
    """, unsafe_allow_html=True)

    c_c1, c_c2 = st.columns(2)
    with c_c1:
        st.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:8px;padding:14px;margin-bottom:10px">
          <div style="font-weight:700;font-size:0.92rem;color:#0F172A;display:flex;align-items:center;gap:6px">
            {SVG['scale']} Reconsideración de Evaluaciones
          </div>
          <div style="font-size:0.8rem;color:#64748B;margin:4px 0 10px 0">
            Plazos reglamentarios, requisitos formales e instancia resolutiva ante reclamos de calificación.
          </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Consultar plazos y trámite de reconsideración →", key="q_recons", use_container_width=True):
            st.session_state.quick_q = "¿Cuándo puedo pedir reconsideración de nota y ante qué autoridad corresponde presentarla?"
            st.rerun()

        st.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:8px;padding:14px;margin:14px 0 10px 0">
          <div style="font-weight:700;font-size:0.92rem;color:#0F172A;display:flex;align-items:center;gap:6px">
            {SVG['chip']} Lineamientos de Inteligencia Artificial
          </div>
          <div style="font-size:0.8rem;color:#64748B;margin:4px 0 10px 0">
            Disposiciones institucionales para docentes y alumnos sobre el uso de herramientas IA en evaluaciones.
          </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Consultar directiva sobre uso de IA →", key="q_ia", use_container_width=True):
            st.session_state.quick_q = "¿Cuáles son los lineamientos sobre el uso de Inteligencia Artificial para docentes y alumnos en ESAN?"
            st.rerun()

    with c_c2:
        st.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:8px;padding:14px;margin-bottom:10px">
          <div style="font-weight:700;font-size:0.92rem;color:#0F172A;display:flex;align-items:center;gap:6px">
            {SVG['grad_cap']} Requisitos para Grado de Bachiller
          </div>
          <div style="font-size:0.8rem;color:#64748B;margin:4px 0 10px 0">
            Cómputo de créditos académicos, acreditación obligatoria de idiomas y condiciones de graduación.
          </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Consultar requisitos de bachillerato →", key="q_bach", use_container_width=True):
            st.session_state.quick_q = "¿Cuáles son los requisitos y créditos necesarios para obtener el Grado de Bachiller?"
            st.rerun()

        st.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:8px;padding:14px;margin:14px 0 10px 0">
          <div style="font-weight:700;font-size:0.92rem;color:#0F172A;display:flex;align-items:center;gap:6px">
            {SVG['book']} Procedimiento de Devolución y Retiro
          </div>
          <div style="font-size:0.8rem;color:#64748B;margin:4px 0 10px 0">
            Plazos de desistimiento, cronograma de retenciones y trámite formal de retiro de programa.
          </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Consultar política de devoluciones y retiros →", key="q_ret", use_container_width=True):
            st.session_state.quick_q = "¿Cuál es el procedimiento formal y plazos para solicitar devolución de dinero o retiro de programa?"
            st.rerun()

else:
    # ── VISTA DE RESULTADOS (2 COLUMNAS: RESPUESTA + RESPALDO NORMATIVO) ───────
    col_resp, col_norm = st.columns([1.35, 1.15], gap="large")

    # ═════════════════════════════════════════ COLUMNA 1: RESPUESTA PROPUESTA ═
    with col_resp:
        _prog_sel = st.session_state.get('programa_sel', '')
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
        elif "no contemplado" in proc_lower or "elevación" in proc_lower:
            badge_bg, badge_border, badge_text = "#FFFBEB", "#FDE68A", "#92400E"
            icon_badge = SVG["alert"]
        else:
            badge_bg, badge_border, badge_text = "#F0FDF4", "#BBF7D0", "#166534"
            icon_badge = SVG["check"]

        st.markdown(f"""
        <div style="background:{badge_bg};border:1.5px solid {badge_border};border-radius:8px;padding:12px 16px;margin-bottom:12px">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
            <span style="font-weight:800;font-size:0.88rem;color:{badge_text};text-transform:uppercase;letter-spacing:0.02em">
              {icon_badge} {procedencia}
            </span>
          </div>
          <div style="font-size:0.86rem;color:#1E293B;line-height:1.5;margin-bottom:8px">
            <b>Criterio resolutivo:</b> {resumen_ejecutivo}
          </div>
          <div style="font-size:0.8rem;color:#475569;display:flex;align-items:center;gap:6px;border-top:1px solid rgba(0,0,0,0.06);padding-top:6px">
            {SVG['shield_check']} <span><b>Instancia competente:</b> {appr}</span>
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
                st.markdown(f'<div class="reason-box">{reason}</div>', unsafe_allow_html=True)

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

        st.markdown(f'<div class="draft-box">{draft}</div>', unsafe_allow_html=True)

        # Botones de Copia Rápidos
        b_c1, b_c2 = st.columns(2)
        wa_text = f"*Coordinación Académica ESAN*\n\n{draft}"
        mail_text = f"Estimado/a estudiante,\n\n{draft}"

        with b_c1:
            if st.button("Copiar formato WhatsApp", key="btn_cp_wa", use_container_width=True):
                st.code(wa_text, language=None)
                st.toast("Texto preparado para WhatsApp — cópialo del recuadro")
        with b_c2:
            if st.button("Copiar formato Correo", key="btn_cp_mail", use_container_width=True):
                st.code(mail_text, language=None)
                st.toast("Texto preparado para Correo — cópialo del recuadro")

    # ═════════════════════════════════════════ COLUMNA 2: RESPALDO NORMATIVO ═══
    with col_norm:
        st.markdown(f'<div class="esan-card-title">{SVG["book"]} Respaldo Normativo y Citas ({len(sources)} fuentes)</div>', unsafe_allow_html=True)

        citas = result.get("citas_normativas", [])
        displayed = {}
        for c in citas:
            num = c.get("numero", 0)
            displayed[num] = c

        if not displayed:
            for i_s, s in enumerate(sources[:6], 1):
                displayed[i_s] = {
                    "numero": i_s,
                    "documento": s["source"],
                    "seccion": s["section"],
                    "pagina": str(s["page"]),
                    "extracto": s["text"][:220],
                }

        for num, c in sorted(displayed.items()):
            doc_name = c.get("documento", "")
            seccion = c.get("seccion", "")
            pagina = c.get("pagina", "")
            extracto = c.get("extracto", "")

            doc_drive_url = get_file_drive_url(doc_name, pagina)
            badge_page = f'<span style="background:#FEF2F2;color:#991B1B;padding:2px 8px;border-radius:6px;font-size:0.73rem;font-weight:700;border:1px solid #FECACA;margin-left:6px">Pág. {pagina}</span>' if pagina else ""

            st.markdown(f"""
            <div class="cite-card">
              <div style="display:flex;align-items:flex-start;gap:8px;margin-bottom:6px">
                <span class="cite-num">{num}</span>
                <div style="flex:1">
                  <div class="cite-source">{doc_name}</div>
                  <div class="cite-section">{seccion}{badge_page}</div>
                </div>
              </div>
              {f'<div class="cite-excerpt">"{extracto}..."</div>' if extracto else ''}
              <div style="margin-top:8px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:6px">
                <a href="{doc_drive_url}" target="_blank" class="drive-link">
                  {SVG["drive"]} Abrir archivo en Google Drive
                </a>
                {f'<span style="font-size:0.74rem;color:#64748B;display:inline-flex;align-items:center;gap:4px">{SVG["pin"]} Ubicación: <b>Pág. {pagina}</b></span>' if pagina else ''}
              </div>
            </div>
            """, unsafe_allow_html=True)

# ── BOTTOM DOCK DE CONSULTA (AMPLIO, ELEGANTE E INSTITUCIONAL) ────────────────
st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

with st.container():
    st.markdown(f"""
    <div style="border-top:2px solid #F1F5F9;padding-top:14px;margin-top:10px">
      <div style="display:flex;align-items:center;gap:6px;font-size:0.88rem;font-weight:700;color:#0F172A;margin-bottom:6px">
        {SVG["search"]} Consulta o Caso del Estudiante
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── SELECTOR DE PROGRAMA (OBLIGATORIO) ────────────────────────────────
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:6px;font-size:0.82rem;font-weight:700;color:#0F172A;margin-bottom:4px">'
        f'{SVG["grad_cap"]} Tipo de programa del participante '
        '<span style="color:#D31418;font-weight:900">*</span>'
        '<span style="font-weight:400;color:#64748B;font-size:0.75rem;margin-left:4px">(obligatorio)</span>'
        '</div>',
        unsafe_allow_html=True
    )
    programa_options = ["PEE", "PADE", "PAE", "Diploma", "CIE", "Maestr\u00eda", "Doctorado"]
    st.radio(
        "Programa",
        options=programa_options,
        index=None,
        horizontal=True,
        label_visibility="collapsed",
        key="programa_sel",
    )
    if not st.session_state.get("programa_sel"):
        st.markdown(
            '<div style="color:#D31418;font-size:0.78rem;margin-bottom:6px;display:flex;align-items:center;gap:4px">'
            f'{SVG["alert"]} Debe seleccionar el tipo de programa antes de emitir el dictamen.'
            '</div>',
            unsafe_allow_html=True
        )
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    default_prompt = st.session_state.quick_q or ""
    c_input, c_btn = st.columns([5.5, 1.5], gap="small")

    with c_input:
        student_msg = st.text_area(
            "Consulta del estudiante",
            value=default_prompt,
            height=85,
            placeholder="Pegue o redacte aquí la consulta del estudiante (ej: Alumno solicita rendir evaluación extemporánea por descanso médico)...",
            label_visibility="collapsed",
            max_chars=4000,
        )

    with c_btn:
        st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)
        can_run = bool(api_key and student_msg.strip() and st.session_state.get("programa_sel"))
        run_query = st.button("Emitir Dictamen →", use_container_width=True, type="primary", disabled=not can_run)
        if st.session_state.history:
            st.caption(f"Historial: {len(st.session_state.history)} consultas")

    with st.expander("Parámetros opcionales del caso (programa, ciclo, antecedentes del alumno)"):
        extra_ctx = st.text_input(
            "Contexto extra",
            placeholder="Ej: Programa DPA, ciclo V, alumno presenta solicitud fuera de plazo de 48 horas...",
            label_visibility="collapsed"
        )

    if st.session_state.quick_q:
        st.session_state.quick_q = None

# ── PROCESAMIENTO DE CONSULTA ─────────────────────────────────────────────────
if run_query and api_key and student_msg.strip():
    with st.spinner("Buscando artículos en reglamentos y redactando dictamen institucional..."):
        try:
            if not is_indexed and api_key:
                build_vector_store(api_key=api_key)
                rag = RAGEngine(api_key=api_key, openrouter_key=or_key)

            result_data, sources_data = rag.answer_question(
                query=student_msg.strip(),
                context_extra=extra_ctx.strip() if 'extra_ctx' in locals() and extra_ctx else "",
                model_label=model_label,
                chat_history=st.session_state.history,
                programa=st.session_state.get("programa_sel", ""),
            )
            st.session_state.last_result = result_data
            st.session_state.last_sources = sources_data
            st.session_state.history.append(student_msg.strip()[:120])
            st.rerun()
        except Exception as e:
            st.error(f"Error al procesar la consulta: {e}")
