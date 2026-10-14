import os
import sys
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
    remove_document_from_store,
    index_single_pdf,
    restore_archived_documents,
    is_supported_file,
    get_all_supported_files,
    DOCS_FOLDER,
    ARCHIVE_FOLDER,
)
from rag_engine import RAGEngine, MODELS, GOOGLE_DRIVE_FOLDER
from drive_sync import sync_from_drive, download_full_folder_gdown, list_drive_files, get_indexed_sources

load_dotenv()

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Asistente de Coordinación Académica · ESAN",
    page_icon="https://www.ue.edu.pe/favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── SVG ICONS ────────────────────────────────────────────────────────────────
SVG = {
    "settings": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#002855" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:5px"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.6 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.6a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>""",
    "db": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#002855" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:5px"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>""",
    "file": """<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#002855" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:4px"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>""",
    "check": """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#16a34a" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:4px"><polyline points="20 6 9 17 4 12"/></svg>""",
    "warn": """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#d97706" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:4px"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>""",
    "book": """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:4px"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>""",
    "drive": """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#1967D2" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:4px"><path d="M21 15a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V9a2 2 0 0 1 2-2h1l2-3h6l2 3h1a2 2 0 0 1 2 2z"/></svg>""",
    "copy": """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:4px"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>""",
    "wa": """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="#25D366" style="vertical-align:-2px;margin-right:4px"><path d="M17.472 14.382c-.297-.149-1.758-.867-2.03-.967-.273-.099-.471-.148-.67.15-.197.297-.767.966-.94 1.164-.173.199-.347.223-.644.075-.297-.15-1.255-.463-2.39-1.475-.883-.788-1.48-1.761-1.653-2.059-.173-.297-.018-.458.13-.606.134-.133.298-.347.446-.52.149-.174.198-.298.298-.497.099-.198.05-.371-.025-.52-.075-.149-.669-1.612-.916-2.207-.242-.579-.487-.5-.669-.51-.173-.008-.371-.01-.57-.01-.198 0-.52.074-.792.372-.272.297-1.04 1.016-1.04 2.479 0 1.462 1.065 2.875 1.213 3.074.149.198 2.096 3.2 5.077 4.487.709.306 1.262.489 1.694.625.712.227 1.36.195 1.871.118.571-.085 1.758-.719 2.006-1.413.248-.694.248-1.289.173-1.413-.074-.124-.272-.198-.57-.347m-5.421 7.403h-.004a9.87 9.87 0 0 1-5.031-1.378l-.361-.214-3.741.982.998-3.648-.235-.374a9.86 9.86 0 0 1-1.51-5.26c.001-5.45 4.436-9.884 9.888-9.884 2.64 0 5.122 1.03 6.988 2.898a9.825 9.825 0 0 1 2.893 6.994c-.003 5.45-4.437 9.884-9.885 9.884m8.413-18.297A11.815 11.815 0 0 0 12.05 0C5.495 0 .16 5.335.157 11.892c0 2.096.547 4.142 1.588 5.945L.057 24l6.305-1.654a11.882 11.882 0 0 0 5.683 1.448h.005c6.554 0 11.89-5.335 11.893-11.893a11.821 11.821 0 0 0-3.48-8.413z"/></svg>""",
}

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Column cards */
.col-card {
    background: #fff;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 18px 16px;
    height: 100%;
}
.col-number {
    font-size: 0.7rem;
    font-weight: 800;
    color: #94A3B8;
    letter-spacing: .12em;
    margin-bottom: 4px;
}
.col-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #002855;
    margin-bottom: 14px;
    border-bottom: 2px solid #E2E8F0;
    padding-bottom: 8px;
}
/* Approval badge */
.appr-badge {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    background: #FFF8E1;
    border: 1px solid #FFD54F;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 12px;
    font-size: 0.86rem;
}
.appr-icon {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: #FFD54F;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
}
/* Draft box */
.draft-box {
    background: #F8FAFC;
    border: 1px solid #CBD5E1;
    border-radius: 8px;
    padding: 14px;
    font-size: 0.88rem;
    line-height: 1.6;
    color: #1E293B;
    white-space: pre-wrap;
    margin-bottom: 10px;
}
/* Action item */
.action-item {
    display: flex;
    gap: 8px;
    align-items: flex-start;
    padding: 7px 10px;
    background: #F1F5F9;
    border-radius: 6px;
    font-size: 0.84rem;
    margin-bottom: 5px;
    color: #334155;
}
/* Reasoning box */
.reason-box {
    background: #F0FDF4;
    border-left: 3px solid #22C55E;
    padding: 10px 12px;
    border-radius: 4px;
    font-size: 0.82rem;
    color: #166534;
    line-height: 1.5;
}
/* Citation card */
.cite-card {
    background: #FAFBFF;
    border: 1px solid #BFDBFE;
    border-left: 4px solid #2563EB;
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 10px;
    font-size: 0.82rem;
}
.cite-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    background: #2563EB;
    color: #fff;
    font-size: 0.72rem;
    font-weight: 700;
    margin-right: 8px;
    flex-shrink: 0;
}
.cite-source {
    font-weight: 700;
    color: #1E3A8A;
}
.cite-section {
    color: #3B82F6;
    font-size: 0.78rem;
}
.cite-excerpt {
    background: #FFF9C4;
    border-radius: 3px;
    padding: 4px 6px;
    margin-top: 6px;
    font-style: italic;
    color: #374151;
    line-height: 1.4;
}
.drive-link {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    color: #1967D2;
    text-decoration: none;
    font-size: 0.8rem;
    font-weight: 600;
    margin-top: 6px;
}
/* Sidebar */
.sb-section {
    font-size: 0.85rem;
    font-weight: 700;
    color: #002855;
    margin: 12px 0 6px 0;
    display: flex;
    align-items: center;
}
.status-ok  { padding:5px 10px; border-radius:6px; background:#F0FDF4; border:1px solid #BBF7D0; color:#166534; font-size:.8rem; margin-bottom:8px; }
.status-warn{ padding:5px 10px; border-radius:6px; background:#FFFBEB; border:1px solid #FDE68A; color:#92400E; font-size:.8rem; margin-bottom:8px; }
.status-info{ padding:5px 10px; border-radius:6px; background:#EFF6FF; border:1px solid #BFDBFE; color:#1E40AF; font-size:.8rem; margin-bottom:8px; }
/* copy button style override */
div[data-testid="stButton"] > button[kind="secondary"] {
    border-radius: 6px;
    font-size: 0.83rem;
}
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ─────────────────────────────────────────────────────────────
for key, default in [
    ("history", []),
    ("last_result", None),
    ("last_sources", []),
    ("quick_q", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Logo_Universidad_ESAN.svg/320px-Logo_Universidad_ESAN.svg.png",
        width=170,
    )

    # API Keys
    st.markdown(f'<div class="sb-section">{SVG["settings"]} Credenciales</div>', unsafe_allow_html=True)
    default_oai = os.getenv("OPENAI_API_KEY", "")
    try:
        default_oai = default_oai or st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        pass

    default_or = os.getenv("OPENROUTER_API_KEY", "")
    try:
        default_or = default_or or st.secrets.get("OPENROUTER_API_KEY", "")
    except Exception:
        pass

    has_env = bool(default_oai)

    if has_env:
        st.markdown(f'<div class="status-ok">{SVG["check"]} Keys cargadas automáticamente (.env)</div>', unsafe_allow_html=True)
        with st.expander("Ver / Cambiar API Keys", expanded=False):
            api_key = st.text_input("OpenAI API Key", value=default_oai, type="password", help="Embeddings + GPT-4o-mini")
            or_key = st.text_input("OpenRouter API Key (MiMo)", value=default_or, type="password", help="MiMo v2.6 Pro")
    else:
        st.markdown(f'<div class="status-warn">{SVG["warn"]} Ingresa tu API Key de OpenAI</div>', unsafe_allow_html=True)
        api_key = st.text_input("OpenAI API Key", value=default_oai, type="password", help="Embeddings + GPT-4o-mini")
        or_key = st.text_input("OpenRouter API Key (MiMo)", value=default_or, type="password", help="MiMo v2.6 Pro")
        st.caption("Tip: Guárdalas en el archivo `.env` para cargarlas sin tener que escribirlas.")

    st.markdown("---")

    # Base documental
    st.markdown(f'<div class="sb-section">{SVG["db"]} Base Documental</div>', unsafe_allow_html=True)
    rag = RAGEngine(api_key=api_key, openrouter_key=or_key) if api_key else None
    is_indexed = rag.is_indexed() if rag else False

    if api_key and not is_indexed and "auto_idx_done" not in st.session_state:
        st.session_state.auto_idx_done = True
        with st.spinner("Inicializando base documental..."):
            try:
                build_vector_store(api_key=api_key)
                rag = RAGEngine(api_key=api_key, openrouter_key=or_key)
                is_indexed = rag.is_indexed()
            except Exception:
                pass

    if is_indexed:
        cnt = rag.get_document_count()
        st.markdown(f'<div class="status-info">{SVG["book"]} <b>{cnt}</b> secciones indexadas</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="status-warn">{SVG["warn"]} Sin indexar</div>', unsafe_allow_html=True)

    # ── Sincronización con Google Drive ──────────────────────────────────────
    st.markdown("---")
    st.markdown(f'<div class="sb-section">{SVG["drive"]} Sincronizar con Google Drive</div>', unsafe_allow_html=True)

    drive_url_input = st.text_input(
        "URL de la carpeta de Drive",
        value=GOOGLE_DRIVE_FOLDER,
        help="Pega el enlace de la carpeta compartida de Google Drive",
        label_visibility="collapsed",
        placeholder="https://drive.google.com/drive/folders/...",
    )

    # Vista previa de archivos en Drive
    if st.button("Ver documentos en Drive", use_container_width=True, key="btn_drive_preview"):
        with st.spinner("Consultando Google Drive..."):
            try:
                drive_files = list_drive_files(drive_url_input)
                indexed = set(get_indexed_sources())
                new_in_drive = [f for f in drive_files if f["name"] not in indexed]
                st.session_state["drive_preview"] = {
                    "all": drive_files,
                    "new": new_in_drive,
                }
            except Exception as e:
                st.error(f"Error al consultar Drive: {e}")

    if "drive_preview" in st.session_state and st.session_state["drive_preview"]:
        preview = st.session_state["drive_preview"]
        total = len(preview["all"])
        new_count = len(preview["new"])
        already = total - new_count
        st.markdown(
            f'<div class="status-info">{SVG["book"]} Drive: <b>{total}</b> docs — '
            f'<b style="color:#16a34a">{already} ya indexados</b> · '
            f'<b style="color:#d97706">{new_count} nuevos</b></div>',
            unsafe_allow_html=True,
        )
        if preview["new"]:
            with st.expander(f"Documentos nuevos en Drive ({new_count})"):
                for nf in preview["new"]:
                    st.markdown(f"{SVG['file']} <small>{nf['name']}</small>", unsafe_allow_html=True)

    # Botón de sincronización
    btn_sync_label = "Sincronizar documentos nuevos desde Drive"
    if st.button(btn_sync_label, use_container_width=True,
                 disabled=not bool(api_key), key="btn_drive_sync"):
        if not drive_url_input:
            st.error("Ingresa la URL de la carpeta de Drive.")
        else:
            status_box = st.empty()
            progress_msgs = []

            def _cb(msg):
                progress_msgs.append(msg)
                status_box.text(msg)

            with st.spinner("Sincronizando con Google Drive..."):
                try:
                    sync_result = sync_from_drive(
                        folder_url=drive_url_input,
                        docs_dir=DOCS_FOLDER,
                        api_key="",  # carpeta pública, no necesita key
                        progress_callback=_cb,
                    )
                    new_files = sync_result.get("new_files", [])
                    failed    = sync_result.get("failed", [])

                    if new_files:
                        _cb(f"Indexando {len(new_files)} documento(s) nuevo(s)...")
                        for nf in new_files:
                            path = os.path.join(DOCS_FOLDER, nf)
                            if os.path.exists(path):
                                n_chunks = index_single_pdf(api_key, path)
                                _cb(f"  ✓ {nf} ({n_chunks} secciones)")
                        st.success(
                            f"Sincronización completa: {len(new_files)} nuevo(s) indexado(s). "
                            f"{sync_result['already_indexed']} ya estaban indexados."
                            + (f" ({len(failed)} fallaron)" if failed else "")
                        )
                        # Limpiar preview para que se refresque
                        st.session_state.pop("drive_preview", None)
                        st.rerun()
                    else:
                        st.info(
                            f"No hay documentos nuevos en Drive. "
                            f"({sync_result['already_indexed']} ya indexados, "
                            f"{sync_result['drive_total']} encontrados en Drive)"
                        )
                except Exception as e:
                    st.error(f"Error durante la sincronización: {e}")

    # Sincronización completa (primera vez o reset)
    with st.expander("Primera vez o descarga completa"):
        st.caption("Si es la primera sincronización o quieres re-descargar todo desde Drive:")
        if st.button("Descargar TODA la carpeta de Drive", use_container_width=True,
                     disabled=not bool(api_key), key="btn_drive_full"):
            with st.spinner("Descargando carpeta completa desde Drive..."):
                sb2 = st.empty()
                try:
                    names = download_full_folder_gdown(
                        drive_url_input, DOCS_FOLDER, progress_callback=lambda t: sb2.text(t)
                    )
                    if names:
                        sb2.text(f"Indexando {len(names)} archivos...")
                        res = build_vector_store(api_key=api_key, progress_callback=lambda t: sb2.text(t))
                        st.success(f"{res['total_files']} docs indexados ({res['total_chunks']} secciones)")
                        st.session_state.pop("drive_preview", None)
                        st.rerun()
                    else:
                        st.warning("No se descargaron archivos. Verifica que la carpeta sea pública.")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown("---")

    # Doc list with delete
    if os.path.exists(DOCS_FOLDER):
        all_doc_paths = get_all_supported_files(DOCS_FOLDER)
        with st.expander(f"Documentos locales ({len(all_doc_paths)})"):
            for i_d, p in enumerate(all_doc_paths):
                d = os.path.basename(p)
                sub_rel = os.path.relpath(p, DOCS_FOLDER).replace("\\", "/")
                c1, c2 = st.columns([4, 1.2])
                with c1:
                    st.markdown(f"{SVG['file']}<small style='word-break:break-all'>{sub_rel}</small>", unsafe_allow_html=True)
                with c2:
                    if st.button("✕", key=f"del_{i_d}", help=f"Quitar {d}"):
                        n_rm = remove_document_from_store(d)
                        st.toast(f"'{d}' eliminado ({n_rm} secciones)")
                        st.rerun()

        if os.path.exists(ARCHIVE_FOLDER):
            arch = [f for f in os.listdir(ARCHIVE_FOLDER) if is_supported_file(f)]
            if arch and st.button(f"Restaurar eliminados ({len(arch)})", use_container_width=True):
                restored = restore_archived_documents()
                if api_key:
                    with st.spinner("Re-indexando..."):
                        for r in restored:
                            index_single_pdf(api_key, os.path.join(DOCS_FOLDER, r))
                st.rerun()

    # Re-indexar (opción manual)
    if st.button("Actualizar / Re-indexar todo", use_container_width=True, disabled=not bool(api_key)):
        with st.spinner("Indexando documentos..."):
            sb = st.empty()
            try:
                res = build_vector_store(api_key=api_key, progress_callback=lambda t: sb.text(t))
                st.success(f"{res['total_files']} docs · {res['total_chunks']} secciones")
                st.rerun()
            except Exception as e:
                st.error(str(e))

    # Quick cases
    st.markdown("---")
    st.markdown("**Casos frecuentes**")
    quick_cases = [
        "¿Cuándo puedo pedir reconsideración de nota y a quién?",
        "¿Cuáles son los requisitos para obtener el Grado de Bachiller?",
        "¿Cómo se tramita la convalidación de un curso?",
        "¿Qué pasa si el alumno supera el límite de inasistencias?",
        "¿Cuáles son los requisitos para postular a beca en el extranjero?",
    ]
    for i_q, qc in enumerate(quick_cases):
        if st.button(qc, key=f"qc_{i_q}", use_container_width=True):
            st.session_state.quick_q = qc

    if st.button("Limpiar sesión", use_container_width=True):
        st.session_state.history = []
        st.session_state.last_result = None
        st.session_state.last_sources = []
        st.rerun()

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;gap:12px;margin-bottom:4px">
  <svg xmlns="http://www.w3.org/2000/svg" width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="#002855" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 10v6M2 10l10-5 10 5-10 5z"/><path d="M6 12v5c3 3 9 3 12 0v-5"/></svg>
  <div>
    <div style="font-size:1.45rem;font-weight:800;color:#002855;line-height:1.2">Asistente de Coordinación Académica</div>
    <div style="font-size:0.88rem;color:#64748B">Universidad ESAN · Soporte normativo para Coordinadoras</div>
  </div>
</div>
<hr style="border:none;border-top:2px solid #E2E8F0;margin:10px 0 18px 0">
""", unsafe_allow_html=True)

# ── 3-COLUMN LAYOUT ───────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1.15, 1.35, 1.2], gap="medium")

# ═══════════════════════════════════════════════════════════ COLUMNA 01 ═══════
with col1:
    st.markdown('<div class="col-number">01</div><div class="col-title">Consulta recibida</div>', unsafe_allow_html=True)

    # Selector de modelo
    model_label = st.radio(
        "Modelo de IA",
        options=list(MODELS.keys()),
        index=0,
        help="GPT-4o-mini: rápido y económico. MiMo v2.6 Pro: razonamiento profundo (requiere OpenRouter Key).",
    )
    model_cfg = MODELS[model_label]
    cost_tag = "~$0.0005/consulta" if model_cfg["provider"] == "openai" else "~$0.00084/consulta"
    st.caption(f"{'⚡ OpenAI' if model_cfg['provider']=='openai' else '🧠 OpenRouter'} · {cost_tag}")

    st.markdown("---")

    # Mensaje del alumno
    default_q = st.session_state.quick_q or ""
    student_msg = st.text_area(
        "Mensaje del alumno",
        value=default_q,
        height=130,
        placeholder="Escribe o pega aquí el mensaje exacto del alumno...",
        max_chars=4000,
    )
    st.caption(f"{len(student_msg)}/4000 caracteres")
    if st.session_state.quick_q:
        st.session_state.quick_q = None

    extra_ctx = st.text_area(
        "Contexto adicional (opcional)",
        height=75,
        placeholder="Curso, ciclo, fecha de evaluación, antecedentes conocidos del alumno...",
        max_chars=2000,
    )

    # Diagnostics: show exactly why the button might be disabled
    can_resolve = bool(api_key and student_msg.strip())
    if not api_key:
        st.warning("Ingresa tu OpenAI API Key en la barra lateral.", icon="🔑")
    elif not student_msg.strip():
        st.info("Escribe el mensaje del alumno para habilitar la consulta.", icon="✏️")

    resolve_btn = st.button(
        "Resolver consulta",
        use_container_width=True,
        disabled=not can_resolve,
        type="primary",
    )

    # History mini-log
    if st.session_state.history:
        st.markdown("---")
        st.caption("Historial de esta sesión")
        for h in reversed(st.session_state.history[-5:]):
            st.caption(f"› {h[:80]}...")

# ═══════════════════════════════════════════════════════════ PROCESS ══════════
if resolve_btn and api_key and student_msg.strip():
    # Spinner visible en columna 2 para que el usuario vea el progreso
    with col2:
        with st.spinner("Analizando normativa y generando respuesta..."):
            try:
                # Si la BD no está indexada, indexar automáticamente primero
                if not is_indexed and api_key:
                    build_vector_store(api_key=api_key)
                    rag = RAGEngine(api_key=api_key, openrouter_key=or_key)

                result, sources = rag.answer_question(
                    query=student_msg.strip(),
                    context_extra=extra_ctx.strip(),
                    model_label=model_label,
                    chat_history=st.session_state.history,
                )
                st.session_state.last_result = result
                st.session_state.last_sources = sources
                st.session_state.history.append(student_msg.strip()[:120])
            except Exception as e:
                st.session_state.last_result = {
                    "requiere_aprobacion": "Error al procesar",
                    "borrador_alumno": f"Error: {e}",
                    "acciones_coordinadora": [],
                    "razonamiento": str(e),
                    "citas_normativas": [],
                }
                st.session_state.last_sources = []
    st.rerun()

# ═══════════════════════════════════════════════════════════ COLUMNA 02 ═══════
with col2:
    st.markdown('<div class="col-number">02</div><div class="col-title">Respuesta propuesta</div>', unsafe_allow_html=True)

    result = st.session_state.last_result

    if result is None:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:#94A3B8">
          <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#CBD5E1" stroke-width="1.5"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
          <div style="margin-top:12px;font-size:0.92rem">Ingresa una consulta en la columna izquierda y haz clic en <b>Resolver consulta</b></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Tarjeta de aprobación
        appr = result.get("requiere_aprobacion", "")
        st.markdown(f"""
        <div class="appr-badge">
          <div class="appr-icon">👤</div>
          <div>
            <div style="font-weight:700;font-size:0.88rem;color:#78350F">Requiere aprobación / validación de:</div>
            <div style="font-size:0.86rem;color:#451A03;margin-top:2px">{appr}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Borrador
        draft = result.get("borrador_alumno", "")
        st.markdown("**Borrador para el alumno**")
        st.markdown(f'<div class="draft-box">{draft}</div>', unsafe_allow_html=True)

        # Botones de copia
        b1, b2 = st.columns(2)
        wa_text = f"📱 *Coordinación Académica ESAN*\n\n{draft}"
        mail_text = f"Estimado alumno,\n\n{draft}"

        with b1:
            if st.button(f"Copiar para WhatsApp", key="cp_wa", use_container_width=True):
                st.code(wa_text, language=None)
                st.toast("Texto mostrado — cópialo desde el bloque de arriba")

        with b2:
            if st.button("Copiar para correo", key="cp_mail", use_container_width=True):
                st.code(mail_text, language=None)
                st.toast("Texto mostrado — cópialo desde el bloque de arriba")

        st.markdown("---")

        # Acciones para la coordinadora
        actions = result.get("acciones_coordinadora", [])
        if actions:
            st.markdown("**Acciones para ti**")
            for act in actions:
                st.checkbox(act, value=False, key=f"act_{hash(act)}")

        # Razonamiento
        reason = result.get("razonamiento", "")
        if reason:
            with st.expander("Razonamiento del asistente"):
                st.markdown(f'<div class="reason-box">{reason}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════ COLUMNA 03 ═══════
with col3:
    st.markdown('<div class="col-number">03</div><div class="col-title">Respaldo normativo y documental</div>', unsafe_allow_html=True)

    result = st.session_state.last_result
    sources = st.session_state.last_sources

    if result is None or not sources:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:#94A3B8">
          <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#CBD5E1" stroke-width="1.5"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>
          <div style="margin-top:12px;font-size:0.92rem">Las citas normativas aparecerán aquí</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        citas = result.get("citas_normativas", [])
        # Mezclar citas del LLM con los sources recuperados para máxima cobertura
        displayed = {}
        for c in citas:
            num = c.get("numero", 0)
            displayed[num] = c

        # Si no hay citas parseadas del LLM, mostrar los sources directos
        if not displayed:
            for i_s, s in enumerate(sources[:6], 1):
                displayed[i_s] = {
                    "numero": i_s,
                    "documento": s["source"],
                    "seccion": s["section"],
                    "pagina": str(s["page"]),
                    "extracto": s["text"][:200],
                }

        for num, c in sorted(displayed.items()):
            doc_name = c.get("documento", "")
            seccion = c.get("seccion", "")
            pagina = c.get("pagina", "")
            extracto = c.get("extracto", "")

            st.markdown(f"""
            <div class="cite-card">
              <div style="display:flex;align-items:flex-start;gap:8px;margin-bottom:6px">
                <span class="cite-num">{num}</span>
                <div>
                  <div class="cite-source">{doc_name}</div>
                  <div class="cite-section">{seccion}{f' · Pág. {pagina}' if pagina else ''}</div>
                </div>
              </div>
              {f'<div class="cite-excerpt">"{extracto}..."</div>' if extracto else ''}
              <a href="{GOOGLE_DRIVE_FOLDER}" target="_blank" class="drive-link">
                {SVG["drive"]} Abrir en Google Drive
              </a>
            </div>
            """, unsafe_allow_html=True)

        # Nota de fuente
        st.markdown(f"""
        <div style="margin-top:14px;padding:8px 12px;background:#F8FAFC;border-radius:6px;font-size:0.78rem;color:#64748B">
          {SVG["drive"]} <a href="{GOOGLE_DRIVE_FOLDER}" target="_blank" style="color:#1967D2;font-weight:600">Ver todos los reglamentos en Google Drive</a>
        </div>
        """, unsafe_allow_html=True)
