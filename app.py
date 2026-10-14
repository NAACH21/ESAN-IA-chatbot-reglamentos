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
api_key = os.getenv("OPENAI_API_KEY", "")
try:
    api_key = api_key or st.secrets.get("OPENAI_API_KEY", "")
except Exception:
    pass

or_key = os.getenv("OPENROUTER_API_KEY", "")
try:
    or_key = or_key or st.secrets.get("OPENROUTER_API_KEY", "")
except Exception:
    pass

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Asistente de Coordinación · Universidad ESAN",
    page_icon="https://cdn.worldvectorlogo.com/logos/universidad-esan.svg",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── SVG ICONS EN ROJO ESAN Y NEUTROS ─────────────────────────────────────────
SVG = {
    "brain": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#D31418" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:6px"><path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 4.44-2.04z"/><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-4.44-2.04z"/></svg>""",
    "db": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#D31418" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:6px"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>""",
    "file": """<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#D31418" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:4px"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>""",
    "check": """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#16a34a" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:4px"><polyline points="20 6 9 17 4 12"/></svg>""",
    "drive": """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#1967D2" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:4px"><path d="M21 15a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V9a2 2 0 0 1 2-2h1l2-3h6l2 3h1a2 2 0 0 1 2 2z"/></svg>""",
    "sparkles": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#D31418" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-3px;margin-right:6px"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/></svg>""",
    "send": """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-left:6px"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>""",
}

# ── STYLING INSTITUCIONAL ESAN (ROJO INSTITUCIONAL #D31418) ──────────────────
st.markdown("""
<style>
/* Tipografía y fondo principal */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Header Institucional */
.esan-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 18px;
    background: #FFFFFF;
    border: 1px solid #F1F5F9;
    border-left: 5px solid #D31418;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.02);
}
.esan-title {
    font-size: 1.35rem;
    font-weight: 800;
    color: #0F172A;
    line-height: 1.2;
}
.esan-title span {
    color: #D31418;
}
.esan-subtitle {
    font-size: 0.85rem;
    color: #64748B;
    margin-top: 2px;
}

/* Tarjetas y Contenedores */
.esan-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    margin-bottom: 16px;
}
.esan-card-title {
    font-size: 1rem;
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
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 14px;
}
.appr-icon {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    background: #FEF3C7;
    border: 1px solid #F59E0B;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
}

/* Borrador del Alumno */
.draft-box {
    background: #FAFAFA;
    border: 1px solid #E2E8F0;
    border-left: 4px solid #D31418;
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
    border-left: 4px solid #D31418;
    border-radius: 8px;
    padding: 12px 14px;
    margin-bottom: 12px;
    transition: all 0.2s ease;
    box-shadow: 0 1px 4px rgba(0,0,0,0.02);
}
.cite-card:hover {
    border-color: #FECACA;
    border-left-color: #D31418;
    box-shadow: 0 4px 12px rgba(211,20,24,0.06);
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
    font-size: 0.82rem;
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
    padding: 30px 20px 20px 20px;
}
.welcome-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 14px;
    margin-top: 20px;
}

/* Bottom Chat Dock (Estilo ChatGPT / Gemini) */
.chat-dock {
    background: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 16px;
    padding: 14px 18px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
    margin-top: 20px;
}

/* Botón primario en Rojo ESAN */
button[kind="primary"] {
    background-color: #D31418 !important;
    border-color: #D31418 !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    transition: all 0.2s ease !important;
}
button[kind="primary"]:hover {
    background-color: #B20D10 !important;
    border-color: #B20D10 !important;
    box-shadow: 0 4px 12px rgba(211,20,24,0.25) !important;
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
    border-radius: 20px;
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
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 700;
}
.sb-section {
    font-size: 0.86rem;
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
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── SIDEBAR (ELECCIÓN DE MODELOS, BASE DOCUMENTAL, SYNC DRIVE) ─────────────────
with st.sidebar:
    # Logo oficial actual de ESAN
    if os.path.exists("assets/logo_esan.svg"):
        st.image("assets/logo_esan.svg", width=190)
    else:
        st.markdown('<h2 style="color:#D31418;margin:0">UNIVERSIDAD ESAN</h2>', unsafe_allow_html=True)

    st.caption("Coordinación Académica · Soporte Normativo")

    # Indicador silencioso de credenciales conectadas
    if api_key:
        st.markdown('<div class="status-pill-ok">● Motor IA activo (.env)</div>', unsafe_allow_html=True)
    else:
        st.warning("Configura tu OPENAI_API_KEY en el archivo .env", icon="🔑")

    st.markdown("---")

    # 1. SELECCIÓN DE MODELO (Movido aquí a la izquierda como pidió el usuario)
    st.markdown(f'<div class="sb-section">{SVG["brain"]} Modelo de Inteligencia Artificial</div>', unsafe_allow_html=True)
    model_label = st.radio(
        "Seleccionar Modelo",
        options=list(MODELS.keys()),
        index=0,
        label_visibility="collapsed",
        help="Elige el modelo según la complejidad de la consulta normativa."
    )
    model_cfg = MODELS[model_label]
    if model_cfg["provider"] == "openai":
        st.markdown('<div class="status-pill-model">⚡ OpenAI · GPT-4o-mini (Rápido y económico)</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-pill-model">🧠 OpenRouter · MiMo v2.6 Pro (Razonamiento profundo)</div>', unsafe_allow_html=True)

    st.markdown("---")

    # 2. BASE DOCUMENTAL
    st.markdown(f'<div class="sb-section">{SVG["db"]} Base Documental</div>', unsafe_allow_html=True)
    rag = RAGEngine(api_key=api_key, openrouter_key=or_key) if api_key else None
    is_indexed = rag.is_indexed() if rag else False

    if is_indexed:
        cnt = rag.get_document_count()
        st.caption(f"📚 **{cnt:,}** secciones vectorizadas en ChromaDB")
    else:
        st.caption("⚠️ Base documental sin inicializar")

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
                    if st.button("✕", key=f"del_{i_d}", help=f"Quitar {d}"):
                        n_rm = remove_document_from_store(d)
                        st.toast(f"'{d}' eliminado ({n_rm} secciones)")
                        st.rerun()

            if st.button("Indexar documentos nuevos locales", use_container_width=True, disabled=not bool(api_key)):
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

# ── TOP HEADER INSTITUCIONAL ──────────────────────────────────────────────────
st.markdown(f"""
<div class="esan-header">
  <div style="display:flex;align-items:center;gap:12px">
    <div style="width:40px;height:40px;border-radius:8px;background:#D31418;display:flex;align-items:center;justify-content:center;color:#fff;font-weight:900;font-size:1.1rem">
      E
    </div>
    <div>
      <div class="esan-title">Asistente Normativo de Coordinación · <span>ESAN</span></div>
      <div class="esan-subtitle">Respuestas institucionales oficiales con citas y respaldo reglamentario verificado</div>
    </div>
  </div>
  <div>
    <div class="status-pill-model">
      {model_label.split('(')[0].strip()}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── ÁREA PRINCIPAL: RESULTADOS O PANTALLA DE BIENVENIDA ───────────────────────
result = st.session_state.last_result
sources = st.session_state.last_sources

if result is None:
    # Pantalla de Bienvenida con tarjetas sugeridas estilo Gemini/GPT
    st.markdown("""
    <div class="hero-box">
      <div style="font-size:1.6rem;font-weight:800;color:#0F172A;margin-bottom:6px">
        ¿En qué consulta de coordinación académica te puedo ayudar hoy?
      </div>
      <div style="font-size:0.92rem;color:#64748B;max-width:680px;margin:0 auto">
        Escribe la consulta del estudiante en la barra inferior o haz clic en alguno de los casos frecuentes para redactar una respuesta normativa inmediata.
      </div>
    </div>
    """, unsafe_allow_html=True)

    c_c1, c_c2 = st.columns(2)
    with c_c1:
        if st.button("📝 Reconsideración de Nota\n\n¿Cuándo y ante quién puede solicitar reconsideración de evaluación?", use_container_width=True):
            st.session_state.quick_q = "¿Cuándo puedo pedir reconsideración de nota y ante qué autoridad corresponde presentarla?"
            st.rerun()

        if st.button("🤖 Uso de Inteligencia Artificial\n\n¿Cuáles son los lineamientos para docentes y alumnos sobre uso de IA?", use_container_width=True):
            st.session_state.quick_q = "¿Cuáles son los lineamientos sobre el uso de Inteligencia Artificial para docentes y alumnos en ESAN?"
            st.rerun()

    with c_c2:
        if st.button("🎓 Requisitos de Grado de Bachiller\n\n¿Cuáles son los requisitos reglamentarios para obtener el grado?", use_container_width=True):
            st.session_state.quick_q = "¿Cuáles son los requisitos y créditos necesarios para obtener el Grado de Bachiller?"
            st.rerun()

        if st.button("💰 Devolución de Dinero y Retiro\n\n¿Cuál es el procedimiento y formato para devoluciones o retiro de programa?", use_container_width=True):
            st.session_state.quick_q = "¿Cuál es el procedimiento formal y plazos para solicitar devolución de dinero o retiro de programa?"
            st.rerun()

else:
    # ── VISTA DE RESULTADOS (2 COLUMNAS: RESPUESTA + RESPALDO NORMATIVO) ───────
    col_resp, col_norm = st.columns([1.35, 1.15], gap="large")

    # ═════════════════════════════════════════ COLUMNA 1: RESPUESTA PROPUESTA ═
    with col_resp:
        st.markdown(f'<div class="esan-card-title">{SVG["sparkles"]} Respuesta propuesta para Coordinación</div>', unsafe_allow_html=True)

        # 1. Badge de Aprobación
        appr = result.get("requiere_aprobacion", "La coordinadora puede resolver directamente")
        st.markdown(f"""
        <div class="appr-badge">
          <div class="appr-icon">👤</div>
          <div>
            <div style="font-weight:700;font-size:0.86rem;color:#78350F">Requiere aprobación / validación de:</div>
            <div style="font-size:0.9rem;color:#92400E;font-weight:600;margin-top:2px">{appr}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # 2. Borrador para el Alumno
        draft = result.get("borrador_alumno", "")
        st.markdown("<b style='font-size:0.88rem;color:#0F172A'>Borrador de respuesta para el estudiante</b>", unsafe_allow_html=True)
        st.markdown(f'<div class="draft-box">{draft}</div>', unsafe_allow_html=True)

        # 3. Botones de Copia Rápidos
        b_c1, b_c2 = st.columns(2)
        wa_text = f"📱 *Coordinación Académica ESAN*\n\n{draft}"
        mail_text = f"Estimado/a estudiante,\n\n{draft}"

        with b_c1:
            if st.button("Copiar para WhatsApp", key="btn_cp_wa", use_container_width=True):
                st.code(wa_text, language=None)
                st.toast("Texto preparado para WhatsApp — cópialo del recuadro")
        with b_c2:
            if st.button("Copiar para Correo", key="btn_cp_mail", use_container_width=True):
                st.code(mail_text, language=None)
                st.toast("Texto preparado para Correo — cópialo del recuadro")

        # 4. Acciones para la Coordinadora
        actions = result.get("acciones_coordinadora", [])
        if actions:
            st.markdown("<b style='font-size:0.88rem;color:#0F172A;margin-top:14px;display:block'>Acciones a verificar por la Coordinadora</b>", unsafe_allow_html=True)
            for act in actions:
                st.checkbox(act, value=False, key=f"chk_{hash(act)}")

        # 5. Razonamiento
        reason = result.get("razonamiento", "")
        if reason:
            with st.expander("Ver razonamiento normativo del asistente"):
                st.markdown(f'<div class="reason-box">{reason}</div>', unsafe_allow_html=True)

    # ═════════════════════════════════════════ COLUMNA 2: RESPALDO NORMATIVO ═══
    with col_norm:
        st.markdown(f'<div class="esan-card-title">{SVG["db"]} Respaldo Normativo y Citas ({len(sources)} fuentes)</div>', unsafe_allow_html=True)

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
            badge_page = f'<span style="background:#FEF2F2;color:#991B1B;padding:2px 8px;border-radius:12px;font-size:0.73rem;font-weight:700;border:1px solid #FECACA;margin-left:6px">Pág. {pagina}</span>' if pagina else ""

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
                {f'<span style="font-size:0.74rem;color:#64748B">📍 Ubicación: <b>Pág. {pagina}</b></span>' if pagina else ''}
              </div>
            </div>
            """, unsafe_allow_html=True)

# ── BOTTOM CHAT DOCK (ESTILO CHATGPT / GEMINI OCUPANDO CASI TODO) ──────────────
st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

with st.container():
    # Prompt dock container
    st.markdown("""
    <div style="border-top:2px solid #F1F5F9;padding-top:16px;margin-top:16px">
      <div style="display:flex;align-items:center;gap:6px;font-size:0.9rem;font-weight:700;color:#0F172A;margin-bottom:6px">
        Consulta del Estudiante
      </div>
    </div>
    """, unsafe_allow_html=True)

    default_prompt = st.session_state.quick_q or ""
    c_input, c_btn = st.columns([5.5, 1.5], gap="small")

    with c_input:
        student_msg = st.text_area(
            "Consulta del estudiante",
            value=default_prompt,
            height=85,
            placeholder="Escribe o pega aquí el mensaje del alumno (ej: 'Profesor, desaprobé la evaluación final y deseo solicitar reconsideración...')...",
            label_visibility="collapsed",
            max_chars=4000,
        )

    with c_btn:
        st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)
        can_run = bool(api_key and student_msg.strip())
        run_query = st.button("Resolver con IA →", use_container_width=True, type="primary", disabled=not can_run)
        if st.session_state.history:
            st.caption(f"Historial: {len(st.session_state.history)} consultas")

    with st.expander("➕ Contexto adicional opcional (curso, ciclo, antecedentes del alumno...)"):
        extra_ctx = st.text_input(
            "Contexto extra",
            placeholder="Ej: Curso Gestión de Operaciones, alumno del 8vo ciclo DPA, no asistió al examen extemporáneo...",
            label_visibility="collapsed"
        )

    # Limpiar trigger rápido si se usó
    if st.session_state.quick_q:
        st.session_state.quick_q = None

# ── PROCESAMIENTO DE CONSULTA ─────────────────────────────────────────────────
if run_query and api_key and student_msg.strip():
    with st.spinner("Buscando artículos en reglamentos y redactando dictamen con IA..."):
        try:
            if not is_indexed and api_key:
                build_vector_store(api_key=api_key)
                rag = RAGEngine(api_key=api_key, openrouter_key=or_key)

            result_data, sources_data = rag.answer_question(
                query=student_msg.strip(),
                context_extra=extra_ctx.strip() if 'extra_ctx' in locals() and extra_ctx else "",
                model_label=model_label,
                chat_history=st.session_state.history,
            )
            st.session_state.last_result = result_data
            st.session_state.last_sources = sources_data
            st.session_state.history.append(student_msg.strip()[:120])
            st.rerun()
        except Exception as e:
            st.error(f"Error al procesar la consulta: {e}")
