import os
import streamlit as st
from dotenv import load_dotenv
from ingest import build_vector_store, DOCS_FOLDER
from rag_engine import RAGEngine

load_dotenv()

# Configuración de página
st.set_page_config(
    page_title="Asistente Normativo - Universidad ESAN",
    page_icon="https://www.ue.edu.pe/favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Iconos SVG profesionales (Feather / Lucide Style)
SVG_GRAD_CAP = """<svg xmlns="http://www.w3.org/2000/svg" width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#002855" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: -5px; margin-right: 8px;"><path d="M22 10v6M2 10l10-5 10 5-10 5z"/><path d="M6 12v5c3 3 9 3 12 0v-5"/></svg>"""

SVG_SETTINGS = """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#002855" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: -2px; margin-right: 6px;"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>"""

SVG_DATABASE = """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#002855" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: -2px; margin-right: 6px;"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>"""

SVG_QUERIES = """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#002855" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: -2px; margin-right: 6px;"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>"""

SVG_FILE = """<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#002855" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: -2px; margin-right: 5px;"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>"""

SVG_CHECK = """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#16a34a" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: -2px; margin-right: 5px;"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>"""

SVG_ALERT = """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#d97706" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: -2px; margin-right: 5px;"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>"""

SVG_BOOK = """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: -2px; margin-right: 5px;"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>"""

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #002855;
        margin-bottom: 0.2rem;
        display: flex;
        align-items: center;
    }
    .sub-header {
        font-size: 1rem;
        color: #4A5568;
        margin-bottom: 1.5rem;
    }
    .section-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #002855;
        margin-top: 1.2rem;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
    }
    .source-box {
        background-color: #F8FAFC;
        border-left: 3px solid #002855;
        padding: 10px 14px;
        margin-top: 8px;
        border-radius: 4px;
        font-size: 0.88rem;
    }
    .status-badge {
        padding: 6px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
    }
    .status-success {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        color: #166534;
    }
    .status-warning {
        background-color: #FFFBEB;
        border: 1px solid #FDE68A;
        color: #92400E;
    }
    .status-info {
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        color: #1E40AF;
    }
</style>
""", unsafe_allow_html=True)

# Inicializar sesión
if "messages" not in st.session_state:
    st.session_state.messages = []
if "quick_prompt" not in st.session_state:
    st.session_state.quick_prompt = None

# --- SIDEBAR: Configuración y Estado ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Logo_Universidad_ESAN.svg/320px-Logo_Universidad_ESAN.svg.png", width=180)
    
    st.markdown(f'<div class="section-title">{SVG_SETTINGS} Configuración</div>', unsafe_allow_html=True)
    
    # Manejo de API Key
    env_key = os.getenv("OPENAI_API_KEY", "")
    api_key_input = st.text_input(
        "OpenAI API Key:",
        value=env_key,
        type="password",
        help="Clave de OpenAI (ej. sk-proj-...)"
    )
    api_key = api_key_input.strip() if api_key_input else env_key
    
    if api_key:
        st.markdown(f'<div class="status-badge status-success">{SVG_CHECK} Clave API vinculada</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="status-badge status-warning">{SVG_ALERT} Ingrese su API Key para habilitar consultas</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f'<div class="section-title">{SVG_DATABASE} Base de Conocimiento</div>', unsafe_allow_html=True)
    
    rag = RAGEngine(api_key=api_key) if api_key else None
    is_indexed = rag.is_indexed() if rag else False
    
    if is_indexed:
        count = rag.get_document_count()
        st.markdown(f'<div class="status-badge status-info">{SVG_BOOK} <b>{count}</b> fragmentos indexados</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="status-badge status-warning">{SVG_ALERT} Base documental no indexada</div>', unsafe_allow_html=True)
        
    if st.button("Indexar / Re-indexar Documentos", use_container_width=True, disabled=not bool(api_key)):
        if not api_key:
            st.error("Por favor, ingrese su API Key primero.")
        else:
            with st.spinner("Procesando documentos y calculando vectores..."):
                status_box = st.empty()
                def update_status(text):
                    status_box.text(text)
                try:
                    res = build_vector_store(api_key=api_key, progress_callback=update_status)
                    st.success(f"Indexación completa: {res['total_files']} reglamentos ({res['total_chunks']} fragmentos).")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error durante la indexación: {e}")

    # Mostrar lista de documentos disponibles
    if os.path.exists(DOCS_FOLDER):
        docs = [f for f in os.listdir(DOCS_FOLDER) if f.lower().endswith(".pdf") and "(1)" not in f]
        with st.expander(f"Documentos disponibles ({len(docs)})"):
            for d in docs:
                st.markdown(f"{SVG_FILE} <small>{d}</small>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f'<div class="section-title">{SVG_QUERIES} Consultas de Prueba</div>', unsafe_allow_html=True)
    quick_questions = [
        "¿Cuáles son los requisitos para obtener el grado de Bachiller?",
        "¿Cómo es el proceso y requisitos para convalidar asignaturas?",
        "¿Qué sanciones se aplican por cometer plagio en evaluaciones?",
        "¿En qué condiciones un estudiante entra en bajo rendimiento académico?",
        "¿Cuáles son los requisitos para postular a una beca en el extranjero?"
    ]
    
    for idx, q in enumerate(quick_questions):
        if st.button(q, key=f"quick_btn_{idx}", use_container_width=True):
            st.session_state.quick_prompt = q

    if st.button("Reiniciar conversación", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- ÁREA PRINCIPAL ---
st.markdown(f'<div class="main-header">{SVG_GRAD_CAP} Asistente Normativo - Universidad ESAN</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Orientación y consulta de normativa de pregrado, admisión, convalidaciones, grados y conducta con citas oficiales.</div>', unsafe_allow_html=True)

# Renderizar historial de mensajes
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander(f"Fuentes oficiales consultadas ({len(msg['sources'])})"):
                for s in msg["sources"]:
                    st.markdown(
                        f"""<div class="source-box">
                        <b>{SVG_FILE} {s['source']}</b> — <i>Página {s['page']}</i><br>
                        <small>{s['text'][:350]}...</small>
                        </div>""", 
                        unsafe_allow_html=True
                    )

# Campo de entrada de texto siempre presente en la parte inferior
chat_input_text = st.chat_input("Escribe tu consulta sobre los reglamentos de ESAN...")

user_prompt = None
if st.session_state.quick_prompt:
    user_prompt = st.session_state.quick_prompt
    st.session_state.quick_prompt = None
elif chat_input_text:
    user_prompt = chat_input_text

if user_prompt:
    # Validar API key y base indexada
    if not api_key:
        st.error("Por favor, ingrese su OpenAI API Key en la barra lateral izquierda.")
    elif not is_indexed:
        st.warning("La base de datos aún no ha sido indexada. Haga clic en 'Indexar / Re-indexar Documentos' en la barra lateral.")
    else:
        # 1. Registrar y mostrar pregunta del usuario
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # 2. Generar respuesta
        with st.chat_message("assistant"):
            with st.spinner("Analizando reglamentos oficiales de ESAN..."):
                try:
                    answer, sources = rag.answer_question(
                        query=user_prompt,
                        chat_history=st.session_state.messages[:-1]
                    )
                    st.markdown(answer)
                    
                    if sources:
                        with st.expander(f"Fuentes oficiales consultadas ({len(sources)})"):
                            for s in sources:
                                st.markdown(
                                    f"""<div class="source-box">
                                    <b>{SVG_FILE} {s['source']}</b> — <i>Página {s['page']}</i><br>
                                    <small>{s['text'][:350]}...</small>
                                    </div>""", 
                                    unsafe_allow_html=True
                                )
                    
                    # Guardar en sesión
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                except Exception as e:
                    err_msg = f"Ocurrió un error al procesar la consulta: {e}"
                    st.error(err_msg)
