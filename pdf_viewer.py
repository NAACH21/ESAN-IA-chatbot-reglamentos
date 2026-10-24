"""
pdf_viewer.py
Visor institucional seguro con PDF.js y Mark.js.
Permite visualizar documentos PDF locales de DOCS/ directamente en la aplicación,
navegar a la página exacta y resaltar en amarillo la frase o artículo citado.
100% privado y seguro (sin exponer documentos en internet).
"""
import os
import re
import json
import base64
import unicodedata
from typing import Any, Tuple

DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DOCS")

_STEMS_MAP = {
    "reglamento": "reglam", "reglamentos": "reglam", "reglam": "reglam",
    "general": "gral", "gral": "gral",
    "diploma": "diplom", "diplomas": "diplom", "diplomado": "diplom", "diplomados": "diplom", "diplom": "diplom",
    "certificado": "certif", "certificados": "certif", "certif": "certif",
    "maestria": "maestria", "maestrias": "maestria",
    "doctorado": "doctoral", "doctorados": "doctoral", "doctorales": "doctoral", "doctoral": "doctoral",
    "educacion": "educacion", "ejecutiva": "ejecutiva", "ejecutivo": "ejecutiva",
    "postgrado": "posgrado", "posgrado": "posgrado",
    "examen": "examen", "examenes": "examen", "extemporaneo": "extemporaneo",
    "moodle": "moodle", "zoom": "zoom", "plataforma": "plataforma",
    "lineamiento": "lineamiento", "lineamientos": "lineamiento",
}

def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")

def _tokenize(text: str):
    text = _strip_accents(text.lower())
    text = re.sub(r"\.(pdf|docx|xlsx)$", "", text)
    words = re.findall(r"[a-z0-9]+", text)
    return {_STEMS_MAP.get(w, w) for w in words if len(_STEMS_MAP.get(w, w)) >= 3}

def _walk_docs_files():
    """
    Recorre DOCS/ de forma recursiva (nivel raíz + subdirectorios) y devuelve
    lista de tuplas (filename_basename, full_path) para solo archivos soportados.
    """
    supported = {".pdf", ".docx", ".xlsx"}
    result = []
    if not os.path.exists(DOCS_DIR):
        return result
    for entry in os.scandir(DOCS_DIR):
        ext = os.path.splitext(entry.name)[1].lower()
        if entry.is_file() and ext in supported:
            result.append((entry.name, entry.path))
        elif entry.is_dir():
            for sub in os.scandir(entry.path):
                sub_ext = os.path.splitext(sub.name)[1].lower()
                if sub.is_file() and sub_ext in supported:
                    result.append((sub.name, sub.path))
    return result


def resolve_local_doc_path(doc_name: str, fallback_source: str = "") -> str:
    """
    Resuelve el nombre del documento indicado en la cita a la ruta del archivo físico en DOCS/.
    Busca recursivamente incluyendo subdirectorios de DOCS/.
    """
    all_entries = _walk_docs_files()  # [(basename, full_path), ...]

    # 1. Fallback exacto por basename
    if fallback_source:
        for bname, fpath in all_entries:
            if bname == fallback_source:
                return fpath

    # 2. Nombre exacto (normalizado de tildes)
    clean_query = _strip_accents(os.path.basename(doc_name).strip().lower())
    for bname, fpath in all_entries:
        if _strip_accents(bname.lower()) == clean_query:
            return fpath

    # 3. Subcadena directa (query dentro del nombre o viceversa)
    for bname, fpath in all_entries:
        f_clean = _strip_accents(bname.lower())
        if clean_query in f_clean or f_clean in clean_query:
            return fpath

    # 4. Fallback subcadena
    if fallback_source:
        fb_clean = _strip_accents(fallback_source.lower())
        for bname, fpath in all_entries:
            f_clean = _strip_accents(bname.lower())
            if fb_clean in f_clean or f_clean in fb_clean:
                return fpath

    # 5. Coincidencia difusa por tokens / raíces
    q_tokens = _tokenize(doc_name)
    best_path = None
    best_score = 0.0
    for bname, fpath in all_entries:
        f_tokens = _tokenize(bname)
        if not f_tokens or not q_tokens:
            continue
        overlap = len(q_tokens.intersection(f_tokens))
        score = overlap / (len(q_tokens) + 0.2 * len(f_tokens))
        if score > best_score:
            best_score = score
            best_path = fpath

    if best_path and best_score >= 0.25:
        return best_path

    # 6. Fallback: búsqueda difusa sobre fallback_source
    if fallback_source:
        fb_tokens = _tokenize(fallback_source)
        best_path_fb = None
        best_score_fb = 0.0
        for bname, fpath in all_entries:
            f_tokens = _tokenize(bname)
            if not f_tokens or not fb_tokens:
                continue
            overlap = len(fb_tokens.intersection(f_tokens))
            score = overlap / (len(fb_tokens) + 0.2 * len(f_tokens))
            if score > best_score_fb:
                best_score_fb = score
                best_path_fb = fpath
        if best_path_fb and best_score_fb >= 0.25:
            return best_path_fb

    return ""


def clean_search_term(raw_text: str) -> str:
    """Limpia el texto a buscar (elimina comillas, puntos suspensivos y espacios extra)."""
    if not raw_text:
        return ""
    t = raw_text.strip()
    t = t.strip(' "\'“”«»')
    t = re.sub(r'\.{2,}', '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    # Tomar una frase inicial significativa (máximo ~85 caracteres para búsqueda precisa sin cortes)
    if len(t) > 85:
        idx = t[:85].rfind(" ")
        if idx > 30:
            t = t[:idx]
        else:
            t = t[:85]
    return t


def parse_page_number(page_val: Any) -> int:
    """Extrae el número de página entero a partir de cualquier formato ('Pág. 5', '5', 5)."""
    if not page_val:
        return 1
    digits = re.findall(r"\d+", str(page_val))
    if digits:
        try:
            return max(1, int(digits[0]))
        except Exception:
            return 1
    return 1


def generate_pdf_viewer_html(
    pdf_b64: str,
    initial_page: int,
    search_text: str,
    doc_title: str,
    section_name: str = "",
) -> str:
    """
    Genera el HTML del visor institucional con PDF.js v3 y Mark.js.
    """
    clean_search = clean_search_term(search_text)
    clean_search_json = json.dumps(clean_search)
    doc_title_escaped = doc_title.replace('"', '&quot;').replace('<', '&lt;')
    section_escaped = section_name.replace('"', '&quot;').replace('<', '&lt;') if section_name else ""

    html = f'''<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf_viewer.min.css">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/mark.js/8.11.1/mark.min.js"></script>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; padding: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: #0F172A; color: #0F172A;
      overflow: hidden; height: 100vh; display: flex; flex-direction: column;
    }}
    .viewer-toolbar {{
      background: #FFFFFF; border-bottom: 2px solid #E2E8F0;
      padding: 10px 16px; display: flex; align-items: center; justify-content: space-between;
      flex-wrap: wrap; gap: 8px; z-index: 50; flex-shrink: 0;
    }}
    .doc-meta {{
      display: flex; align-items: center; gap: 8px; max-width: 55%; overflow: hidden;
    }}
    .doc-name {{
      font-size: 0.86rem; font-weight: 700; color: #0F172A; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }}
    .badge-sect {{
      background: #FEF2F2; color: #991B1B; border: 1px solid #FECACA;
      padding: 2px 8px; border-radius: 4px; font-size: 0.72rem; font-weight: 700; white-space: nowrap;
    }}
    .tools-group {{
      display: flex; align-items: center; gap: 6px;
    }}
    .tool-btn {{
      background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 6px;
      padding: 5px 11px; font-size: 0.78rem; font-weight: 600; cursor: pointer; color: #334155;
      display: inline-flex; align-items: center; gap: 4px; transition: all 0.15s ease;
    }}
    .tool-btn:hover {{ background: #F8FAFC; border-color: #94A3B8; color: #0F172A; }}
    .tool-btn:disabled {{ opacity: 0.35; cursor: not-allowed; }}
    .page-counter {{
      font-size: 0.8rem; font-weight: 700; color: #0F172A; margin: 0 4px; white-space: nowrap;
    }}
    .divider-v {{
      width: 1px; height: 18px; background: #E2E8F0; margin: 0 4px;
    }}
    .search-alert-bar {{
      background: #FFFBEB; border-bottom: 1px solid #FDE68A; padding: 7px 16px;
      font-size: 0.77rem; color: #92400E; display: flex; align-items: center; justify-content: space-between;
      flex-shrink: 0;
    }}
    .search-alert-text {{
      display: flex; align-items: center; gap: 6px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }}
    .viewer-body {{
      flex: 1; overflow: auto; padding: 20px 10px;
      display: flex; flex-direction: column; align-items: center; background: #334155;
    }}
    .page-wrapper {{
      position: relative; background: #FFFFFF;
      box-shadow: 0 8px 24px rgba(0,0,0,0.35); border-radius: 4px;
      margin-bottom: 24px;
    }}
    .textLayer {{
      opacity: 1 !important;
    }}
    /* Resaltado de frase continuo en amarillo con acento ESAN */
    mark.pdf-highlight {{
      background-color: #FEF08A !important;
      color: #0F172A !important;
      border-bottom: 2.5px solid #D31418 !important;
      border-radius: 2px !important;
      padding: 2px 1px !important;
      font-weight: 600 !important;
      box-shadow: 0 1px 3px rgba(0,0,0,0.12) !important;
    }}
    .loading-overlay {{
      color: #FFFFFF; font-size: 0.88rem; font-weight: 600;
      margin-top: 40px; display: flex; align-items: center; gap: 10px;
    }}
    .spinner-dot {{
      width: 18px; height: 18px; border: 2.5px solid #D31418;
      border-top-color: transparent; border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }}
    @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
  </style>
</head>
<body>

  <!-- BARRA DE HERRAMIENTAS INSTITUCIONAL -->
  <div class="viewer-toolbar">
    <div class="doc-meta">
      <span class="doc-name" title="{doc_title_escaped}">{doc_title_escaped}</span>
      {f'<span class="badge-sect">{section_escaped}</span>' if section_escaped else ''}
    </div>

    <div class="tools-group">
      <button class="tool-btn" id="prev-btn" onclick="changePage(-1)">◀ Anterior</button>
      <span class="page-counter">Pág. <span id="current-page">{initial_page}</span> de <span id="total-pages">-</span></span>
      <button class="tool-btn" id="next-btn" onclick="changePage(1)">Siguiente ▶</button>
      
      <span class="divider-v"></span>

      <button class="tool-btn" onclick="zoom(0.15)">＋ Zoom</button>
      <button class="tool-btn" onclick="zoom(-0.15)">－</button>
      <button class="tool-btn" onclick="resetZoom()">100%</button>
    </div>
  </div>

  <!-- AVISO DE TEXTO RESALTADO -->
  <div class="search-alert-bar" id="search-bar" style="display: none;">
    <div class="search-alert-text">
      <span>🎯 <b>Fragmento normativo buscado:</b> <i id="query-preview"></i></span>
    </div>
    <span style="font-weight:700; color:#B45309; font-size:0.73rem;" id="match-status">Buscando...</span>
  </div>

  <!-- CUERPO DEL VISOR -->
  <div class="viewer-body" id="viewer-container">
    <div class="loading-overlay" id="loading-box">
      <div class="spinner-dot"></div>
      <span>Cargando documento normativo oficial...</span>
    </div>

    <div class="page-wrapper" id="page-wrapper" style="display: none;">
      <canvas id="pdf-canvas"></canvas>
      <div id="text-layer" class="textLayer"></div>
    </div>
  </div>

  <script>
    pdfjsLib.GlobalWorkerOptions.workerSrc = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";

    const pdfB64Data = "{pdf_b64}";
    const targetPage = {initial_page};
    const targetQuery = {clean_search_json};

    if (targetQuery) {{
      document.getElementById("search-bar").style.display = "flex";
      document.getElementById("query-preview").textContent = targetQuery;
    }}

    let pdfDoc = null;
    let pageNum = targetPage;
    let scale = 1.35;
    let isRendering = false;

    // Decodificar Base64 seguro en memoria del cliente
    try {{
      const raw = atob(pdfB64Data);
      const uint8 = new Uint8Array(raw.length);
      for (let i = 0; i < raw.length; i++) {{
        uint8[i] = raw.charCodeAt(i);
      }}

      pdfjsLib.getDocument({{ data: uint8 }}).promise.then(doc => {{
        pdfDoc = doc;
        document.getElementById("total-pages").textContent = doc.numPages;
        document.getElementById("loading-box").style.display = "none";
        document.getElementById("page-wrapper").style.display = "block";
        if (pageNum > doc.numPages) pageNum = doc.numPages;
        if (pageNum < 1) pageNum = 1;
        renderPage(pageNum);
      }}).catch(err => {{
        document.getElementById("loading-box").innerHTML = "<span style='color:#FCA5A5'>Error al procesar el archivo PDF: " + err.message + "</span>";
      }});
    }} catch (e) {{
      document.getElementById("loading-box").innerHTML = "<span style='color:#FCA5A5'>Error al decodificar PDF: " + e.message + "</span>";
    }}

    async function renderPage(num) {{
      if (isRendering || !pdfDoc) return;
      isRendering = true;

      document.getElementById("current-page").textContent = num;
      document.getElementById("prev-btn").disabled = (num <= 1);
      document.getElementById("next-btn").disabled = (num >= pdfDoc.numPages);

      const page = await pdfDoc.getPage(num);
      const viewport = page.getViewport({{ scale: scale }});

      const canvas = document.getElementById("pdf-canvas");
      const ctx = canvas.getContext("2d");
      canvas.height = viewport.height;
      canvas.width = viewport.width;

      const pageWrapper = document.getElementById("page-wrapper");
      pageWrapper.style.width = viewport.width + "px";
      pageWrapper.style.height = viewport.height + "px";

      const textLayerDiv = document.getElementById("text-layer");
      textLayerDiv.innerHTML = "";
      textLayerDiv.style.width = viewport.width + "px";
      textLayerDiv.style.height = viewport.height + "px";
      textLayerDiv.style.setProperty("--scale-factor", viewport.scale);

      // 1. Renderizar Canvas
      await page.render({{ canvasContext: ctx, viewport: viewport }}).promise;

      // 2. Renderizar Text Layer para selección y resaltado
      const textContent = await page.getTextContent();
      const textLayer = pdfjsLib.renderTextLayer({{
        textContentSource: textContent,
        container: textLayerDiv,
        viewport: viewport,
        textDivs: []
      }});
      await textLayer.promise;

      // 3. Subrayar y resaltar el fragmento con Mark.js (frase exacta o subfrases consecutivas, cero palabras sueltas)
      if (targetQuery && targetQuery.trim().length > 0) {{
        const cleanQ = targetQuery.trim().replace(/^["'“”«»]+|["'“”«»]+$/g, "").replace(/\.{2,}/g, "").trim();
        const markInst = new Mark(textLayerDiv);

        // Paso 1: Intentar frase completa continua a través de spans del PDF
        markInst.mark(cleanQ, {{
          className: "pdf-highlight",
          separateWordSearch: false,
          acrossElements: true,
          diacritics: true,
          accuracy: "partially",
          done: function(count) {{
            if (count > 0) {{
              updateMatchStatus(count);
              scrollToHighlight();
              return;
            }}

            // Paso 2: Si la frase completa no coincide por saltos de línea con guión,
            // buscar las primeras 5 a 6 palabras consecutivas de la cita (nunca palabras sueltas aisladas)
            const words = cleanQ.split(/\s+/).filter(w => w.length > 0);
            if (words.length >= 4) {{
              const subPhrase1 = words.slice(0, 5).join(" ");
              markInst.mark(subPhrase1, {{
                className: "pdf-highlight",
                separateWordSearch: false,
                acrossElements: true,
                diacritics: true,
                done: function(subCount1) {{
                  if (subCount1 > 0) {{
                    updateMatchStatus(subCount1);
                    scrollToHighlight();
                    return;
                  }}
                  if (words.length >= 8) {{
                    const subPhrase2 = words.slice(3, 8).join(" ");
                    markInst.mark(subPhrase2, {{
                      className: "pdf-highlight",
                      separateWordSearch: false,
                      acrossElements: true,
                      diacritics: true,
                      done: function(subCount2) {{
                        updateMatchStatus(subCount2);
                        if (subCount2 > 0) scrollToHighlight();
                      }}
                    }});
                  }} else {{
                    updateMatchStatus(0);
                  }}
                }}
              }});
            }} else {{
              updateMatchStatus(0);
            }}
          }}
        }});
      }}

      isRendering = false;
    }}

    function updateMatchStatus(count) {{
      const stEl = document.getElementById("match-status");
      if (!stEl) return;
      if (count > 0) {{
        stEl.textContent = "✓ Resaltado en página";
        stEl.style.color = "#15803D";
      }} else {{
        stEl.textContent = "Revise el contexto de la página";
        stEl.style.color = "#94A3B8";
      }}
    }}

    function scrollToHighlight() {{
      setTimeout(() => {{
        const firstHl = document.querySelector("mark.pdf-highlight");
        if (firstHl) {{
          firstHl.scrollIntoView({{ behavior: "smooth", block: "center" }});
        }}
      }}, 120);
    }}

    function changePage(delta) {{
      if (!pdfDoc) return;
      const target = pageNum + delta;
      if (target >= 1 && target <= pdfDoc.numPages) {{
        pageNum = target;
        renderPage(pageNum);
      }}
    }}

    function zoom(delta) {{
      const newScale = Math.min(Math.max(scale + delta, 0.75), 2.3);
      if (newScale !== scale) {{
        scale = newScale;
        renderPage(pageNum);
      }}
    }}

    function resetZoom() {{
      if (scale !== 1.35) {{
        scale = 1.35;
        renderPage(pageNum);
      }}
    }}
  </script>
</body>
</html>'''
    return html
