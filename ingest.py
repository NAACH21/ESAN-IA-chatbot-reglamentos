import os
import re
import sys
import shutil
from typing import List, Dict, Any
from pypdf import PdfReader
from openai import OpenAI

# Compatibilidad SQLite para despliegues en la nube (Linux / Streamlit Cloud)
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import chromadb

CHROMA_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
DOCS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DOCS")
ARCHIVE_FOLDER = os.path.join(DOCS_FOLDER, "_archivados")

def detect_section_or_article(text: str) -> str:
    """Detecta referencias a Títulos, Capítulos o Artículos dentro del fragmento."""
    matches = []
    
    cap_match = re.search(r'(CAP[ÍI]TULO\s+[IVXLCDM\d]+|T[ÍI]TULO\s+[IVXLCDM\d]+)', text, re.IGNORECASE)
    if cap_match:
        matches.append(cap_match.group(1).upper())
        
    art_matches = re.findall(r'(Art(?:[íi]culo|\.)\s*\d+(?:\s*°|\º)?)', text, re.IGNORECASE)
    if art_matches:
        unique_arts = []
        for a in art_matches:
            clean_a = re.sub(r'\s+', ' ', a).strip()
            if clean_a not in unique_arts:
                unique_arts.append(clean_a)
        matches.append(", ".join(unique_arts[:3]))
        
    if matches:
        return " | ".join(matches)
    return "Sección General / Normativa"

def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """Extrae texto página por página con metadata."""
    reader = PdfReader(pdf_path)
    filename = os.path.basename(pdf_path)
    pages_data = []
    
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            text = re.sub(r'\n\s*\n', '\n\n', text)
            pages_data.append({
                "filename": filename,
                "page": page_num,
                "text": text
            })
    return pages_data

def split_into_chunks(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    """Divide texto en chunks semánticos con solapamiento."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:].strip())
            break
        
        split_at = text.rfind("\n\n", start, end)
        if split_at == -1 or split_at <= start:
            split_at = text.rfind(". ", start, end)
        if split_at == -1 or split_at <= start:
            split_at = text.rfind("\n", start, end)
        if split_at == -1 or split_at <= start:
            split_at = end
        else:
            split_at += 1
            
        chunk = text[start:split_at].strip()
        if chunk:
            chunks.append(chunk)
        start = max(start + 1, split_at - overlap)
        
    return chunks

def remove_document_from_store(filename: str, docs_dir: str = DOCS_FOLDER) -> int:
    """Elimina los vectores de un documento en ChromaDB y retira el PDF de DOCS."""
    chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
    removed_chunks = 0
    try:
        collection = chroma_client.get_collection(name="esan_reglamentos")
        existing = collection.get(where={"source": filename})
        if existing and existing.get("ids"):
            removed_chunks = len(existing["ids"])
            collection.delete(where={"source": filename})
    except Exception:
        pass

    # Mover el archivo a _archivados para que deje de estar en DOCS (con opción de restaurar después de la demo)
    src_path = os.path.join(docs_dir, filename)
    if os.path.exists(src_path):
        os.makedirs(ARCHIVE_FOLDER, exist_ok=True)
        dst_path = os.path.join(ARCHIVE_FOLDER, filename)
        shutil.move(src_path, dst_path)

    return removed_chunks

def index_single_pdf(api_key: str, pdf_path: str) -> int:
    """Indexa un único PDF nuevo directamente en la colección existente de ChromaDB."""
    client_ai = OpenAI(api_key=api_key)
    chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
    collection = chroma_client.get_or_create_collection(
        name="esan_reglamentos",
        metadata={"hnsw:space": "cosine"}
    )
    filename = os.path.basename(pdf_path)
    # Limpiar si ya existía previamente
    try:
        collection.delete(where={"source": filename})
    except Exception:
        pass

    pages = extract_text_from_pdf(pdf_path)
    chunks_list = []
    metas_list = []
    ids_list = []

    for page in pages:
        chunks = split_into_chunks(page["text"])
        for c_idx, chunk in enumerate(chunks):
            section_ref = detect_section_or_article(chunk)
            chunks_list.append(chunk)
            metas_list.append({
                "source": filename,
                "page": page["page"],
                "section": section_ref,
                "chunk_id": c_idx
            })
            ids_list.append(f"doc_{filename}_p{page['page']}_c{c_idx}")

    if not chunks_list:
        return 0

    response = client_ai.embeddings.create(
        model="text-embedding-3-small",
        input=chunks_list
    )
    embeddings = [data.embedding for data in response.data]

    collection.add(
        ids=ids_list,
        embeddings=embeddings,
        documents=chunks_list,
        metadatas=metas_list
    )
    return len(chunks_list)

def restore_archived_documents(docs_dir: str = DOCS_FOLDER) -> List[str]:
    """Restaura los documentos eliminados desde la papelera local a DOCS."""
    restored = []
    if os.path.exists(ARCHIVE_FOLDER):
        for f in os.listdir(ARCHIVE_FOLDER):
            if f.lower().endswith(".pdf"):
                shutil.move(os.path.join(ARCHIVE_FOLDER, f), os.path.join(docs_dir, f))
                restored.append(f)
    return restored

def build_vector_store(api_key: str, docs_dir: str = DOCS_FOLDER, progress_callback=None):
    """Procesa los PDFs y crea los embeddings en ChromaDB con metadata de sección."""
    client_ai = OpenAI(api_key=api_key)
    chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
    
    collection_name = "esan_reglamentos"
    try:
        chroma_client.delete_collection(name=collection_name)
    except Exception:
        pass
        
    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    
    os.makedirs(docs_dir, exist_ok=True)
    pdf_files = [f for f in os.listdir(docs_dir) if f.lower().endswith(".pdf")]
    pdf_files = [f for f in pdf_files if not (" (1)" in f or "(1)" in f)]
    pdf_files.sort()
    
    total_files = len(pdf_files)
    all_chunks = []
    all_metadatas = []
    all_ids = []
    
    doc_counter = 0
    for idx, pdf_name in enumerate(pdf_files):
        pdf_path = os.path.join(docs_dir, pdf_name)
        if progress_callback:
            progress_callback(f"Extrayendo normativa [{idx+1}/{total_files}]: {pdf_name}")
            
        pages = extract_text_from_pdf(pdf_path)
        for page in pages:
            chunks = split_into_chunks(page["text"])
            for c_idx, chunk in enumerate(chunks):
                doc_counter += 1
                section_ref = detect_section_or_article(chunk)
                all_chunks.append(chunk)
                all_metadatas.append({
                    "source": page["filename"],
                    "page": page["page"],
                    "section": section_ref,
                    "chunk_id": c_idx
                })
                all_ids.append(f"doc_{doc_counter}")

    if not all_chunks:
        raise ValueError("No se encontraron fragmentos de texto en los documentos.")

    batch_size = 80
    total_batches = (len(all_chunks) + batch_size - 1) // batch_size
    
    for i in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[i:i + batch_size]
        batch_meta = all_metadatas[i:i + batch_size]
        batch_ids = all_ids[i:i + batch_size]
        
        batch_num = (i // batch_size) + 1
        if progress_callback:
            progress_callback(f"Indexando vectores: lote {batch_num}/{total_batches} ({len(batch_chunks)} secciones)...")
            
        response = client_ai.embeddings.create(
            model="text-embedding-3-small",
            input=batch_chunks
        )
        embeddings = [data.embedding for data in response.data]
        
        collection.add(
            ids=batch_ids,
            embeddings=embeddings,
            documents=batch_chunks,
            metadatas=batch_meta
        )
        
    return {
        "total_files": total_files,
        "total_chunks": len(all_chunks),
        "collection_name": collection_name
    }

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Define OPENAI_API_KEY en tu entorno o archivo .env")
    else:
        print(f"Indexando documentos desde {DOCS_FOLDER}...")
        res = build_vector_store(api_key, progress_callback=print)
        print(f"Listo! Se indexaron {res['total_files']} PDFs ({res['total_chunks']} chunks).")
