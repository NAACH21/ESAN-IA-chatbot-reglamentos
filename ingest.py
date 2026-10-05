import os
import re
from typing import List, Dict, Any
from pypdf import PdfReader
from openai import OpenAI
import chromadb
from chromadb.config import Settings

CHROMA_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
DOCS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DOCS")

def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """Extrae texto página por página con metadata."""
    reader = PdfReader(pdf_path)
    filename = os.path.basename(pdf_path)
    pages_data = []
    
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            # Limpieza básica de saltos de línea excesivos
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
        
        # Buscar corte natural por punto o salto de línea
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

def build_vector_store(api_key: str, docs_dir: str = DOCS_FOLDER, progress_callback=None):
    """Procesa los PDFs y crea los embeddings en ChromaDB."""
    client_ai = OpenAI(api_key=api_key)
    chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
    
    # Obtener o recrear colección
    collection_name = "esan_reglamentos"
    try:
        chroma_client.delete_collection(name=collection_name)
    except Exception:
        pass
        
    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    
    pdf_files = [f for f in os.listdir(docs_dir) if f.lower().endswith(".pdf")]
    # Filtrar duplicados con sufijo (1)
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
            progress_callback(f"Extrayendo texto de [{idx+1}/{total_files}]: {pdf_name}")
            
        pages = extract_text_from_pdf(pdf_path)
        for page in pages:
            chunks = split_into_chunks(page["text"])
            for c_idx, chunk in enumerate(chunks):
                doc_counter += 1
                all_chunks.append(chunk)
                all_metadatas.append({
                    "source": page["filename"],
                    "page": page["page"],
                    "chunk_id": c_idx
                })
                all_ids.append(f"doc_{doc_counter}")

    if not all_chunks:
        raise ValueError("No se encontraron fragmentos de texto en los documentos.")

    # Generar embeddings por lotes para optimizar llamadas a la API
    batch_size = 80
    total_batches = (len(all_chunks) + batch_size - 1) // batch_size
    
    for i in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[i:i + batch_size]
        batch_meta = all_metadatas[i:i + batch_size]
        batch_ids = all_ids[i:i + batch_size]
        
        batch_num = (i // batch_size) + 1
        if progress_callback:
            progress_callback(f"Generando embeddings: lote {batch_num}/{total_batches} ({len(batch_chunks)} fragmentos)...")
            
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
