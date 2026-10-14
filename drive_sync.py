"""
drive_sync.py
Sincronización automática desde Google Drive (carpeta pública compartida).
Detecta archivos nuevos no indexados y los descarga para indexarlos.
"""
import os
import sys
import re
import json
import tempfile
import shutil
import requests
from typing import List, Dict, Tuple

try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import chromadb

CHROMA_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
COLLECTION_NAME  = "esan_reglamentos"

# Extensiones admitidas en Drive
DRIVE_SUPPORTED_MIME = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
}

# ── Utilidades de ChromaDB ────────────────────────────────────────────────────

def get_indexed_sources() -> List[str]:
    """Retorna lista de nombres de archivo ya indexados en ChromaDB."""
    try:
        client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
        col = client.get_collection(name=COLLECTION_NAME)
        result = col.get(include=["metadatas"])
        sources = list({m["source"] for m in result["metadatas"] if "source" in m})
        return sources
    except Exception:
        return []


# ── Lectura del folder de Drive ───────────────────────────────────────────────

def extract_folder_id(url: str) -> str:
    """Extrae el folder ID desde una URL de Google Drive."""
    patterns = [
        r"drive\.google\.com/drive/folders/([a-zA-Z0-9_-]+)",
        r"drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)",
        r"id=([a-zA-Z0-9_-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"No se pudo extraer el folder ID de: {url}")


def _list_drive_folder_api(folder_id: str, api_key: str = "") -> List[Dict]:
    """
    Lista archivos en un folder de Drive usando la API v3.
    Si no hay api_key, intenta con la API pública (solo funciona para carpetas públicas).
    """
    base = "https://www.googleapis.com/drive/v3/files"
    params = {
        "q": f"'{folder_id}' in parents and trashed=false",
        "fields": "files(id,name,mimeType,modifiedTime,size)",
        "pageSize": 200,
    }
    if api_key:
        params["key"] = api_key

    files = []
    page_token = None
    while True:
        if page_token:
            params["pageToken"] = page_token
        try:
            resp = requests.get(base, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            raise RuntimeError(f"Error al consultar Google Drive API: {e}")

        for f in data.get("files", []):
            if f.get("mimeType") in DRIVE_SUPPORTED_MIME:
                ext = DRIVE_SUPPORTED_MIME[f["mimeType"]]
                name = f["name"]
                if not name.lower().endswith(tuple(DRIVE_SUPPORTED_MIME.values())):
                    name += ext
                files.append({
                    "id": f["id"],
                    "name": name,
                    "mimeType": f["mimeType"],
                    "modifiedTime": f.get("modifiedTime", ""),
                })

        page_token = data.get("nextPageToken")
        if not page_token:
            break

    return files


def _list_drive_folder_gdown(folder_id: str) -> List[Dict]:
    """
    Lista archivos usando gdown (sin necesidad de API key).
    Recorre recursivamente todas las subcarpetas de Google Drive.
    """
    import gdown
    import io
    import contextlib

    url = f"https://drive.google.com/drive/folders/{folder_id}"
    files_info = []

    try:
        # Redirigir stdout y stderr para evitar UnicodeEncodeError en Windows con emojis
        buff = io.StringIO()
        with contextlib.redirect_stdout(buff), contextlib.redirect_stderr(buff):
            items = gdown.download_folder(
                url=url,
                skip_download=True,
                quiet=True,
                use_cookies=False,
            )

        if items:
            for item in items:
                file_rel = item.path if hasattr(item, "path") else ""
                name = os.path.basename(file_rel)
                lower = name.lower()
                if lower.endswith(tuple(DRIVE_SUPPORTED_MIME.values())):
                    files_info.append({
                        "id": getattr(item, "id", None),
                        "name": name,
                        "rel_path": file_rel,
                        "mimeType": "application/pdf" if lower.endswith(".pdf")
                                    else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                                    if lower.endswith(".docx")
                                    else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        "modifiedTime": "",
                    })
            if files_info:
                return files_info
    except Exception:
        pass

    # Fallback básico por HTML si falla gdown
    try:
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        resp = session.get(f"https://drive.google.com/drive/folders/{folder_id}", timeout=20)
        names_found = re.findall(r'"([^"]+\.(?:pdf|docx|xlsx))"', resp.text, re.IGNORECASE)
        unique_names = list(dict.fromkeys(names_found))
        for name in unique_names:
            files_info.append({
                "id": None,
                "name": name,
                "rel_path": name,
                "mimeType": "application/pdf" if name.lower().endswith(".pdf")
                            else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                            if name.lower().endswith(".docx")
                            else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "modifiedTime": "",
            })
        return files_info
    except Exception as e:
        raise RuntimeError(f"Error al listar la carpeta desde Drive: {e}")


def list_drive_files(folder_url: str, api_key: str = "") -> List[Dict]:
    """
    Lista archivos PDF/DOCX/XLSX de una carpeta pública de Google Drive.
    Prioriza la API v3 (más precisa) y cae en gdown si no hay api_key.
    """
    folder_id = extract_folder_id(folder_url)
    try:
        if api_key:
            return _list_drive_folder_api(folder_id, api_key)
        else:
            # Intentar con API sin key (funciona para carpetas públicas)
            return _list_drive_folder_api(folder_id, api_key="")
    except Exception:
        # Fallback a gdown
        return _list_drive_folder_gdown(folder_id)


# ── Descarga individual desde Drive ──────────────────────────────────────────

def download_file_from_drive(file_id: str, dest_path: str) -> bool:
    """Descarga un archivo de Drive por su ID usando gdown."""
    import gdown
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        gdown.download(url, dest_path, quiet=True, fuzzy=True)
        return os.path.exists(dest_path) and os.path.getsize(dest_path) > 100
    except Exception:
        return False


# ── Sincronización principal ──────────────────────────────────────────────────

def sync_from_drive(
    folder_url: str,
    docs_dir: str,
    api_key: str = "",
    progress_callback=None,
) -> Dict:
    """
    Compara los archivos en Drive con los indexados en ChromaDB.
    Descarga e indexa solo los archivos nuevos o no presentes.

    Retorna:
        {
            "drive_total": int,     - archivos encontrados en Drive
            "already_indexed": int, - ya indexados
            "new_files": [str],     - nombres de archivos nuevos descargados
            "failed": [str],        - archivos que no se pudieron descargar
        }
    """
    os.makedirs(docs_dir, exist_ok=True)

    if progress_callback:
        progress_callback("Consultando carpeta de Google Drive...")

    try:
        drive_files = list_drive_files(folder_url, api_key)
    except Exception as e:
        raise RuntimeError(f"No se pudo acceder a la carpeta de Drive: {e}")

    if progress_callback:
        progress_callback(f"Encontrados {len(drive_files)} documentos en Drive.")

    indexed_sources = get_indexed_sources()
    indexed_set = set(indexed_sources)

    new_files  = []
    failed     = []
    already    = 0

    for f in drive_files:
        name = f["name"]
        # Normalizar: quitar rutas si las hubiera
        base_name = os.path.basename(name)

        if base_name in indexed_set:
            already += 1
            if progress_callback:
                progress_callback(f"  [YA INDEXADO] {base_name}")
            continue

        dest = os.path.join(docs_dir, base_name)
        if progress_callback:
            progress_callback(f"  [DESCARGANDO] {base_name}...")

        ok = False
        if f.get("id"):
            ok = download_file_from_drive(f["id"], dest)

        # Fallback: si el archivo con ese nombre ya existe localmente, usarlo
        if not ok and os.path.exists(dest) and os.path.getsize(dest) > 100:
            ok = True

        if ok:
            new_files.append(base_name)
        else:
            failed.append(base_name)
            if progress_callback:
                progress_callback(f"  [FALLO] No se pudo descargar: {base_name}")

    return {
        "drive_total": len(drive_files),
        "already_indexed": already,
        "new_files": new_files,
        "failed": failed,
    }


# ── Descarga completa de carpeta (sync total) ─────────────────────────────────

def download_full_folder_gdown(folder_url: str, docs_dir: str, progress_callback=None) -> List[str]:
    """
    Descarga TODOS los archivos de la carpeta de Drive usando gdown.
    Útil para la primera sincronización. Retorna lista de nombres descargados.
    """
    import gdown
    import io
    import contextlib

    os.makedirs(docs_dir, exist_ok=True)
    if progress_callback:
        progress_callback("Descargando carpeta completa desde Google Drive (con subcarpetas)...")
    try:
        buff = io.StringIO()
        with contextlib.redirect_stdout(buff), contextlib.redirect_stderr(buff):
            downloaded = gdown.download_folder(
                url=folder_url,
                output=docs_dir,
                quiet=True,
                use_cookies=False,
                remaining_ok=True,
            )
        if downloaded:
            names = [os.path.basename(p) for p in downloaded]
            if progress_callback:
                progress_callback(f"Descargados {len(names)} archivos.")
            return names
        return []
    except Exception as e:
        raise RuntimeError(f"Error al descargar la carpeta de Drive: {e}")
