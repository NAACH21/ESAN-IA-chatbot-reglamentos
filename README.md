# 🎓 Asistente Virtual Normativo - Universidad ESAN (Pregrado)

Prototipo Beta de Chatbot RAG (*Retrieval-Augmented Generation*) para responder consultas académicas y normativas sobre los reglamentos oficiales de la Universidad ESAN con citas de artículos y páginas.

---

## 🚀 Cómo ejecutar la aplicación

### Opción A: Desde la terminal
Ejecuta el siguiente comando en la carpeta raíz del proyecto:
```bash
streamlit run app.py
```

Se abrirá automáticamente en tu navegador web en: `http://localhost:8501`

---

## ⚙️ Pasos para la Demo

1. **Ingresar la API Key:**
   - Puedes colocarla directamente en el archivo `.env` (`OPENAI_API_KEY=sk-...`) o escribirla en el campo seguro de la barra lateral izquierda en la web.
2. **Indexar los Reglamentos:**
   - En la barra lateral izquierda, haz clic en el botón **"⚡ Indexar / Re-indexar PDFs"**.
   - El sistema leerá todos los reglamentos de la carpeta `DOCS`, extraerá el texto, dividirá los artículos en fragmentos semánticos y generará los embeddings en una base de datos vectorial local (ChromaDB).
   - *(Este proceso se hace una sola vez y toma ~30 segundos. El costo aproximado en OpenAI es de menos de $0.01 USD).*
3. **Probar el Chatbot:**
   - Escribe cualquier pregunta o haz clic en los botones rápidos de ejemplo en la barra lateral:
     - *¿Cuáles son los requisitos para obtener el grado de Bachiller?*
     - *¿Cómo es el proceso y requisitos para convalidar asignaturas?*
     - *¿Qué sanciones se aplican por cometer plagio en evaluaciones?*
     - *¿En qué condiciones un estudiante entra en bajo rendimiento académico?*
     - *¿Cuáles son los requisitos para postular a una beca en el extranjero?*
4. **Verificar las Citas:**
   - Debajo de cada respuesta verás el desplegable **"🔍 Ver fuentes oficiales consultadas"** que muestra el PDF de origen, la página exacta y el fragmento textual recuperado.

---

## 📁 Estructura del Proyecto

* `app.py`: Interfaz de usuario interactiva construida en Streamlit con branding de ESAN.
* `rag_engine.py`: Motor de búsqueda semántica (ChromaDB) y generación de respuestas con `gpt-4o-mini` y citas obligatorias.
* `ingest.py`: Extracción de texto de PDFs (`pypdf`), chunking y generación de embeddings (`text-embedding-3-small`).
* `DOCS/`: Carpeta con los 10 reglamentos oficiales en formato PDF.
* `chroma_db/`: Base de datos vectorial persistente generada localmente.

## 🔎 Control de citas (actualización)

Ver [`MEJORAS_CITAS.md`](MEJORAS_CITAS.md) para el flujo de citas textualmente verificadas,
control de duplicados, diferencias entre programas, pruebas locales y límites de la validación.

```bash
python -m unittest discover -s tests -v
```
