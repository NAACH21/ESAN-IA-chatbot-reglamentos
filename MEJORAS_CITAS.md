# Mejora del sistema de citas normativas (implementación local)

## Qué cambió

- **`citation_utils.py`** (nuevo): valida que cada cita textual se encuentre en un fragmento recuperado, recupera del servidor el nombre y página reales, detecta extractos iguales o solapados y mantiene extractos diferentes de una misma página. Incluye detección conservadora del ámbito Doctorado / Diploma-PEE / Maestría.
- **`rag_engine.py`**: entrega al modelo fragmentos identificados como `F1`, `F2`, ... y exige que devuelva `fragmento_id` + `extracto` literal; ya no acepta que el modelo decida libremente fuente, página o sección. Revisa y audita cada cita; si no hay evidencia literal, marca el dictamen como pendiente. Aplica un filtro de ámbito cuando el programa está explícito. El contexto adicional ahora se considera también en la recuperación.
- **`app.py`**: muestra solo citas validadas, no resultados brutos del buscador; distingue cantidad de citas y documentos; informa duplicados descartados; advierte cuando falta ámbito o validación; elimina el saludo duplicado en el botón de correo; evita HTML crudo en el texto insertado por el modelo y deja de llamar “verificación oficial” al flujo automático.
- **`tests/test_citations.py`**: once pruebas unitarias, que incluyen casos construidos a partir de las capturas y citas tomadas de las páginas 7 y 32 de los PDF proporcionados.

## Ejecutar

Usar las dependencias de `requirements.txt` como antes:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Probar reglas de citación sin llamar a OpenAI:

```bash
python -m unittest discover -s tests -v
```

**La base ChromaDB existente puede seguir funcionando:** no se cambió el formato de los embeddings. Si ha cargado documentos repetidos o versiones anteriores, es recomendable revisarlos y reindexar tras depurar el repositorio documental.

## Qué se espera en los dos ejemplos

1. **Justificación de inasistencias sin programa identificado:** no emitir “No procede” como conclusión general. Advertir que se debe conocer el programa y su reglamento; la misma frase sobre tardanza recuperada en varios chunks genera una sola cita, no cinco.
2. **Retiro de curso de Diploma:** excluir como fuentes normativas específicas los reglamentos de Doctorado y de Maestrías. Se pueden conservar varias citas de la misma página 32 del reglamento de Educación Ejecutiva si cada una acredita una regla distinta (solicitud por escrito, plazo, efectos económicos, etc.).

## Distinciones esenciales

- **Coincidencia literal:** el extracto aparece en el texto de un fragmento recuperado (verificado por código).
- **Pertinencia:** ese texto realmente justifica la conclusión y aplica al tipo de programa (el filtrado básico ayuda, pero necesita revisión).
- **Vigencia:** el documento usado es la versión oficialmente aplicable al estudiante y periodo académico (no puede inferirse por aparecer en la carpeta `DOCS`).

Por tanto, una cita "textualmente verificada" **no constituye una verificación legal/oficial**. El sistema sigue siendo una ayuda para la Coordinación y no debería enviar borradores sin validación.

## Mejoras recomendadas siguientes

1. **Catálogo de normativa** con `tipo_documento`, `programa`, `version`, `vigente_desde`, `vigente_hasta`, `aprobacion` y `hash_archivo`; evitar que versiones duplicadas compitan por resultados y evitar mezclar comunicaciones internas con reglamentos.
2. **Verificación de pertinencia por afirmación**: desglosar el dictamen en proposiciones identificables, asociar a cada una las citas que la sustentan y comprobar la implicación normativa; una coincidencia textual por sí sola no basta.
3. **Citas a nivel artículo / inciso**, no solo página: anclar cada regla a su epígrafe y, cuando se disponga, a su numeración oficial.
4. **Revisión de paginación**: diferenciar página del visor PDF de la numeración impresa en el reglamento (pueden diferir).
5. **Pruebas de regresión con consultas reales**: medir duplicación, citas inexistentes, documentos fuera de ámbito, citas pertinentes y decisiones mal clasificadas. Confirmar el comportamiento con cada proveedor/modelo configurado.
6. **Aprobación humana antes del envío:** enlazar el protocolo de verificación con la habilitación de los botones de copia o exportación; hoy las marcas no son una certificación de revisión.

## Alcance de la prueba realizada

Se ejecutaron **11 pruebas unitarias correctamente**, incluyendo los dos PDFs locales mencionados. **No se ejecutó una consulta RAG extremo a extremo**: esta entrega no contiene credenciales de OpenAI y en el entorno de pruebas no se instaló ChromaDB ni Streamlit. Deberá realizarse la prueba integrada en su despliegue antes de usarlo institucionalmente.
