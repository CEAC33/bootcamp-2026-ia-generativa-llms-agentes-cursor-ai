# bootcamp-2026-ia-generativa-llms-agentes-cursor-ai
Bootcamp 2026 IA Generativa, LLM Apps, Agentes IA, Cursor AI

### Arquitectura Básica - Toy Demo
- LLM
- Orchestration Framework
- Vector Database
- Toy UI (Streamlit)

### Arquitectura Avanzada - Profesional
- External APIs
- LLM
- AWS S3, cloud storage
- Backend Server - Render.com (fastAPI, orchestration frm, database)
- Frontend Server - Vercel.com (next.js)

### Componentes de la Técnica RAG
- Embeddings
- Vector Database

### Tips Avanzados RAG
- Splitters
- Retrieval QA Chain
- Vector Database
- Debugging

### RAG vs CAG

**Generación Aumentada por Recuperación (RAG)**

RAG combina las capacidades de recuperación de información y generación de texto para mejorar las respuestas de los modelos de lenguaje. Cuando se presenta una consulta, el sistema recupera documentos relevantes desde una base de datos externa y utiliza esa información para generar una respuesta más informada. Este enfoque permite a los modelos acceder a información actualizada y específica del dominio sin necesidad de ser reentrenados continuamente. En algunos casos, RAG puede introducir complejidades como mayor latencia debido a la recuperación en tiempo real y posibles errores en la selección de documentos.

**Generación Aumentada por Caché (CAG)**

La Generación Aumentada por Caché (CAG) todavía está en una etapa muy inicial. Algunos dicen que, en un futuro cercano, podrá abordar algunas de las limitaciones inherentes de RAG precargando el conocimiento relevante directamente en la ventana de contexto extendida del modelo de lenguaje. Este método implica curar un conjunto de datos estático, precargarlo en el contexto del modelo y utilizar el almacenamiento en caché del estado de inferencia para guardar estados intermedios. Al hacerlo, CAG elimina la necesidad de recuperación en tiempo real, reduciendo así la latencia y simplificando la arquitectura del sistema. Este enfoque es especialmente útil en escenarios donde la base de conocimiento es estable y puede caber dentro del contexto del modelo.

