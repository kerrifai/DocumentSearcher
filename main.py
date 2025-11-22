# main.py
import streamlit as st

from pdf_processing import cargar_y_procesar_pdfs
from llm_chains import create_llm, create_summary_chain, create_qa_chain


# -------------------------------
# Configuración básica de Streamlit
# -------------------------------
st.set_page_config(page_title="Asistente PDF", layout="wide")
st.title("📄 Asistente para explicar y consultar PDFs")
st.write(
    "Sube uno o varios archivos PDF y el asistente generará un resumen "
    "del contenido y responderá a tus preguntas usando LangChain + OpenAI."
)

# -------------------------------
# Entrada de API Key
# -------------------------------
with st.sidebar:
    st.header("🔐 Configuración")
    openai_api_key = st.text_input(
        "Introduce tu OpenAI API Key",
        type="password",
        help="Tu clave NO se guarda. Solo se usa en esta sesión.",
    )

    chunk_size = st.slider(
        "Tamaño de fragmento (tokens aprox.)", 500, 2000, 1000, 100
    )
    chunk_overlap = st.slider(
        "Solapamiento entre fragmentos", 0, 300, 100, 50
    )
    k_passages = st.slider(
        "Nº de fragmentos a recuperar (k)", 2, 10, 5, 1
    )

    st.markdown("---")
    st.caption("Consejo: usa gpt-4o-mini para pruebas (más barato).")

if not openai_api_key:
    st.info("⚠️ Introduce tu API Key en la barra lateral para empezar.")
    st.stop()

# -------------------------------
# Carga de PDFs
# -------------------------------
uploaded_files = st.file_uploader(
    "Sube uno o varios PDFs",
    type="pdf",
    accept_multiple_files=True,
)

# Estado de sesión
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
if "docs_text" not in st.session_state:
    st.session_state.docs_text = ""
if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False

# -------------------------------
# Botón para procesar PDFs
# -------------------------------
col1, col2 = st.columns([1, 2])

with col1:
    procesar = st.button("📥 Procesar PDF(s)")

if procesar:
    if not uploaded_files:
        st.warning("Primero debes subir al menos un PDF.")
    else:
        with st.spinner("Procesando PDFs y generando embeddings..."):
            vectordb, full_text = cargar_y_procesar_pdfs(
                uploaded_files,
                api_key=openai_api_key,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            st.session_state.vectordb = vectordb
            st.session_state.docs_text = full_text
            st.session_state.pdf_loaded = True
        st.success(
            "✅ PDFs procesados correctamente. Ya puedes pedir un resumen o hacer preguntas."
        )

# -------------------------------
# Inicializar LLM y chains
# -------------------------------
llm = create_llm(openai_api_key)
summary_chain = create_summary_chain(llm)
qa_chain = create_qa_chain(llm)

# -------------------------------
# Sección: Explicar el contenido del PDF
# -------------------------------
st.markdown("## 🧾 Explicar el contenido del/los PDF(s)")

if not st.session_state.pdf_loaded:
    st.info("Sube y procesa uno o varios PDFs para generar el resumen.")
else:
    if st.button("🧠 Generar explicación completa del contenido"):
        with st.spinner("Generando resumen y explicación del contenido..."):
            resumen = summary_chain.run(document=st.session_state.docs_text)
        st.subheader("📚 Explicación del documento")
        st.write(resumen)

# -------------------------------
# Sección: Preguntas y respuestas sobre el PDF
# -------------------------------
st.markdown("## ❓ Preguntas sobre el contenido del PDF")

pregunta = st.text_area(
    "Escribe tu pregunta (ejemplo: *¿Qué dice el manual sobre la interfaz UART?*):",
    height=100,
)

if st.button("Enviar pregunta"):
    if not st.session_state.pdf_loaded:
        st.warning("Primero debes subir y procesar al menos un PDF.")
    elif not pregunta.strip():
        st.warning("Por favor, escribe una pregunta.")
    else:
        vectordb = st.session_state.vectordb
        resultados_similares = vectordb.similarity_search(
            pregunta, k=k_passages
        )

        contexto = ""
        for doc in resultados_similares:
            contexto += doc.page_content + "\n\n"

        from langchain_core.documents import Document  # opcional si quieres tipos

        with st.spinner("Buscando en el documento y generando respuesta..."):
            respuesta = qa_chain.run(input=pregunta, context=contexto)

        st.subheader("Respuesta")
        st.write(respuesta)

        with st.expander("🔎 Ver fragmentos de contexto usados"):
            for i, doc in enumerate(resultados_similares, start=1):
                page = doc.metadata.get("page", "N/A")
                st.markdown(f"**Fragmento {i} (página {page}):**")
                st.write(doc.page_content)
                st.write("---")
