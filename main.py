import tempfile
from pathlib import Path

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain


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
        help="Tu clave NO se guarda. Solo se usa en esta sesión."
    )

    chunk_size = st.slider("Tamaño de fragmento (tokens aprox.)", 500, 2000, 1000, 100)
    chunk_overlap = st.slider("Solapamiento entre fragmentos", 0, 300, 100, 50)
    k_passages = st.slider("Nº de fragmentos a recuperar (k)", 2, 10, 5, 1)

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
    accept_multiple_files=True
)

if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
if "docs_text" not in st.session_state:
    st.session_state.docs_text = ""
if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False


def cargar_y_procesar_pdfs(files, api_key: str):
    """Crea documentos, vector DB y texto total a partir de uno o varios PDFs."""
    all_docs = []

    # Guardar PDFs temporales y cargarlos con PyPDFLoader
    for f in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(f.read())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        all_docs.extend(docs)

    # Dividir en fragmentos
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunked_docs = splitter.split_documents(all_docs)

    # Crear base vectorial en memoria (sin persistencia a disco)
    vectordb = Chroma.from_documents(
        documents=chunked_docs,
        embedding=OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=api_key
        )
    )

    # Texto completo para resumen (capado para no reventar el contexto)
    full_text = "\n\n".join(d.page_content for d in all_docs)
    # Opcional: limitar tamaño a X caracteres
    full_text = full_text[:30000]

    return vectordb, full_text


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
            vectordb, full_text = cargar_y_procesar_pdfs(uploaded_files, openai_api_key)
            st.session_state.vectordb = vectordb
            st.session_state.docs_text = full_text
            st.session_state.pdf_loaded = True
        st.success("✅ PDFs procesados correctamente. Ya puedes pedir un resumen o hacer preguntas.")


# -------------------------------
# Inicializar LLM y chains
# -------------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",  # puedes cambiar a "gpt-4o"
    api_key=openai_api_key,
    temperature=0.2
)

# Prompt para explicar el contenido completo del PDF
summary_prompt_template = """
Eres un asistente experto que explica el contenido de uno o varios documentos PDF.

A partir del siguiente texto (que puede estar fragmentado o incompleto), genera:
1. Un resumen general del documento en español claro.
2. Una lista de los puntos clave en viñetas.
3. Una explicación sencilla de los conceptos técnicos importantes, como si se lo explicaras
   a un estudiante de ingeniería que no ha leído el documento.

Texto del documento:
{document}

Responde de forma estructurada usando títulos y listas.
"""

summary_prompt = PromptTemplate.from_template(summary_prompt_template)
summary_chain = LLMChain(llm=llm, prompt=summary_prompt)

# Prompt para preguntas concretas
qa_prompt_template = """
Eres un asistente que responde preguntas basadas EXCLUSIVAMENTE en el contexto proporcionado.

Contexto:
{context}

Pregunta del usuario:
{input}

Instrucciones:
- Usa SOLO la información del contexto.
- Si la respuesta no está en el contexto, responde exactamente:
"Lo siento, no tengo la información necesaria para responder a esa pregunta."

Responde en español.
"""

qa_prompt = PromptTemplate.from_template(qa_prompt_template)
qa_chain = LLMChain(llm=llm, prompt=qa_prompt)


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
    height=100
)

if st.button("Enviar pregunta"):
    if not st.session_state.pdf_loaded:
        st.warning("Primero debes subir y procesar al menos un PDF.")
    elif not pregunta.strip():
        st.warning("Por favor, escribe una pregunta.")
    else:
        # Recuperar fragmentos relevantes
        vectordb = st.session_state.vectordb
        resultados_similares = vectordb.similarity_search(pregunta, k=k_passages)

        contexto = ""
        for doc in resultados_similares:
            contexto += doc.page_content + "\n\n"

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
