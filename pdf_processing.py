# pdf_processing.py
import tempfile
from typing import Tuple, List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


def cargar_y_procesar_pdfs(
    files,
    api_key: str,
    chunk_size: int,
    chunk_overlap: int,
    max_chars_full_text: int = 30000,
):
    """
    Crea documentos, vector DB y texto total a partir de uno o varios PDFs.

    - files: lista de UploadedFile de Streamlit (o file-like)
    - api_key: API key de OpenAI
    - chunk_size, chunk_overlap: parámetros del splitter
    - max_chars_full_text: recorte del texto total para el resumen
    """
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
        chunk_overlap=chunk_overlap,
    )
    chunked_docs = splitter.split_documents(all_docs)

    # Crear base vectorial en memoria (sin persistencia a disco)
    vectordb = Chroma.from_documents(
        documents=chunked_docs,
        embedding=OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=api_key,
        ),
    )

    # Texto completo para resumen (capado para no reventar el contexto)
    full_text = "\n\n".join(d.page_content for d in all_docs)
    full_text = full_text[:max_chars_full_text]

    return vectordb, full_text
