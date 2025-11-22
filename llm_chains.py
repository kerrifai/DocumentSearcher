# llm_chains.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain


def create_llm(api_key: str, model: str = "gpt-4o-mini", temperature: float = 0.2):
    """
    Crea y devuelve una instancia de ChatOpenAI.
    """
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        temperature=temperature,
    )


def create_summary_chain(llm: ChatOpenAI) -> LLMChain:
    """
    Devuelve la chain para generar el resumen / explicación del PDF.
    """
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
    return LLMChain(llm=llm, prompt=summary_prompt)


def create_qa_chain(llm: ChatOpenAI) -> LLMChain:
    """
    Devuelve la chain para preguntas y respuestas basadas en contexto.
    """
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
    return LLMChain(llm=llm, prompt=qa_prompt)
