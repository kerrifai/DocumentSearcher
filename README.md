# 📊 PDF Assistant (Streamlit + LangChain)

Interactive web application built with Python, Streamlit, and LangChain to upload one or multiple PDF files, automatically extract their content, generate structured explanations, and perform question-answering based exclusively on the PDF text.

The app processes uploaded PDF files and automatically:
  - Extracts and loads the full text from each PDF.
  - Splits the text into optimized chunks for embeddings.
  - Creates a vector database using OpenAI embeddings.
  - Generates a structured summary of the complete document(s).
  - Performs context-based question answering strictly using PDF content.
  - Provides an interactive user interface to upload PDFs, summarize them, and ask questions.
---

## 🚀 What the APP does?

When the user uploads PDFs, the system performs:

### 1. PDF ingestion & text extraction (implemented in pdf_processing.py )
  - Temporary storage of uploaded files
  - Page-by-page text extraction via PyPDFLoader
  - Document preprocessing

### 2. Chunking & vector database creation
  - Splitting into chunks using RecursiveCharacterTextSplitter
  - Embeddings generated with OpenAI text-embedding-3-large
  - Storage in an in-memory ChromaDB for fast retrieval

### 3. Summarization Chain (defined in llm_chains.py )
  - Generates a structured explanation of the PDF
  - Provides bullet points, global summary, and simplified technical explanations
  - Uses ChatOpenAI for LLM inference

### 4. Context-based Question Answering
  - Retrieves most relevant chunks (k passages)
  - Uses a strict QA chain that answers only using the retrieved context

### 5. Interactive Streamlit Interface (defined in main.py )
  - PDF upload interface
  - API key input
  - Buttons to generate summaries and ask questions
  - Adjust Advanced Parameter, inside the sidebar, you can tune:
      - chunk_size 
      - chunk_overlap
      - k_passages for retrieval 

These parameters affect how the PDF is split and how precisely the assistant retrieves information.
## ▶️ How to Run the Application

### 1. Open VSCode terminal and clone the repository
```bash
git clone https://github.com/kerrifai/DocumentSearcher.git

cd DocumentSearcher
```

### 2. Create a virtual environment and activate it.
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```


### 5 .Run the Streamlit application
```bash
streamlit run main.py
```

## Notes
You must enter your OpenAI API Key in the sidebar.
This key is:
  - Not saved
  - Used only in your session
  - Required for LLMs and embeddings
 

