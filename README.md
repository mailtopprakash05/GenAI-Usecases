# GenAI-Usecases — Healthcare Assistant (RAG)

Author: Prakash Pujari

This repository contains a Retrieval-Augmented Generation (RAG) example: a
healthcare assistant web app built with Streamlit that retrieves information
from your PDF documents using a FAISS vector store and OpenAI embeddings.

It includes resilient fallbacks so the app still responds even when OpenAI
embedding calls fail (it uses a lightweight local TF-IDF-like retriever).

Contents
--------
- `Healthcare_Assistant/` — main example app
	- `data/` — place your source PDFs (and .txt samples are provided)
	- `src/` — application source (Streamlit app, loader, vector store, converter)
	- `.env` — example environment file for your `OPENAI_API_KEY`
	- `requirements.txt` — Python dependencies used during development

Quick overview
--------------
1. Add healthcare PDFs to `Healthcare_Assistant/data/` (diseases, treatments,
 	 wellness, etc.).
2. (Optional) Convert provided `.txt` samples to PDF with the included
 	 converter: `src/convert_to_pdf.py`.
3. Create and activate a Python virtual environment and install
 	 dependencies.
4. Provide your `OPENAI_API_KEY` via environment variable or paste it in the app sidebar.
5. Run the Streamlit app: it will build the vector index and provide a chat
 	 interface for healthcare questions.

Architecture (high level)
-------------------------
- Streamlit frontend (`Healthcare_Assistant/src/app.py`) — UI, API-key handling, and orchestration.
- Document loader (`Healthcare_Assistant/src/document_loader.py`) — PDF extraction and chunking.
- Vector store (`Healthcare_Assistant/src/vector_store.py`) — OpenAI embeddings + FAISS primary; TF-IDF fallback for offline or limited environments.
- OpenAI (chat + embeddings) — used for embeddings and generative answers when API key and quota are available.
- Local TF-IDF fallback — pure-Python retriever used when OpenAI embeddings or network are unavailable.

See `Healthcare_Assistant/docs/portal_diagram.svg` for an architectural diagram visualizing the components and data flow.

Detailed setup (Windows PowerShell)
----------------------------------
Open PowerShell and run the following from the repository root `c:\pp\GitHub\GenAI-Usecases`.

1) Create and activate a virtual environment (recommended):

```powershell
# from repository root
python -m venv .venv
& .venv\Scripts\Activate.ps1
```

2) Install dependencies into the activated venv. Use the project `requirements.txt`.

```powershell
# inside the activated venv
pip install -r Healthcare_Assistant\requirements.txt
```

Notes about dependencies
- `faiss-cpu`: prebuilt wheels are used where possible; if installation fails
	for your Python version consider installing a compatible FAISS wheel or
	using the TF-IDF fallback (the app includes a pure-Python fallback).
- `langchain-community` is required for some LangChain embeddings imports.


3) Configure your OpenAI key

You can provide the key in two ways (the app prefers a sidebar override if present):

- Environment variable (recommended for local runs):

```powershell
setx OPENAI_API_KEY "sk-...your key..."
```

- Or, start the app and paste the key into the left sidebar's "Configuration" box. The key is kept only in memory for that Streamlit session and is not written to disk.

4) (Optional) Convert sample `.txt` files shipped in `Healthcare_Assistant/data`
 	 to PDF using the included converter (the app prefers PDFs). Run:

```powershell
& C:/pp/GitHub/GenAI-Usecases/.venv/Scripts/python.exe Healthcare_Assistant/src/convert_to_pdf.py
```

This will generate PDFs next to the `.txt` files.

5) Run the Streamlit app

```powershell
# Use the venv python to run Streamlit so it picks up installed packages
& C:/pp/GitHub/GenAI-Usecases/.venv/Scripts/python.exe -m streamlit run Healthcare_Assistant/src/app.py
```

Open the Local URL printed by Streamlit (usually http://localhost:8501).

How the app works (high level)
-----------------------------
- `src/document_loader.py` reads PDFs from `Healthcare_Assistant/data/`,
	extracts text, and splits into chunks.
- `src/vector_store.py` creates embeddings and a FAISS index. If OpenAI
	embedding calls fail (e.g. network or DNS problems), the class falls back
	to a simple pure-Python TF-IDF-like retriever so the app still answers
	queries using local text similarity.
- `src/app.py` is the Streamlit front-end: it loads the documents (cached),
	performs retrieval, and sends context + user prompt to OpenAI's chat
	completion endpoint for generation.

Adding your own PDFs
--------------------
1. Put PDFs into `Healthcare_Assistant/data/`. Filenames should end with
	 `.pdf` (case-insensitive).
2. Restart the Streamlit app (the loader runs at startup). If you need to
	 re-index without restarting, clear Streamlit's cache (restart is simplest).

Troubleshooting
---------------
- ModuleNotFoundError: `PyPDF2` — install into your venv: `pip install PyPDF2`.
- LangChain import errors about `langchain_community` — install
	`langchain-community` in venv: `pip install -U langchain-community`.
- tiktoken / encoding download DNS failure (e.g. error resolving
	`openaipublic.blob.core.windows.net`) — the app's vector store will fall
	back to a local TF-IDF fallback. You can still ask questions; quality of
	retrieval will be lower than with OpenAI embeddings.
- faiss-cpu installation errors — if FAISS fails to install, the
	TF-IDF fallback will be used automatically. To get full FAISS + OpenAI
	embedding experience, use a Python version with compatible FAISS wheels
	(or install FAISS via conda where supported).

Common commands and checks
--------------------------
- Check venv python path (used in instructions):

```powershell
C:/pp/GitHub/GenAI-Usecases/.venv/Scripts/python.exe -V
```

- Run the converter (converts `.txt` → `.pdf` in `data/`):

```powershell
& C:/pp/GitHub/GenAI-Usecases/.venv/Scripts/python.exe Healthcare_Assistant/src/convert_to_pdf.py
```

- Start Streamlit using the venv Python (recommended):

```powershell
& C:/pp/GitHub/GenAI-Usecases/.venv/Scripts/python.exe -m streamlit run Healthcare_Assistant/src/app.py
```

Development notes and next steps
-------------------------------
- To improve retrieval quality, provide more high-quality source PDFs with
	structured sections and clear headings. Chunking settings are in
	`src/document_loader.py` (defaults set there).
- To show citations in answers, update the retrieval step to return source
	metadata (filename and page number) and include them in the assistant
	response prompt — I can add this if you'd like.
- To export a conversation as PDF, the repository already contains
	`src/convert_to_pdf.py` (used for converting text to PDF), but I can add
	a Streamlit button that generates a PDF of the current chat.

Files you may want to review
----------------------------
- `Healthcare_Assistant/src/app.py` — Streamlit interface and RAG logic
- `Healthcare_Assistant/src/document_loader.py` — PDF extraction and chunking
- `Healthcare_Assistant/src/vector_store.py` — FAISS/OpenAI embeddings
	with TF-IDF fallback (local)
- `Healthcare_Assistant/src/convert_to_pdf.py` — helper: convert `.txt` → `.pdf`
- `Healthcare_Assistant/requirements.txt` — pinned dependency list used
	when I set up the project in this workspace

For a step-by-step user guide, sample questions & model-style answers,
see `Healthcare_Assistant/USAGE.md`.

If anything errors when you run these steps, copy the full traceback or
error text here and I'll help debug it. If you'd like I can also:
- add citation metadata and surface it in responses,
- add an export-to-PDF button in the UI,
- or package the app into a small Docker image for easier deployment.

Enjoy exploring the Healthcare Assistant RAG demo — tell me which
improvement you'd like next and I'll implement it.
# GenAI-Usecases
