import os
from typing import List, Dict
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentLoader:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def load_pdfs(self, directory: str) -> List[Dict]:
        """Load PDFs from a directory and return list of text chunks with metadata.

        Each chunk is a dict: {"text": str, "meta": {"source": filename, "page": page_number}}
        """
        text_chunks = []

        # If directory doesn't exist, return empty list instead of throwing
        if not os.path.exists(directory):
            return text_chunks

        for filename in os.listdir(directory):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(directory, filename)
                try:
                    pdf = PdfReader(pdf_path)
                except Exception:
                    # skip files that can't be read as PDFs
                    continue

                # Extract text from each page and split per-page so we can
                # attach page-level metadata to each chunk.
                for page_idx, page in enumerate(pdf.pages, start=1):
                    try:
                        page_text = page.extract_text() or ""
                    except Exception:
                        page_text = ""
                    if not page_text.strip():
                        continue

                    # Split the page text into chunks and attach metadata
                    chunks = self.text_splitter.split_text(page_text)
                    for c in chunks:
                        text_chunks.append({
                            "text": c,
                            "meta": {"source": filename, "page": page_idx}
                        })

        return text_chunks