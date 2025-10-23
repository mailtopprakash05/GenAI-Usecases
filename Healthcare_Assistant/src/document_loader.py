import os
from typing import List
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentLoader:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def load_pdfs(self, directory: str) -> List[str]:
        """Load PDFs from a directory and return list of text chunks."""
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

                # Extract text from each page
                text = ''
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text

                # Split text into chunks
                if text:
                    chunks = self.text_splitter.split_text(text)
                    text_chunks.extend(chunks)

        return text_chunks