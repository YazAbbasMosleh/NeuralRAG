import os
from typing import List


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document

# from config import Config


class DocumentProcessor:
    def __init__(self, chunking_method: str="recursive_character",
                 chunk_size: int=1000, chunk_overlap: int=200):
        
        assert chunking_method in ["recursive_character", "character", "semantic"], "Invalid chunking method."
        self.chunking_method = chunking_method
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        if chunking_method == "recursive_character":
            self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                           chunk_overlap=chunk_overlap)
        
    def load_pdf(self, pdf_path: str):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        if not pdf_path.endswith(".pdf"):
            raise ValueError(f"PDF path must end with .pdf: {pdf_path}")
        
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        if documents is None:
            raise ValueError(f"Failed to load PDF document: {pdf_path}")
        return documents
    def split_document(self, documents: List[Document]):
        if not documents:
            raise ValueError("No documents to split.")
        return self.splitter.split_documents(documents)