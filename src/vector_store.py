import os 
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import numpy as np

class VectorStore:
    def __init__(self, embedding_model, persist_path: str):
        self.persist_path = persist_path
        self.db = None
        self.embedding_model = embedding_model
    
    def build(self, documents: List[Document]):
        print(f"Building vector store with {len(documents)} documents...")
        
        # For custom embeddings, we need to create the FAISS index manually
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Create embeddings
        embeddings = self.embedding_model.embed_documents(texts)
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Create FAISS index
        import faiss
        dimension = len(embeddings[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)
        
        # Create the FAISS vector store
        self.db = FAISS.from_embeddings(
            text_embeddings=list(zip(texts, embeddings)),
            embedding=self.embedding_model,
            metadatas=metadatas
        )
        
        print("Vector store built successfully!")
        return self.db
    
    def save(self):
        if self.db:
            self.db.save_local(self.persist_path)
            print(f"Vector store saved to {self.persist_path}")
        else:
            raise ValueError("No vector store to save.")
    
    def load(self):
        if not os.path.exists(self.persist_path):
            raise FileNotFoundError(f"Vector store not found at: {self.persist_path}")
        
        self.db = FAISS.load_local(
            folder_path=self.persist_path,
            embeddings=self.embedding_model,
            allow_dangerous_deserialization=True
        )
        print(f"Vector store loaded from {self.persist_path}")
        return self.db
    
    def similarity_search(self, query, k=4):
        if not self.db:
            raise ValueError("Vector store not initialized. Call build() or load() first.")
        return self.db.similarity_search(query=query, k=k)