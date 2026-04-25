import os
from typing import List
from langchain_core.embeddings import Embeddings as BaseEmbeddings
from llama_cpp import Llama

class Embeddings(BaseEmbeddings):
    """Custom embedding class that works with GGUF embedding models"""
    
    def __init__(self, config, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Embedding model file not found: {model_path}")
        
        print(f"Loading embedding model from: {model_path}")
        
        # Initialize the llama model in embedding mode
        self.client = Llama(
            model_path=model_path,
            n_ctx=512,  # Smaller context for embeddings
            n_threads=8,
            embedding=True,  # Critical: Enable embedding mode
            verbose=False,
            n_batch=512,
        )
        print("Embedding model loaded successfully!")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        embeddings = []
        total = len(texts)
        
        print(f"Creating embeddings for {total} chunks...")
        
        for i, text in enumerate(texts):
            if i % 50 == 0:
                print(f"  Progress: {i}/{total}")
            
            # Truncate text to fit within context window
            if len(text) > 2000:
                text = text[:2000]
            
            try:
                # Create embedding for this text
                embedding = self.client.embed(text)
                embeddings.append(embedding)
            except Exception as e:
                print(f"Warning: Failed to embed chunk {i}: {e}")
                # Return zero vector as fallback
                embeddings.append([0.0] * 768)
        
        print(f"Completed! Created {len(embeddings)} embeddings")
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        if len(text) > 2000:
            text = text[:2000]
        
        try:
            return self.client.embed(text)
        except Exception as e:
            print(f"Warning: Failed to embed query: {e}")
            return [0.0] * 768