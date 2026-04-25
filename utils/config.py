import os
import yaml


class Config:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)    
        self.llm_model = self.config.get("llm_model", "qwen2.5-coder:0.5b")
        self.embedding_model_path = self.config.get("embedding_model_path", "embedding_models/nomic-embed-text-v2.gguf")
        if not self.embedding_model_path.endswith(".gguf"):
            raise ValueError(f"Embedding model path must end with .gguf: {self.embedding_model_path}")
        
        self.pdf_path = self.config.get("pdf_path", "./data/Groking_Algorithms.pdf")
        if not self.pdf_path.endswith(".pdf"):
            raise ValueError(f"PDF path must end with .pdf: {self.pdf_path}")
        
        self.vector_store_path = self.config.get("vector_store_path", "./vector_store")
        self.chunk_size = self.config.get("chunk_size", 1000)
        self.chunk_overlap = self.config.get("chunk_overlap", 200)
        
        self.temperature = self.config.get("temperature", 0.7)
        if self.temperature < 0 or self.temperature > 1:
            raise ValueError(f"Temperature must be between 0 and 1: {self.temperature}")
        
        self.max_tokens = self.config.get("max_tokens", 2048)
        self.top_p = self.config.get("top_p", 0.9)
        if self.top_p < 0 or self.top_p > 1:
            raise ValueError(f"Top-p must be between 0 and 1: {self.top_p}")
        self.n_ctx = self.config.get("n_ctx", 2048)
        if self.n_ctx < 0:
            raise ValueError(f"n_ctx value should be a positive number.")
    def _load_config(self, config_path: str):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, "r") as file:
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found: {config_path}")
            with open(config_path, "r") as file:
                try:
                    config = yaml.safe_load(file)
                    return config
                except yaml.YAMLError as e:
                    raise ValueError(f"Error parsing YAML config: {e}")