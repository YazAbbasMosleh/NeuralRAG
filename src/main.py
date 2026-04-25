import logging



from utils.config import Config
from src.document_processor import DocumentProcessor
from src.embeddings import Embeddings
from src.vector_store import VectorStore
from src.llm import LLM
from src.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def main():
    config = Config("config.yaml")
    
    
    # Load + Split PDF
    processor = DocumentProcessor(chunking_method="recursive_character",
                                  chunk_size=config.chunk_size,
                                  chunk_overlap=config.chunk_overlap)
    docs = processor.load_pdf(pdf_path=config.pdf_path)
    chunks = processor.split_document(docs)
    
    logger.info(f"Loaded {len(docs)} documents and split into {len(chunks)} chunks.")
    
    # Embedding
    embedding_model = Embeddings(config=config, model_path=config.embedding_model_path)

    
    
    # Vector Store
    vector_store = VectorStore(embedding_model=embedding_model,
                               persist_path=config.vector_store_path)
    
    db = vector_store.build(chunks)
    vector_store.save()
    
    # llm
    llm_model = LLM(config=config, model_name=config.llm_model)
    llm = llm_model.get()
    
    # RAG Pipeline
    rag_pipeline = RAGPipeline(llm=llm, vector_store=vector_store)
    
    # 6. Chat loop
    logger.info("\n🚀 RAG system ready. Ask questions!\n")
    

    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break

        answer = rag_pipeline.run(query)
        logger.info(f"\nAssistant: {answer}")
        logger.info("-" * 60)
if __name__ == "__main__":
    main()