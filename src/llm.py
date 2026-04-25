from langchain_ollama import ChatOllama

from utils.config import Config


class LLM:
    def __init__(self, config: Config, model_name: str="qwen2.5-coder:0.5b"):
        self.llm = ChatOllama(
            model=model_name,
            temperature=config.temperature,
            top_p=config.top_p,
        )
    def get(self):
        return self.llm
    