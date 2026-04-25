from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


class RAGPipeline:
    def __init__(self, llm, vector_store):
        self.llm = llm
        self.vector_store = vector_store

        self.prompt = PromptTemplate.from_template("""
You are a helpful assistant.

Use ONLY the context below to answer the question.

Context:
{context}

Question:
{question}

Answer clearly and concisely:
""")

        self.chain = self.prompt | self.llm | StrOutputParser()

    def retrieve_context(self, query: str):
        docs = self.vector_store.similarity_search(query, k=4)
        return "\n\n".join([d.page_content for d in docs])

    def run(self, query: str):
        context = self.retrieve_context(query)

        response = self.chain.invoke({
            "context": context,
            "question": query
        })

        return response