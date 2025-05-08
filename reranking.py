from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.docstore.document import Document
from typing import List, Any
from langchain_core.prompts import PromptTemplate
from functions import generation_model
from langchain_core.retrievers import BaseRetriever


class RatingScore(BaseModel):
    relevance_score: float = Field(..., description="The relevance score of a document to a query.")

def rerank_documents(query: str, docs: List[Document], top_n: int = 3) -> List[Document]:
    prompt_template = PromptTemplate(
        input_variables=["query", "doc"],
        template="""On a scale of 1-10, rate the relevance of the following document to the query. Consider the specific context and intent of the query, not just keyword matches.
        Query: {query}
        Document: {doc}
        Relevance Score:"""
    )
    
    llm = generation_model(model = "gpt-4o", max_tokens = 4000)
    llm_chain = prompt_template | llm.with_structured_output(RatingScore)
    
    scored_docs = []
    for doc in docs:
        input_data = {"query": query, "doc": doc.page_content}
        score = llm_chain.invoke(input_data).relevance_score
        try:
            score = float(score)
        except ValueError:
            score = 0  # Default score if parsing fails
        scored_docs.append((doc, score))
    
    reranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in reranked_docs[:top_n]]


# Create a custom retriever class
class CustomRetriever(BaseRetriever, BaseModel):
    
    vectorstore: Any = Field(description="Vector store for initial retrieval", alias="vector_store")

    class Config:
        arbitrary_types_allowed = True

    def get_relevant_documents(self, query: str, num_docs=2) -> List[Document]:
        initial_docs = self.vectorstore.similarity_search(query, k=10)
        return rerank_documents(query, initial_docs, top_n=num_docs)