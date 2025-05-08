from langchain.docstore.document import Document
from typing import List
from rank_bm25 import BM25Okapi
import numpy as np
from langchain.schema import BaseRetriever

##############################################################################################################################x

def create_bm25_index(documents: List[Document]) -> BM25Okapi:
    # Tokenize each document by splitting on whitespace
    # This is a simple approach and could be improved with more sophisticated tokenization
    tokenized_docs = [doc.page_content.split() for doc in documents]
    return BM25Okapi(tokenized_docs)


def fusion_retrieval(vectorstore, bm25, query: str, k: int = 5, alpha: float = 0.5) -> List[Document]:
    
    epsilon = 1e-8

    # Step 1: Get all documents from the vectorstore
    all_docs = vectorstore.similarity_search("", k=vectorstore.index.ntotal)

    # Step 2: Perform BM25 search
    bm25_scores = bm25.get_scores(query.split())

    # Step 3: Perform vector search
    vector_results = vectorstore.similarity_search_with_score(query, k=len(all_docs))
    
    # Step 4: Normalize scores
    vector_scores = np.array([score for _, score in vector_results])
    vector_scores = 1 - (vector_scores - np.min(vector_scores)) / (np.max(vector_scores) - np.min(vector_scores) + epsilon)

    bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) -  np.min(bm25_scores) + epsilon)

    # Step 5: Combine scores
    combined_scores = alpha * vector_scores + (1 - alpha) * bm25_scores  

    # Step 6: Rank documents
    sorted_indices = np.argsort(combined_scores)[::-1]
    
    # Step 7: Return top k documents
    return [all_docs[i] for i in sorted_indices[:k]]



class FusionRetriever(BaseRetriever):
    def __init__(self, vector_db, bm25_index, alpha=0.5, k=5):
        self.vector_db = vector_db
        self.bm25_index = bm25_index
        self.alpha = alpha
        self.k = k
        
    def get_relevant_documents(self, query):
        return fusion_retrieval(self.vector_db, self.bm25_index, query, k=self.k, alpha=self.alpha)

    async def aget_relevant_documents(self, query):
        return self.get_relevant_documents(query)

