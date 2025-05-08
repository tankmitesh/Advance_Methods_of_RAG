# Inbuild packages
import os, shutil 
import warnings
warnings.filterwarnings("ignore")

# Third party packages
import streamlit as st
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


from functions import (vector_store, generation_model, indexing_storing, context_extraction)
from query_transformation import query_rewriter, query_stepback, query_decomposer
from hyde import HydeRetriver
from dense_information_retrieval import DenseInformationRetrieval
from context_chunk_header import ContextChunkHeader
# from reranking import rerank_documents, CustomRetriever
from fusion_retrival import create_bm25_index, fusion_retrieval, FusionRetriever
from prompt import SYSTEM_PROMPT 

#############################################################################################################################

st.title("PDF Question Answering System")

# ------------- Load input files and Text -------------
file = st.sidebar.file_uploader("Upload your input files", type=["pdf"], key="input_files")
method_select = st.sidebar.selectbox("Query enhancement methods", ["None",  
                                                                    "Query Rewrite",
                                                                    "Query Step Back",
                                                                    "Query Decompose",
                                                                    "HyDe"], key = "method_select")

pre_retrieval_select = st.sidebar.selectbox("Context enhancement methods", ["None", 
                                                                            "Dense Information Retrieval",
                                                                            "Context Chunk Header"], key = "pre_retrieval")

advance_select = st.sidebar.selectbox("Advanced methods", ["None", "Reranking", "Fusion Ranking"], key = "advance_select")


# Clear cache button
st.sidebar.markdown("<div style='margin-top: auto; height: 350context_chunk_header_retrievalpx;'></div>", unsafe_allow_html=True)
if st.sidebar.button("Clear Cache", key="clear_cache") :
    if os.path.exists("faiss_index") :
        shutil.rmtree("faiss_index")
        st.success("Cache cleared successfully!")
    else :
        st.warning("No cache to clear!")

# User query
user_query = st.text_input("Enter your query", key="user_query")


if file is not None :
    with st.spinner("Processing..."):

        if os.path.exists("faiss_index") :
            vector_db = vector_store()

        else :

            # one more step to extract context
            context = context_extraction(file)

            if pre_retrieval_select == "Dense Information Retrieval" :
                context = DenseInformationRetrieval().dense_information_retrieval(context)
                splitted_texts = indexing_storing(context)
                vector_db = vector_store()

            elif pre_retrieval_select == "Context Chunk Header" :
                context = ContextChunkHeader().context_chunk_header_retrieval(context)
                splitted_texts = indexing_storing(context)
                vector_db = vector_store()

            else :
                splitted_texts = indexing_storing(context)
                vector_db = vector_store()


    st.sidebar.success("File processed successfully!")


if file is not None and user_query:
    with st.spinner("Generating response..."):

        # Load LLM
        llm = generation_model(model = "gpt-4o-mini", max_tokens = 4000)

        if advance_select == "Reranking" :
            # Create custom retriever
            retriver = CustomRetriever(vectorstore = vector_db).as_retriever(search_kwargs={"k": 5}),         

        elif advance_select == "Fusion Ranking" :
            # Create BM25 index
            bm25_index = create_bm25_index(splitted_texts)
            # Perform fusion retrieval
            retriver = FusionRetriever(vector_db = vector_db, bm25_index = bm25_index, alpha = 0.5, k = 5)

        else :
            retriver = vector_db.as_retriever(search_kwargs={"k": 5})

                    
        # Create RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(llm = llm,
                                            chain_type = "stuff",
                                            retriever = retriver,
                                            return_source_documents=True)
        

        st.write(qa_chain)
            
        # Create RAG chain
        rag_chain = (
            {"context": qa_chain, "question": RunnablePassthrough()}
            | SYSTEM_PROMPT
            | llm
            | StrOutputParser())
        
        # Rewrite query
        if method_select == "Query Rewrite" :
            user_query = query_rewriter(user_query)

        elif method_select == "Query Step Back" :
            user_query = query_stepback(user_query)

        elif method_select == "Query Decompose" :
            user_query = query_decomposer(user_query)

        elif method_select == "HyDe" :
            user_query = HydeRetriver().hy_document_generation(user_query)

    
        answer = rag_chain.invoke(user_query)

        # Display answer
        st.subheader("Answer:")
        st.write(answer)



















    




