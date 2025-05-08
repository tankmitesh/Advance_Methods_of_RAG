# Third Party packages
from langchain.prompts import PromptTemplate

# Custom packages
from functions import generation_model

##############################################################################################################################

def query_rewriter(query):

    """Make queries more specific and detailed, improving the probability of retrieving the most relevant information"""
        
    template = """
                You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system. 
                Given the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information.

                Original query: {original_query}

                Rewritten query:
                """
    
    prompt = PromptTemplate(input_variables = ["original_query"], template = template)

    # LLM Model
    llm = generation_model(model = "gpt-3.5-turbo")

    generator = prompt | llm

    return generator.invoke(query).content.strip()


##############################################################################################################################

def query_stepback(query):

    """Generate broader and more general queries that can help retrieve relevant background information"""
        
    template = """
                You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system. 
                Given the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information.

                Original query: {original_query}

                Rewritten query:
                """
    
    prompt = PromptTemplate(input_variables = ["original_query"], template = template)

    # LLM Model
    llm = generation_model(model = "gpt-3.5-turbo")

    generator = prompt | llm

    return generator.invoke(query).content.strip()

##############################################################################################################################

def query_decomposer(query):

    """Decompose complex queries into simpler sub-queries to improve retrieval in a RAG system"""

    template = """
                You are an AI assistant tasked with breaking down complex queries into simpler sub-queries for a RAG system.
                Given the original query, decompose it into 2-4 simpler sub-queries that, when answered together, would provide a comprehensive response to the original query.

                Original query: {original_query}

                example: What are the impacts of climate change on the environment?

                Sub-queries:
                1. What are the impacts of climate change on biodiversity?
                2. How does climate change affect the oceans?
                3. What are the effects of climate change on agriculture?
                4. What are the impacts of climate change on human health?
                """
    
    prompt = PromptTemplate(input_variables = ["original_query"], template = template)

    # LLM Model
    llm = generation_model(model = "gpt-3.5-turbo")

    generator = prompt | llm

    query_answer = generator.invoke(query).content.strip()
    query_answer = query_answer.replace("Sub-queries:", "")
    
    return query_answer






