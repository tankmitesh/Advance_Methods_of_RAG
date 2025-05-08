from langchain.prompts import PromptTemplate
from functions import generation_model

###############################################################################################################################

class HydeRetriver:
    def __init__(self) :
        self.chink_size = 500
        self.llm = generation_model(model = "gpt-3.5-turbo")

    def hyde_prompt(self) :

        template = """Given the question '{query}', generate a hypothetical document that directly answers this question. 
                   The document should be detailed and in-depth.the document size has be exactly {chunk_size} characters."""
        
        return PromptTemplate(input_variables = ["query", "chunk_size"], template = template)
    
    def hy_document_generation(self, query) :
        
        # get prompt
        prompt_template = self.hyde_prompt()

        # Combine prompt and llm model
        llm_chain = prompt_template | self.llm

        # Generate fake document
        return llm_chain.invoke({"query" : query, "chunk_size" : self.chink_size}).content.strip()



        

