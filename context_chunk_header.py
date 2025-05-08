from functions import generation_model
from langchain.prompts import PromptTemplate
import tiktoken
from prompt import CONTEXT_CHUNK_HEADER_PROMPT

############################################################################################################

class ContextChunkHeader :

    """Class to handle context chunking and header generation."""
    
    def __init__(self) :
        self.llm = generation_model(model = "gpt-3.5-turbo")

    def context_extraction(self, file_context) :
        return file_context.split("\n\n")
    
    def contect_chunk_header_prompt(self) :
        return PromptTemplate(input_variables = ['user_context'], template = CONTEXT_CHUNK_HEADER_PROMPT)
    
    def truncate_text(self, text, max_tokens) :
        """Truncate text to fit within the token limit."""
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        tokens = encoding.encode(text)
        if len(tokens) > max_tokens :
            tokens = tokens[:max_tokens]
        return encoding.decode(tokens)
    
    def context_chunk_header_retrieval(self, file_context, token_limit = 2000) :

        # Extract context from the file
        context_list = self.context_extraction(file_context)

        # Use the prompt template to format the context
        prompt = self.contect_chunk_header_prompt()

        dense_information_context = ""
        for xpage in context_list :

            # Combine prompt with llm model
            llm_chain = prompt | self.llm

            # Truncate the text to fit within the token limit
            xpage = self.truncate_text(xpage, token_limit)

            # Generate dense information
            chunk_header = llm_chain.invoke([xpage]).content.strip()

            # Merge chunk header with the original context
            dense_information_context += f"Chunk Header: {chunk_header}\nContext: {xpage}\n\n"

        return dense_information_context




