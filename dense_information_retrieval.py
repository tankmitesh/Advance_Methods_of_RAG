from functions import generation_model
from langchain.prompts import PromptTemplate

############################################################################################################

class DenseInformationRetrieval:
    def __init__(self) :
        self.llm = generation_model(model = "gpt-3.5-turbo")

    def context_extraction(self, file_context) :
        return file_context.split("\n\n")
    
    def dense_information_prompt(self) :

        template = """Process the provided context '{user_context}' in a RAG pipeline to generate a high-density output:

                                    Clean Data: Remove irrelevant or redundant content.
                                    Deduplicate: Remove duplicate information.
                                    Abstractively Summarize: Write a 50–100 word summary in precise, rephrased language.
                                    Prioritize Facts: List 2–4 key facts with confidence scores (0–100).
                                    Tags/Categories: Generate 3–5 tags and 1–2 categories.

                                Output:

                                    Summary: 50–100 words, high-density.
                                    Facts: 2–4 bullets with scores.
                                    Tags: 3–5, comma-separated.
                                    Categories: 1–2, comma-separated.

                                Ensure concise, clear, and relevant output. Note ambiguities and suggest clarifications if needed."""

        return PromptTemplate(input_variables = ['user_context'], template = template)
    
    def dense_information_retrieval(self, file_context) :
        # Extract context from the file
        context_list = self.context_extraction(file_context)

        # Use the prompt template to format the context
        prompt = self.dense_information_prompt()

        dense_information_context = ""
        for xpage in context_list :
            # Combine prompt with llm model
            llm_chain = prompt | self.llm
            # Generate dense information
            dense_information_context += "\n\n" + llm_chain.invoke([xpage]).content.strip()

        return dense_information_context




