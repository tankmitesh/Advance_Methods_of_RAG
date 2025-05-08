from langchain_core.prompts import ChatPromptTemplate


SYSTEM_PROMPT = ChatPromptTemplate.from_template(
                """Use the following pieces of context to answer the question at the end. 
                If you don't know the answer, just say that you don't know, don't try to make up an answer.
                Keep the answer concise and limited to three sentences.

                Context:
                {context}

                Question: {question}

                Answer:"""
            )


DENSE_INFORMATION_PROMPT = """Process the provided context '{user_context}' in a RAG pipeline to generate a high-density output:

                            Clean Data: Remove irrelevant or redundant content.
                            Deduplicate: Consolidate overlapping information.
                            Abstractively Summarize: Write a 50–100 word summary in precise, rephrased language.
                            Prioritize Facts: List 2–4 key facts with confidence scores (0–100).
                            Tags/Categories: Generate 3–5 tags and 1–2 categories.

                        Output:

                            Summary: 50–100 words, high-density.
                            Facts: 2–4 bullets with scores.
                            Tags: 3–5, comma-separated.
                            Categories: 1–2, comma-separated.

                        Ensure concise, clear, and relevant output. Note ambiguities and suggest clarifications if needed."""


CONTEXT_CHUNK_HEADER_PROMPT = """
                                INSTRUCTIONS
                                What is the title of the following document?

                                Your response MUST be the title of the document, and nothing else. DO NOT respond with anything else.

                                NOTE : Also note that the document text provided below is just the first ~few praragraph words of the document. 
                                       That should be plenty for this task. 
                                       Your response should still pertain to the entire document, not just the text provided below.

                                Document text:
                                {user_context}
                                
                            """.strip()