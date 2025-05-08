# Advanced Methods of RAG (Retrieval-Augmented Generation)

## Overview
This project implements advanced methods for Retrieval-Augmented Generation (RAG) pipelines. It combines various techniques to enhance information retrieval, query transformation, and response generation using large language models (LLMs). The project is designed to process and retrieve information from documents, particularly PDFs, and provide concise, accurate answers to user queries.

## Features
- **Query Transformation**: Includes methods like query rewriting, stepping back, and decomposition to improve retrieval accuracy.
- **Context Enhancement**: Supports dense information retrieval and context chunk header generation for better document processing.
- **Advanced Retrieval**: Implements reranking and fusion ranking techniques to improve the relevance of retrieved documents.
- **PDF Processing**: Extracts and processes text from PDF files for embedding and retrieval.
- **Streamlit Interface**: Provides a user-friendly interface for uploading files, entering queries, and selecting retrieval methods.

## Key Components
- **`app.py`**: The main application file that integrates all components and provides a Streamlit-based interface.
- **`functions.py`**: Contains utility functions for embedding, vector storage, and text splitting.
- **`query_transformation.py`**: Implements query rewriting, stepping back, and decomposition.
- **`dense_information_retrieval.py`**: Processes context to generate high-density outputs.
- **`fusion_retrival.py`**: Combines vector-based and BM25-based retrieval for improved results.
- **`reranking.py`**: Reranks documents based on relevance scores.
- **`hyde.py`**: Generates hypothetical documents to enhance retrieval.
- **`context_chunk_header.py`**: Generates headers for context chunks to improve document organization.
- **`prompt.py`**: Defines prompt templates for various tasks.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Advance_Methods_of_RAG
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
2. Upload a PDF file using the sidebar.
3. Select query enhancement and context enhancement methods.
4. Enter your query and view the generated response.

## Dependencies
- Python 3.12+
- Streamlit
- LangChain
- OpenAI API
- ChromaDB
- FAISS
- PyPDF2
- Rank-BM25

## Project Structure
- **`app.py`**: Main application file.
- **`functions.py`**: Utility functions.
- **`query_transformation.py`**: Query transformation methods.
- **`dense_information_retrieval.py`**: Dense information retrieval.
- **`fusion_retrival.py`**: Fusion retrieval methods.
- **`reranking.py`**: Document reranking.
- **`hyde.py`**: Hypothetical document generation.
- **`context_chunk_header.py`**: Context chunk header generation.
- **`prompt.py`**: Prompt templates.
- **`pyproject.toml`**: Project metadata and dependencies.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- OpenAI for GPT models.
- LangChain for retrieval and prompt management.
- Streamlit for the user interface.