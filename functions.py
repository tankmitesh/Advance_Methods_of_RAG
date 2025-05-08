import os, PyPDF2
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Local environment variables
load_dotenv()


def generation_model(model="gpt-3.5-turbo", max_tokens=2000):
    return ChatOpenAI(model=model, openai_api_key=os.getenv("OPENAI_API_KEY"), max_tokens=max_tokens)


def embedding_model(model="text-embedding-3-small"):
    return OpenAIEmbeddings(model=model, openai_api_key=os.getenv("OPENAI_API_KEY"))


def vector_store():
    # FAISS does not require a persist directory like Chroma
    return FAISS.load_local("faiss_index", embedding_model(), allow_dangerous_deserialization=True)


def text_splitter(chunk_size=1000, chunk_overlap=200):
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def indexing_storing(context):
    """Load PDF file and store in vector database"""

    # Create Document
    documents = [Document(page_content=context)]

    # Split text into chunks
    texts = text_splitter().split_documents(documents)

    # Store chunks in vector database as vectors
    vector_db = FAISS.from_documents(documents=texts, embedding=embedding_model())

    # Save FAISS index locally
    vector_db.save_local("faiss_index")

    return texts


def context_extraction(file):
    # Load PDF file
    pdf_reader = PyPDF2.PdfReader(file)
    pdf_pages = pdf_reader.pages

    # Create Context
    context = "\n\n".join(page.extract_text() for page in pdf_pages)

    return context