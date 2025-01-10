import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os

# Set page configuration for better layout
st.set_page_config(page_title="RAG Chatbot", layout="wide")

# Add custom CSS for styling
st.markdown("""
    <style>
        .main {
            padding: 2rem;
            background-color: #f0f4f8;
        }
        .sidebar .sidebar-content {
            background-color: #ffffff;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            padding: 12px 20px;
            border-radius: 8px;
            width: 100%;
        }
        .stTextInput>div>input {
            font-size: 16px;
            padding: 12px;
            border-radius: 8px;
            width: 100%;
            background-color: #f9f9f9;
        }
        .stTextInput label {
            font-weight: bold;
        }
        .stHeader {
            color: #333;
        }
    </style>
""", unsafe_allow_html=True)

# Introduction with attractive layout and markdown text
st.markdown("""
## RAG Chatbot: Get Instant Insights from Your Documents 📄🤖

Welcome to **RAG Chatbot**! This chatbot is powered by Google's Generative AI model **Gemini-PRO**, using the Retrieval-Augmented Generation (RAG) framework. It processes your uploaded PDFs and provides intelligent, context-aware answers to any question you ask. Here's how you can get started:

1. **Enter Your Google API Key**: To interact with the AI, you'll need a valid API key. [Get your API key here](https://makersuite.google.com/app/apikey).
2. **Upload Your Documents**: Upload one or more PDF files for processing.
3. **Ask a Question**: After processing, simply ask a question related to your documents, and receive accurate, contextually relevant answers.

Get started by entering your API key and uploading your PDFs below!
""", unsafe_allow_html=True)

# API Key input
api_key = st.text_input("Enter your Google API Key:", type="password", key="api_key_input")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not available, just say, "answer is not available in the context".
    
    Context: {context}
    Question: {question}
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("**Answer:**", response["output_text"])

def main():
    st.header("AI Clone Chatbot 💬")

    # User input for asking questions
    user_question = st.text_input("Ask a Question from the PDF Files", key="user_question", label_visibility="collapsed")

    if user_question and api_key:  # Ensure API key and user question are provided
        user_input(user_question, api_key)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True, key="pdf_uploader")
        if st.button("Submit & Process", key="process_button") and api_key:  # Check if API key is provided before processing
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks, api_key)
                st.success("Document processed successfully! You can now ask questions.")

if __name__ == "__main__":
    main()
