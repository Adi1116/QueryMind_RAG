import os
import streamlit as st
from io import BytesIO
from langchain_community.vectorstores import Chroma  # Updated import
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.llms import Cohere
from langchain.chains import RetrievalQA

# Set the environment variable for Cohere API Key
os.environ["COHERE_API_KEY"] = 'Ox97SolGnL68xrDjbNAMiVaWCqZ5Fny3d7hYAub6'

# Streamlit app
st.title("PDF-based Document Q&A")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# If PDF is uploaded
if uploaded_file is not None:
    # Convert the uploaded file to a BytesIO object (works as a file-like object)
    pdf_file = BytesIO(uploaded_file.read())

    # Load the PDF using PyPDFLoader
    loader = PyPDFLoader(pdf_file)
    docs = loader.load()

    # Split the document into chunks
    text_splitter = CharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=50
    )
    docs_split = text_splitter.split_documents(docs)
    
    # Initialize Cohere embeddings and Chroma VectorStore
    embeddings = CohereEmbeddings(model="embed-english-v2.0")  # Use correct model version
    vectordb = Chroma.from_documents(documents=docs_split, embedding=embeddings)

    # Create a retriever from the VectorStore
    retriever = vectordb.as_retriever()

    # Initialize the LLM (Cohere-based) and RetrievalQA chain
    llm = Cohere(model="command-xlarge-2023")  # Cohere LLM for Q&A
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Using simple "stuff" strategy for retrieved docs
        retriever=retriever
    )

    # Initialize Streamlit chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Get user input (query)
    query = st.text_input("Enter your question:")

    # If user submits a query
    if query:
        # Use the agent to generate a response based on the PDF
        response = rag_chain.run(query)

        # Store question and response in the chat history
        st.session_state.chat_history.append((query, response))

    # Display chat history
    if st.session_state.chat_history:
        st.write("### Chat History")
        for idx, (question, answer) in enumerate(st.session_state.chat_history):
            st.write(f"**Q{idx+1}:** {question}")
            st.write(f"**A{idx+1}:** {answer}")
