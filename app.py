import os
import streamlit as st
from streamlit_chat import message
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import requests
from audio_recorder_streamlit import audio_recorder
import json
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool, AgentType

# Load environment variables and set the API key
load_dotenv()
os.environ["COHERE_API_KEY"] = 'Ox97SolGnL68xrDjbNAMiVaWCqZ5Fny3d7hYAub6'
os.environ['API_KEY'] = "b1afee3b-c36c-4abf-8c35-5aeec8cba897"

# Document Preprocessing
@st.cache_data
def doc_preprocessing():
    loader = DirectoryLoader(
        'data/',
        glob='**/*.pdf',     # only the PDFs
        show_progress=True
    )
    docs = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=0
    )
    docs_split = text_splitter.split_documents(docs)
    return docs_split

# Embeddings Store
@st.cache_resource
def embeddings_store():
    embedding = CohereEmbeddings(model="embed-english-v3.0")
    texts = doc_preprocessing()
    persist_directory = 'db'
    vectordb = Chroma.from_documents(documents=texts,
                                     embedding=embedding,
                                     persist_directory=persist_directory)
    retriever = vectordb.as_retriever()
    return retriever

# Conversational Retrieval Chain
@st.cache_resource
def search_db():
    retriever = embeddings_store()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    
    # Conversational Retrieval Chain
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatCohere(),
        memory=memory,
        retriever=retriever
    )
    return qa

# RAG Agent integration
@st.cache_resource
def rag_agent():
    retriever = embeddings_store()
    
    # Create a RAG chain
    rag_chain = RetrievalQA.from_chain_type(
        llm=ChatCohere(),
        chain_type="stuff",  # Use the "stuff" method for chaining
        retriever=retriever
    )

    # Define a tool for querying the RAG chain
    rag_tool = Tool(
        name="DocumentQA",
        func=rag_chain.run,
        description="Use this tool to answer questions by retrieving and using context from documents."
    )

    # Initialize an agent with the RAG tool
    agent = initialize_agent(
        tools=[rag_tool],
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        llm=ChatCohere(),
        verbose=True
    )

    return agent
os.environ['API_KEY'] = "b1afee3b-c36c-4abf-8c35-5aeec8cba897"
# Audio transcription using Sarvam API
def transcribe_audio(audio_file_path):
    with open(audio_file_path, 'rb') as audio_file:
        files = {
            'file': ('test.wav', audio_file, 'audio/wav'),
        }
        data = {
            'prompt': '<string>',
            'model': 'saaras:v1'
        }
        url = "https://api.sarvam.ai/speech-to-text-translate"
        headers = {
            "API-Subscription-Key": os.environ['API_KEY'],
        }

        response = requests.post(url, files=files, data=data, headers=headers)
      

# Assuming response.text contains the JSON string
        data = json.loads(response.text)

# Now you can extract the transcript
        transcript = data["transcript"]
        return transcript

# Display conversation history using Streamlit messages
def display_conversation(history):
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=str(i) + "_user")
        message(history["generated"][i], key=str(i))

# Main Streamlit function
def main_f():
    st.title("LLM Powered Chatbot with Audio Input")

    # Initialize RAG agent
    rag_agent_instance = rag_agent()
    
    # Initialize session state for generated responses and past messages
    if "generated" not in st.session_state:
        st.session_state["generated"] = ["I am ready to help you"]
    if "past" not in st.session_state:
        st.session_state["past"] = ["Hey there!"]

    # Record audio
    audio_bytes = audio_recorder()

    # If audio is recorded, save and transcribe it
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")

        # Save the audio to a .wav file
        audio_file_path = "recorded_audio.wav"
        with open(audio_file_path, "wb") as f:
            f.write(audio_bytes)

        st.success("Audio recorded and saved as 'recorded_audio.wav'.")

        # Transcribe the audio
        with st.spinner("Transcribing audio..."):
            transcribed_text = transcribe_audio(audio_file_path)
            st.write(f"Transcribed Text: {transcribed_text}")

            # Use RAG agent or conversational retrieval based on checkbox
            if st.checkbox("Use Document-based QA", key="rag_toggle"):
                output = rag_agent_instance.run(input=transcribed_text)
            else:
                input_dict = {"question": transcribed_text}
                qa = search_db()
                output = qa(input_dict)

            # Store conversation
            st.session_state.past.append(transcribed_text)
            st.session_state.generated.append(output)

    # Display conversation history
    if st.session_state["generated"]:
        display_conversation(st.session_state)

if __name__ == "__main__":
    main_f()
