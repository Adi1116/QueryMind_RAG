import os
import streamlit as st
from streamlit_chat import message
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import requests
from audio_recorder_streamlit import audio_recorder
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.agents import Tool, AgentExecutor
from langchain.agents.agent import create_zero_shot_agent
from langchain.prompts import ChatPromptTemplate

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

os.environ["COHERE_API_KEY"] = 'Ox97SolGnL68xrDjbNAMiVaWCqZ5Fny3d7hYAub6'
os.environ['API_KEY'] = "b1afee3b-c36c-4abf-8c35-5aeec8cba897"

# Document Preprocessing
@st.cache_data
def doc_preprocessing():
    loader = PyPDFLoader("iesc111.pdf")
    docs = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs_split = text_splitter.split_documents(docs)
    return docs_split

# Embeddings Store
@st.cache_resource
def embeddings_store():
    embedding = CohereEmbeddings(model="embed-english-v3.0")
    texts = doc_preprocessing()
    vectordb = FAISS.from_documents(documents=texts, embedding=embedding)
    retriever = vectordb.as_retriever()
    return retriever

# Conversational Retrieval Chain
@st.cache_resource
def search_db():
    retriever = embeddings_store()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatCohere(),
        memory=memory,
        retriever=retriever
    )
    return qa

# RAG Agent integration with proper parsing error handling
@st.cache_resource
def rag_agent():
    retriever = embeddings_store()

    llm = ChatCohere()
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

    rag_tool = Tool(
        name="DocumentQA",
        func=rag_chain.run,
        description="Use this tool to answer questions using document context."
    )

    # Custom prompt and agent setup
    prompt = ChatPromptTemplate.from_template(
        "Answer the following question using the available tools:\n\n{input}"
    )

    agent = create_zero_shot_agent(llm=llm, tools=[rag_tool], prompt=prompt)

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=[rag_tool],
        verbose=True,
        handle_parsing_errors=True  # ✅ Key Fix
    )

    return agent_executor

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
        data = json.loads(response.text)
        transcript = data["transcript"]
        return transcript

# Display conversation history using Streamlit messages
def display_conversation(history):
    for i in range(len(history["generated"])):
        user_message = history["past"][i] if isinstance(history["past"][i], str) else "Invalid user message"
        bot_message = history["generated"][i] if isinstance(history["generated"][i], str) else "Invalid bot response"
        message(user_message, is_user=True, key=str(i) + "_user")
        message(bot_message, key=str(i))

# Main Streamlit function
def main_f():
    st.title("LLM Powered Chatbot with Audio Input")

    rag_agent_instance = rag_agent()

    if "generated" not in st.session_state:
        st.session_state["generated"] = ["I am ready to help you"]
    if "past" not in st.session_state:
        st.session_state["past"] = ["Hey there!"]

    audio_bytes = audio_recorder()

    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")

        audio_file_path = "recorded_audio.wav"
        with open(audio_file_path, "wb") as f:
            f.write(audio_bytes)

        st.success("Audio recorded and saved as 'recorded_audio.wav'.")

        with st.spinner("Transcribing audio..."):
            transcribed_text = transcribe_audio(audio_file_path)
            st.write(f"Transcribed Text: {transcribed_text}")

            if st.checkbox("Use Document-based QA", key="rag_toggle"):
                output = rag_agent_instance.invoke({"input": transcribed_text})  # ✅ Use invoke
            else:
                input_dict = {"question": transcribed_text}
                qa = search_db()
                output = qa(input_dict)

            st.session_state.past.append(transcribed_text)
            st.session_state.generated.append(output)

    if st.session_state["generated"]:
        display_conversation(st.session_state)

if __name__ == "__main__":
    main_f()
