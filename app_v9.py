import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain_community.document_loaders import CSVLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
import tempfile
import time

# Import HTML templates and utility functions (assuming they're in separate files)
from htmlTemplates import css, bot_template, user_template
from utils import preprocess_dataframe_grouped, create_map, get_conversational_response

load_dotenv()

# Constants
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_OPTIONS = ['llama3-8b-8192', 'llama-3.1-70b-versatile', 'llama3-70b-8192', 'llama-guard-3-8b' 'gemma2-9b-it', 'mixtral-8x7b-32768']

@st.cache_resource
def load_llm(model_name: str) -> ChatGroq:
    return ChatGroq(groq_api_key=GROQ_API_KEY, model_name=model_name)

@st.cache_data
def load_and_process_data(file):
    df = pd.read_csv(file)
    return preprocess_dataframe_grouped(df)

def initialize_session_state():
    if "df" not in st.session_state:
        st.session_state.df = None
    if "df_processed" not in st.session_state:
        st.session_state.df_processed = None
    if "pandas_agent" not in st.session_state:
        st.session_state.pandas_agent = None
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "llm" not in st.session_state:
        st.session_state.llm = None
    if "vectors" not in st.session_state:
        st.session_state.vectors = None
    if "embeddings_initialized" not in st.session_state:
        st.session_state.embeddings_initialized = False
    if "map_displayed" not in st.session_state:
        st.session_state.map_displayed = False

def setup_sidebar():
    st.sidebar.title("McDonald's Store Analysis")
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type="csv")
    model = st.sidebar.selectbox('Choose a model', MODEL_OPTIONS, key='model_choice')
    return uploaded_file, model


def load_initial_data(uploaded_file):
    if uploaded_file is not None and st.session_state.df is None:
        try:
            # Read and process the dataframe
            st.session_state.df = pd.read_csv(uploaded_file)
            st.session_state.df_processed = preprocess_dataframe_grouped(st.session_state.df)
            
            st.session_state.file_processed = True
            st.session_state.vector_embedding_complete = False
            
            st.sidebar.success("File uploaded complete!")
            # st.sidebar.subheader("Data Preview (Raw)")
            # st.sidebar.dataframe(st.session_state.df.head())
        except Exception as e:
            st.sidebar.error(f"Error processing file: {str(e)}")

def process_vector_embeddings():
    if not st.session_state.vector_embedding_complete:
        try:
            # Load documents for RAG
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(st.session_state.uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ','})
            st.session_state.documents = loader.load()
            os.unlink(tmp_file_path)
            
            # Perform vector embedding
            vector_embedding()
            
            # Create the pandas agent and RAG chain
            create_pandas_agent()
            create_rag_chain()
                
            st.session_state.vector_embedding_complete = True
            st.sidebar.success("Vector embeddings processed successfully!")
        except Exception as e:
            st.sidebar.error(f"Error processing vector embeddings: {str(e)}")

def create_pandas_agent():
    if st.session_state.df is not None and st.session_state.llm is not None:
        st.session_state.pandas_agent = create_pandas_dataframe_agent(
            st.session_state.llm,
            st.session_state.df_processed,
            verbose=True,
            agent_type="zero-shot-react-description",
            max_iterations=3,
            return_intermediate_steps=True,
            allow_dangerous_code=True,
        )

def vector_embedding():
    if not st.session_state.embeddings_initialized:
        
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
     
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        texts = text_splitter.split_documents(st.session_state.documents)
        st.sidebar.write(f"{len(texts)} chunks created.")

        st.sidebar.write("Creating vector embeddings in batches of 100...")
        # Process embeddings in batches of 100
        batch_size = 100
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        # Create a progress bar in the sidebar
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        vectors = None
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            if vectors is None:
                vectors = FAISS.from_documents(batch, st.session_state.embeddings)
            else:
                batch_vectors = FAISS.from_documents(batch, st.session_state.embeddings)
                vectors.merge_from(batch_vectors)
            
            # Update progress
            progress = (i + batch_size) / len(texts)
            progress_bar.progress(min(progress, 1.0))
            status_text.text(f"Processing batch {(i // batch_size) + 1} of {total_batches}")
        
        st.session_state.vectors = vectors
        # st.sidebar.write("Vector embeddings created.")
        st.session_state.embeddings_initialized = True

def create_rag_chain():
    if st.session_state.vectors is not None and st.session_state.llm is not None:
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
        retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 50})
        st.session_state.rag_chain = ConversationalRetrievalChain.from_llm(
            llm=st.session_state.llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True
        )

def display_map():
    if st.session_state.df_processed is not None and not st.session_state.map_displayed:
        st.subheader("Store Locations")
        fig = create_map(st.session_state.df_processed)
        st.plotly_chart(fig, use_container_width=True)
        st.session_state.map_displayed = True

def determine_approach(question: str) -> str:
    prompt = f"""
    You are an AI assistant tasked with determining the best approach to answer a user's question about McDonald's store data. You have two options:

    1. RAG (Retrieval-Augmented Generation): This approach is best for questions that require general information, customer reviews, or qualitative data that might be spread across multiple entries in the dataset.

    2. PANDAS: This approach is best for questions that require specific numerical analysis, statistics, or data manipulation on structured data.

    Based on the following user question, determine which approach would be more appropriate. Respond with either "RAG" or "PANDAS" followed by a brief explanation of your choice.

    User question: {question}

    Your response:
    """

    response = st.session_state.llm.invoke(prompt).content
    approach = "RAG" if "RAG" in response.split("\n")[0].upper() else "PANDAS"
    explanation = "\n".join(response.split("\n")[1:])
    
    return approach, explanation

def handle_user_input(user_question: str):
    if user_question and st.session_state.pandas_agent and st.session_state.rag_chain:
        with st.spinner("Analyzing..."):
            live_output = st.empty()
            st_callback = StreamlitCallbackHandler(live_output)

            try:
                approach, explanation = determine_approach(user_question)
                start = time.process_time()
                
                if approach == 'PANDAS':
                    response = st.session_state.pandas_agent.invoke(user_question, callbacks=[st_callback])
                    response_content = response.get('output', str(response)) if isinstance(response, dict) else str(response)
                    thought_process = response.get('intermediate_steps', []) if isinstance(response, dict) else []
                    formatted_thought_process = "\n".join([
                        f"Thought: {step[0].log}\nAction: {step[0].tool}\nAction Input: {step[0].tool_input}\nObservation: {step[1]}\n"
                        for step in thought_process
                    ])
                else:  # RAG approach
                    response = st.session_state.rag_chain({'question': user_question})
                    response_content = response['answer']
                    thought_process = []
                    formatted_thought_process = "Used RAG approach to answer the question."

                end = time.process_time()
                processing_time = end - start

                conversational_response = get_conversational_response(st.session_state.llm, response_content, user_question)

                full_thought_process = (
                    f"Approach chosen: {approach}\nReasoning: {explanation}\n\n"
                    f"{formatted_thought_process}\nFinal Answer: {response_content}\n\n"
                    f"Processing time: {processing_time:.2f} seconds"
                )

                st.session_state.chat_history.append({"role": "human", "content": user_question})
                st.session_state.chat_history.append({
                    "role": "ai",
                    "content": conversational_response,
                    "thought_process": full_thought_process
                })

                display_chat_history()

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                

def display_chat_history():
    for message in reversed(st.session_state.chat_history):
        if message["role"] == "human":
            st.write(user_template.replace("{{MSG}}", message['content']), unsafe_allow_html=True)
        else:
            bot_response = bot_template.replace("{{MSG}}", message['content'])
            bot_response = bot_response.replace("{{THOUGHT_PROCESS}}", message.get('thought_process', ''))
            st.write(bot_response, unsafe_allow_html=True)

    
def main():
    st.set_page_config(page_title="McDonald's Store Analysis", layout="wide")
    st.write(css, unsafe_allow_html=True)
    # st.title("McDonald's Store Performance Analysis")

    initialize_session_state()
    uploaded_file, model = setup_sidebar()

    if st.session_state.llm is None or st.session_state.llm.model_name != model:
        st.session_state.llm = load_llm(model)

    # Create two columns for the entire page layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Store Locations")
        map_placeholder = st.empty()

    with col2:
        st.subheader("Chat Interface")
        chat_placeholder = st.empty()

    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        load_initial_data(uploaded_file)

    if st.session_state.get('file_processed', False):
        # Display map in the first column
        with col1:
            fig = create_map(st.session_state.df_processed)
            map_placeholder.plotly_chart(fig, use_container_width=True)

        # Display chat interface in the second column
        with col2:
            user_question = st.text_input("Ask for analysis or suggestions:")
            if user_question:
                if st.session_state.vector_embedding_complete:
                    handle_user_input(user_question)
                else:
                    st.warning("Vector embeddings are still being processed. Please wait a moment before asking questions.")
            
            
        # Process vector embeddings in the background
        if not st.session_state.get('vector_embedding_complete', False):
            process_vector_embeddings()

    st.sidebar.markdown("---")
    st.sidebar.write("Powered by Groq and Streamlit")

if __name__ == '__main__':
    main()