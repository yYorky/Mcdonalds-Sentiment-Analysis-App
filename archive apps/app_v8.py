import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from langchain.callbacks import StreamlitCallbackHandler
from langchain_community.document_loaders import CSVLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_experimental.agents import create_pandas_dataframe_agent
from dotenv import load_dotenv
import os
import tempfile
import asyncio

# Import HTML templates
from htmlTemplates import css, bot_template, user_template

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_OPTIONS = ['llama-3.1-70b-versatile', 'llama-3.2-90b-text-preview', 'llama-3.1-8b-instant']

@st.cache_resource
def load_llm(model_name: str) -> ChatGroq:
    return ChatGroq(groq_api_key=GROQ_API_KEY, model_name=model_name)

@st.cache_resource
def create_df_agent(_df, _llm):
    return create_pandas_dataframe_agent(_llm, 
                                         _df, 
                                         verbose=True,
                                         allow_dangerous_code=True
                                         )

@st.cache_data
def load_and_process_data(file):
    return pd.read_csv(file)

def initialize_session_state():
    if "df" not in st.session_state:
        st.session_state.df = None
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

def setup_sidebar():
    st.sidebar.title("McDonald's Store Analysis")
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type="csv")
    model_name = st.sidebar.selectbox('Choose a model', MODEL_OPTIONS, key='model_choice')
    return uploaded_file, model_name

def handle_file_upload(uploaded_file):
    if uploaded_file is not None and st.session_state.df is None:
        try:
            with st.spinner("Processing file..."):
                st.session_state.df = load_and_process_data(uploaded_file)
                
                # Create vector store
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ','})
                documents = loader.load()
                os.unlink(tmp_file_path)
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                texts = text_splitter.split_documents(documents)
                
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                
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
                        vectors = FAISS.from_documents(batch, embeddings)
                    else:
                        batch_vectors = FAISS.from_documents(batch, embeddings)
                        vectors.merge_from(batch_vectors)
                    
                    # Update progress
                    progress = (i + batch_size) / len(texts)
                    progress_bar.progress(min(progress, 1.0))
                    status_text.text(f"Processing batch {(i // batch_size) + 1} of {total_batches}")
                
                st.session_state.vector_store = vectors
                
                # Clear the progress bar and status text
                progress_bar.empty()
                status_text.empty()
            
            st.sidebar.success("File uploaded and processed successfully!")
            st.sidebar.dataframe(st.session_state.df.head())
        except Exception as e:
            st.sidebar.error(f"Error processing file: {str(e)}")

async def hybrid_query(question: str, thought_process_placeholder):
    docs = st.session_state.vector_store.similarity_search(question, k=200)
    context = "\n".join([doc.page_content for doc in docs])
    
    thought_process = "Analyzing question and context...\n"
    thought_process_placeholder.text(thought_process)
    
    # Use the dataframe agent with Groq model to answer the question
    agent = create_df_agent(st.session_state.df, st.session_state.llm)
    thought_process += "Querying dataframe using Groq-based AI agent...\n"
    thought_process_placeholder.text(thought_process)
    
    agent_response = await asyncio.to_thread(agent.run, question)
    thought_process += f"Agent response:\n{agent_response}\n"
    thought_process_placeholder.text(thought_process)
    
    # Generate final response using context and agent's response
    final_prompt = f"""
    Use the following context, dataframe analysis, and the question to provide a comprehensive answer.
    Ensure that your response directly addresses the question and incorporates the agent's analysis.
    If the agent's response provides a specific numerical answer, make sure to include and emphasize this in your response.
    
    Context: {context}
    Dataframe Analysis: {agent_response}
    
    Question: {question}
    
    Your response (make sure to include and emphasize the agent's numerical answer if provided):
    """
    
    thought_process += "Generating final response...\n"
    thought_process_placeholder.text(thought_process)
    final_response = st.session_state.llm.invoke(final_prompt).content
    thought_process += "Response generated. Updating chat history...\n"
    thought_process_placeholder.text(thought_process)
       
    return final_response, thought_process

def handle_user_input(user_question: str):
    if user_question and st.session_state.vector_store:
        thought_process_placeholder = st.empty()
        with st.spinner("Analyzing..."):
            response, thought_process = asyncio.run(hybrid_query(user_question, thought_process_placeholder))
            
            st.session_state.chat_history.append({"role": "human", "content": user_question})
            st.session_state.chat_history.append({"role": "ai", "content": response, "thought_process": thought_process})
        
        display_chat_history()

def display_chat_history():
    for message in st.session_state.chat_history:
        if message["role"] == "human":
            st.markdown(user_template.replace("{{MSG}}", message['content']), unsafe_allow_html=True)
        else:
            st.markdown(bot_template.replace("{{MSG}}", message['content']).replace("{{THOUGHT_PROCESS}}", message.get('thought_process', 'No thought process available.')), unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="McDonald's Store Analysis", layout="wide")
    st.markdown(css, unsafe_allow_html=True)
    st.title("McDonald's Store Performance Analysis")

    initialize_session_state()
    uploaded_file, model_name = setup_sidebar()
    
    st.session_state.llm = load_llm(model_name)

    handle_file_upload(uploaded_file)

    st.subheader("Ask for Analysis")
    user_question = st.text_input("Ask a question about the McDonald's store data:")
    if user_question:
        handle_user_input(user_question)

    st.sidebar.markdown("---")
    st.sidebar.write("Powered by Groq and Streamlit")

if __name__ == '__main__':
    main()