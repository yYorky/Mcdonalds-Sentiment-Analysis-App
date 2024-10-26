import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from langchain_groq import ChatGroq
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv
import os
import tempfile
import time

from utils import preprocess_dataframe, create_map, get_conversational_response, generate_insights
from htmlTemplates import css, bot_template, user_template

load_dotenv()

# Constants
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_OPTIONS = ['llama-3.1-70b-versatile', 'llama-3.1-8b-instant']

@st.cache_resource
def load_llm(model_name: str) -> ChatGroq:
    return ChatGroq(groq_api_key=GROQ_API_KEY, model_name=model_name)

@st.cache_data
def load_and_process_data(file):
    df = pd.read_csv(file)
    return preprocess_dataframe(df)

def initialize_session_state():
    if "df" not in st.session_state:
        st.session_state.df = None
    if "df_processed" not in st.session_state:
        st.session_state.df_processed = None
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "llm" not in st.session_state:
        st.session_state.llm = None

def setup_sidebar():
    st.sidebar.title("McDonald's Store Analysis")
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type="csv")
    model = st.sidebar.selectbox('Choose a model', MODEL_OPTIONS, key='model_choice')
    return uploaded_file, model

def handle_file_upload(uploaded_file):
    if uploaded_file is not None and st.session_state.df is None:
        try:
            with st.spinner("Processing file..."):
                # Read the raw dataframe
                st.session_state.df = pd.read_csv(uploaded_file)
                
                # Process the dataframe for visualization purposes
                st.session_state.df_processed = preprocess_dataframe(st.session_state.df)
                
            st.sidebar.success("File uploaded and processed successfully!")
            st.sidebar.subheader("Data Preview (Raw)")
            st.sidebar.dataframe(st.session_state.df.head())
            
            # # Generate and display insights
            # insights = generate_insights(st.session_state.df_processed)
            # st.subheader("Initial Insights")
            # st.write(insights)
            
            # Create the pandas agent with the raw dataframe
            create_pandas_agent()
        except Exception as e:
            st.sidebar.error(f"Error processing file: {str(e)}")

def create_pandas_agent():
    if st.session_state.df is not None and st.session_state.llm is not None:
        st.session_state.agent = create_pandas_dataframe_agent(
            st.session_state.llm,
            st.session_state.df_processed,  # Use the processed dataframe
            verbose=True,
            agent_type="zero-shot-react-description",
            return_intermediate_steps=True,
            allow_dangerous_code=True,
        )

def display_map():
    if st.session_state.df_processed is not None:
        st.subheader("Store Locations")
        fig = create_map(st.session_state.df_processed)
        st.plotly_chart(fig, use_container_width=True)

def handle_user_input(user_question: str):
    if user_question and st.session_state.agent:
        with st.spinner("Analyzing..."):
            live_output = st.empty()
            st_callback = StreamlitCallbackHandler(live_output)

            try:
                start = time.process_time()
                response = st.session_state.agent.invoke(user_question, callbacks=[st_callback])
                end = time.process_time()
                processing_time = end - start

                response_content = response.get('output', str(response)) if isinstance(response, dict) else str(response)
                thought_process = response.get('intermediate_steps', []) if isinstance(response, dict) else []

                formatted_thought_process = "\n".join([
                    f"Thought: {step[0].log}\nAction: {step[0].tool}\nAction Input: {step[0].tool_input}\nObservation: {step[1]}\n"
                    for step in thought_process
                ])

                conversational_response = get_conversational_response(st.session_state.llm, response_content, user_question)

                full_response = (
                    f"{conversational_response}\n\n<details><summary>View thought process</summary><pre>"
                    f"{formatted_thought_process}\nFinal Answer: {response_content}\n\n"
                    f"Processing time: {processing_time:.2f} seconds</pre></details>"
                )

                st.session_state.chat_history.append({"role": "human", "content": user_question})
                st.session_state.chat_history.append({
                    "role": "ai",
                    "content": conversational_response,
                    "thought_process": f"{formatted_thought_process}\nFinal Answer: {response_content}\n\nProcessing time: {processing_time:.2f} seconds"
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
    st.title("McDonald's Store Performance Analysis")

    initialize_session_state()
    uploaded_file, model = setup_sidebar()

    if st.session_state.llm is None or st.session_state.llm.model_name != model:
        st.session_state.llm = load_llm(model)

    handle_file_upload(uploaded_file)
    create_pandas_agent()

    col1, col2 = st.columns(2)

    with col1:
        display_map()

    with col2:
        st.subheader("Ask for Analysis")
        user_question = st.text_input("Ask for analysis or suggestions:")
        if user_question:
            handle_user_input(user_question)

    st.sidebar.markdown("---")
    st.sidebar.write("Powered by Groq and Streamlit")

if __name__ == '__main__':
    main()