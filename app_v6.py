import os
from typing import Tuple
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from langchain_experimental.agents import create_csv_agent
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_groq import ChatGroq
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
import io

# tools for python_repl_ast
import pandas as pd
import re

# Constants
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_OPTIONS = ['llama-3.1-70b-versatile', 'llama-3.2-90b-text-preview', 'llama-3.1-8b-instant']

@st.cache_resource
def load_llm(model_name: str) -> ChatGroq:
    return ChatGroq(groq_api_key=GROQ_API_KEY, model_name=model_name)

def setup_sidebar() -> Tuple[st.file_uploader, str, bool]:
    st.sidebar.title("McDonald's Store Analysis")
    
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")
    model = st.sidebar.selectbox('Choose a model', MODEL_OPTIONS, key='model_choice')
    show_data = st.sidebar.checkbox('Show raw data')
    return uploaded_file, model, show_data

def display_data_info(df: pd.DataFrame):
    st.sidebar.subheader("Dataset Information")
    st.sidebar.write(f"Number of rows: {df.shape[0]}")
    st.sidebar.write(f"Number of columns: {df.shape[1]}")
    st.sidebar.write("Data types:")
    st.sidebar.write(df.dtypes)

def load_csv(file) -> pd.DataFrame:
    try:
        # Try to read the CSV file
        df = pd.read_csv(file)
        
        # Check if the DataFrame is empty
        if df.empty:
            st.error("The uploaded CSV file is empty. Please upload a non-empty CSV file.")
            return None
        
        # Check if there are any columns
        if len(df.columns) == 0:
            st.error("The uploaded CSV file has no columns. Please check the file format.")
            return None
        
        return df
    except pd.errors.EmptyDataError:
        st.error("The uploaded CSV file is empty. Please upload a non-empty CSV file.")
    except pd.errors.ParserError:
        st.error("Unable to parse the CSV file. Please check if it's a valid CSV format.")
    except Exception as e:
        st.error(f"An error occurred while reading the CSV file: {str(e)}")
    
    return None

def agentic_reasoning(agent, question: str, callbacks) -> str:
    """
    Agent will reason through steps and stop when the answer is found.
    """
    # Initialize the agent's response reasoning process
    prompt = (
        f"Analyze the data to answer this question: {question}\n"
        f"At each step, check if the Final Answer is found and agent has completed the steps. If so, stop and provide the final answer."
    )
    
    response = agent.run(prompt, callbacks=callbacks)

    return response
    
    # # If further steps are necessary, the agent continues reasoning
    # additional_prompt = (
    #     f"Proceed with the next necessary steps to refine the analysis, "
    #     f"but check again if the answer can be provided early."
    # )
    
    # # Continue processing until the agent concludes with an answer
    # response = agent.run(additional_prompt, callbacks=callbacks)
    # return response

def main():
    load_dotenv()

    if GROQ_API_KEY is None or GROQ_API_KEY == "":
        st.error("GROQ_API_KEY is not set")
        return
    
    st.set_page_config(page_title="Ask your CSV", layout="wide")
    st.header("Ask your CSV 📈")

    csv_file, model_name, show_data = setup_sidebar()

    if csv_file is not None:
        df = load_csv(csv_file)
        
        if df is not None:
            llm = load_llm(model_name)
            
            # Reset file pointer to the beginning
            csv_file.seek(0)
            
            agent = create_pandas_dataframe_agent(
                llm, 
                df, 
                verbose=True,
               agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                handle_parsing_errors=True,
                allow_dangerous_code=True
            )

            if show_data:
                st.subheader("Raw Data")
                st.write(df)
            
            display_data_info(df)

            st.subheader("Ask a question about your CSV")
            user_question = st.text_input("Enter your question here: ")
            
            if user_question:
                with st.spinner(text="Analyzing..."):
                    st_callback = StreamlitCallbackHandler(st.container())
                    response = agentic_reasoning(agent, user_question, [st_callback])
                    st.write("Answer with Reasoning:")
                    st.write(response)

        else:
            st.warning("Please upload a valid CSV file to proceed.")

if __name__ == "__main__":
    main()
