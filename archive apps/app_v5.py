import os
from dotenv import load_dotenv
import streamlit as st
from langchain_experimental.agents import create_csv_agent
from langchain_groq import ChatGroq

# Constants
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_OPTIONS = ['llama-3.1-70b-versatile', 'llama-3.2-90b-text-preview', 'llama-3.1-8b-instant']

def load_llm(model_name: str) -> ChatGroq:
    return ChatGroq(groq_api_key=GROQ_API_KEY, model_name=model_name)

def setup_sidebar():
    st.sidebar.title("McDonald's Store Analysis")
    
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")
    model = st.sidebar.selectbox('Choose a model', MODEL_OPTIONS, key='model_choice')
    return uploaded_file, model

def main():
    load_dotenv()
    
    # Load the Groq API key from the environment variable
    if GROQ_API_KEY is None or GROQ_API_KEY == "":
        st.error("GROQ_API_KEY is not set")
        return
    else:
        print("GROQ_API_KEY is set")

    st.set_page_config(page_title="Ask your CSV")
    st.header("Ask your CSV 📈")

    csv_file, model_name = setup_sidebar()

    if csv_file is not None:
        llm = load_llm(model_name)
        agent = create_csv_agent(llm, 
                                 csv_file, 
                                 verbose=True,
                                 allow_dangerous_code=True,
                                 show_intermediate_steps=True
                                 )

        user_question = st.text_input("Ask a question about your CSV: ")
        if user_question is not None and user_question != "":
            with st.spinner(text="In progress..."):
                st.write(agent.run(user_question))

if __name__ == "__main__":
    main()