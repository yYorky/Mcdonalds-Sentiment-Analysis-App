import os
from typing import Tuple
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_groq import ChatGroq
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.agents import Tool, AgentExecutor, ZeroShotAgent
from langchain.memory import ConversationBufferMemory
from langchain.agents.conversational.output_parser import ConvoOutputParser
from langchain.agents import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish
from typing import Union
import re

# Constants
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_OPTIONS = ['llama-3.1-70b-versatile', 'llama-3.2-90b-text-preview', 'llama-3.1-8b-instant']

@st.cache_resource
def load_llm(model_name: str) -> ChatGroq:
    return ChatGroq(groq_api_key=GROQ_API_KEY, model_name=model_name)

def setup_sidebar() -> Tuple[st.file_uploader, str, bool]:
    st.sidebar.title("Advanced CSV Analysis")
    
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
        df = pd.read_csv(file)
        if df.empty or len(df.columns) == 0:
            st.error("The uploaded CSV file is empty or has no columns. Please check the file.")
            return None
        return df
    except Exception as e:
        st.error(f"An error occurred while reading the CSV file: {str(e)}")
    return None

def create_pandas_agent(llm, df):
    return create_pandas_dataframe_agent(
        llm, 
        df, 
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        allow_dangerous_code=True
    )

def create_text_analysis_chain(llm):
    prompt_template = """
    You are an expert in analyzing textual data. Given the following text, please analyze it and provide insights:
    
    {text}
    
    Question: {question}
    
    Please provide a detailed analysis, including:
    1. Key themes or topics
    2. Sentiment analysis
    3. Any notable patterns or trends
    4. Relevant quotes or examples
    
    Your analysis:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["text", "question"])
    return LLMChain(llm=llm, prompt=prompt)

def summarize_text(text, llm):
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(text)
    docs = [Document(page_content=t) for t in texts]
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    return chain.run(docs)

class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        
        regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            return AgentFinish(
                return_values={"output": llm_output.strip()},
                log=llm_output,
            )
        action = match.group(1).strip()
        action_input = match.group(2)
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

def create_agent(llm, pandas_agent, text_analysis_chain, df):
    tools = [
        Tool(
            name="Pandas Agent",
            func=lambda q: pandas_agent.run(q),
            description="Useful for when you need to answer questions about the dataframe. Input should be a fully formed question."
        ),
        Tool(
            name="Text Analysis",
            func=lambda q: text_analysis_chain.run(text=summarize_text(' '.join(df.select_dtypes(include=['object']).sample(min(100, len(df))).values.flatten()), llm), question=q),
            description="Useful for when you need to analyze text data or reviews. Input should be a fully formed question about the text content."
        )
    ]

    prefix = """You are an AI assistant designed to analyze CSV data efficiently. You have access to two tools:
    1. Pandas Agent: Use this for numerical analysis and questions about the dataframe structure.
    2. Text Analysis: Use this for analyzing text data, sentiments, or reviews.

    To use a tool, please use the following format:
    Thought: Consider the question and decide which tool to use.
    Action: Tool Name
    Action Input: The input to the tool
    Observation: The result of the action

    When you have a final answer or if you can answer directly without using tools, respond with:
    Thought: I now know the final answer
    Final Answer: Your final answer here

    Important guidelines:
    1. Limit yourself to a maximum of 3 tool uses per question.
    2. If you can answer the question directly without using tools, do so.
    3. Avoid repeating the same action unless absolutely necessary.
    4. If you're unsure or the question is unclear, ask for clarification instead of guessing.

    Begin! Remember to be concise and efficient in your analysis.

    Human: {input}
    {agent_scratchpad}"""

    prompt = ZeroShotAgent.create_prompt(
        tools, 
        prefix=prefix, 
        suffix="", 
        input_variables=["input", "agent_scratchpad"]
    )

    llm_chain = LLMChain(llm=llm, prompt=prompt)

    tool_names = [tool.name for tool in tools]
    agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names, output_parser=CustomOutputParser())

    return AgentExecutor.from_agent_and_tools(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        max_iterations=3,  # Limit the maximum number of steps
        early_stopping_method="generate",  # Stop if the agent decides it's done
    )

def main():
    load_dotenv()

    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY is not set")
        return
    
    st.set_page_config(page_title="Advanced CSV Analyzer", layout="wide")
    st.header("Advanced CSV Analyzer 📊")

    csv_file, model_name, show_data = setup_sidebar()

    if csv_file is not None:
        df = load_csv(csv_file)
        
        if df is not None:
            llm = load_llm(model_name)
            
            csv_file.seek(0)
            
            pandas_agent = create_pandas_agent(llm, df)
            text_analysis_chain = create_text_analysis_chain(llm)
            agent = create_agent(llm, pandas_agent, text_analysis_chain, df)

            if show_data:
                st.subheader("Raw Data")
                st.write(df)
            
            display_data_info(df)

            st.subheader("Ask a question about your CSV")
            user_question = st.text_input("Enter your question here: ")
            
            if user_question:
                with st.spinner(text="Analyzing..."):
                    st_callback = StreamlitCallbackHandler(st.container())
                    try:
                        response = agent.run(user_question, callbacks=[st_callback])
                        st.write("Analysis and Answer:")
                        st.write(response)
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        st.write("The agent couldn't complete the analysis. You may want to try rephrasing your question or breaking it down into smaller parts.")

        else:
            st.warning("Please upload a valid CSV file to proceed.")

if __name__ == "__main__":
    main()