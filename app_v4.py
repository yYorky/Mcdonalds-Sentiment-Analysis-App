# Issues cannot answer basic store number question properly
# not clearing thought process before display chat history
# requires multiple reload of dataset for each question


import streamlit as st
import pandas as pd
import numpy as np
from langchain_groq import ChatGroq
from langchain.agents import Tool
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents import AgentExecutor
from langchain.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv
import os
import time
import json
import groq
import plotly.express as px
import re
import ast
import io

# Import HTML templates and utility functions (assuming they're in separate files)
from htmlTemplates import css, bot_template, user_template
from utils import preprocess_dataframe_grouped, create_map

load_dotenv()

# Constants
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_OPTIONS = ['llama-3.1-70b-versatile', 'llama-3.2-90b-text-preview', 'llama-3.1-8b-instant']

@st.cache_resource
def load_llm(model_name: str) -> ChatGroq:
    return ChatGroq(groq_api_key=GROQ_API_KEY, model_name=model_name)

@st.cache_data
def load_and_process_data(file):
    df = pd.read_csv(file)
    return preprocess_dataframe_grouped(df)

def initialize_session_state():
    default_states = {
        "df": None,
        "df_processed": None,
        "chat_history": [],
        "llm": None,
        "groq_client": groq.Groq(),
        "pandas_agent": None,
    }
    for key, default_value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def setup_sidebar():
    st.sidebar.title("McDonald's Store Analysis")
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type="csv")
    model = st.sidebar.selectbox('Choose a model', MODEL_OPTIONS, key='model_choice')
    return uploaded_file, model

def handle_file_upload(uploaded_file):
    if uploaded_file is not None and st.session_state.df is None:
        try:
            with st.spinner("Processing file..."):
                st.session_state.df = pd.read_csv(uploaded_file)
                st.session_state.df_processed = preprocess_dataframe_grouped(st.session_state.df)
                                
            st.sidebar.success("File uploaded and processed successfully!")
            st.sidebar.subheader("Data Preview (Raw)")
            st.sidebar.dataframe(st.session_state.df.head())
            
            # Create pandas agent
            st.session_state.pandas_agent = create_pandas_dataframe_agent(
                st.session_state.llm,
                st.session_state.df,
                verbose=True,
                agent_type="zero-shot-react-description",
                handle_parsing_errors=True,
                return_intermediate_steps=True,
                allow_dangerous_code=True,
            )
            
            # Log DataFrame info for debugging
            st.sidebar.write(f"DataFrame shape: {st.session_state.df.shape}")
            st.sidebar.write(f"DataFrame columns: {st.session_state.df.columns.tolist()}")
            
        except Exception as e:
            st.sidebar.error(f"Error processing file: {str(e)}")
            
    else:
        st.session_state.df = None
        st.session_state.df_processed = None
        st.session_state.pandas_agent = None

def make_api_call(messages, max_tokens, is_final_answer=False):
    client = st.session_state.groq_client
    
    for attempt in range(3):
        try:
            if is_final_answer:
                response = client.chat.completions.create(
                    model=st.session_state.llm.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.2,
                ) 
                return response.choices[0].message.content
            else:
                response = client.chat.completions.create(
                    model=st.session_state.llm.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )
                return json.loads(response.choices[0].message.content)
        except Exception as e:
            if attempt == 2:
                if is_final_answer:
                    return {"title": "Error", "content": f"Failed to generate final answer after 3 attempts. Error: {str(e)}"}
                else:
                    return {"title": "Error", "content": f"Failed to generate step after 3 attempts. Error: {str(e)}", "next_action": "final_answer"}
            time.sleep(1)  # Wait for 1 second before retrying

def get_dataset_info():
    if st.session_state.df is not None:
        columns = st.session_state.df.columns.tolist()
        sample_data = st.session_state.df.head().to_dict(orient='records')
        return f"""
        Dataset Information:
        Columns: {columns}
        Sample Data (first 5 rows): {json.dumps(sample_data, indent=2)}
        """
    return "No dataset loaded."

def execute_pandas_agent(query):
    if st.session_state.pandas_agent is None:
        return "Error: Pandas agent is not initialized. Please upload a CSV file first."
    try:
        result = st.session_state.pandas_agent.invoke(query)
        return result
    except Exception as e:
        return f"Error executing pandas agent: {str(e)}"

def generate_response(prompt):
    if st.session_state.df is None:
        return [("Error", "No dataset loaded. Please upload a CSV file first.", 0)], 0

    dataset_info = get_dataset_info()
    messages = [
        {"role": "system", "content": f"""You are an expert AI assistant analyzing McDonald's store data. 
         You have access to the following dataset:

         {dataset_info}

         You can perform data analysis by using the pandas_agent. To use it, format your request as:
         [PANDAS_AGENT: your question or analysis request]

         Example: [PANDAS_AGENT: What is the average revenue across all stores?]

         Explain your reasoning step by step. For each step, provide a title that describes what you're doing in that step, 
         along with the content. Decide if you need another step or if you're ready to give the final answer. 
         Respond in JSON format with 'title', 'content', and 'next_action' (either 'continue' or 'final_answer') keys. 
         USE AS MANY REASONING STEPS AS POSSIBLE. AT LEAST 3. 
         BE AWARE OF YOUR LIMITATIONS AS AN LLM AND WHAT YOU CAN AND CANNOT DO. IN YOUR REASONING, 
         INCLUDE EXPLORATION OF ALTERNATIVE ANSWERS. CONSIDER YOU MAY BE WRONG, AND IF YOU ARE WRONG IN YOUR REASONING, 
         WHERE IT WOULD BE. FULLY TEST ALL OTHER POSSIBILITIES. YOU CAN BE WRONG. WHEN YOU SAY YOU ARE RE-EXAMINING, 
         ACTUALLY RE-EXAMINE, AND USE ANOTHER APPROACH TO DO SO. DO NOT JUST SAY YOU ARE RE-EXAMINING. 
         USE AT LEAST 3 METHODS TO DERIVE THE ANSWER. USE BEST PRACTICES.
         ENSURE YOUR FINAL ANSWER IS CONSISTENT WITH THE INTERMEDIATE STEPS AND RESULTS.
         Base your analysis and answers on the provided dataset information and the analyses you perform using the pandas_agent."""},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "I will now think step by step to answer your question about McDonald's store data, based on the provided dataset and using the pandas_agent for analysis."}
    ]
    
    steps = []
    step_count = 1
    total_thinking_time = 0
    
    while True:
        start_time = time.time()
        step_data = make_api_call(messages, 300)
        end_time = time.time()
        thinking_time = end_time - start_time
        total_thinking_time += thinking_time
        
        # Check if the step contains a pandas_agent request
        content = step_data['content']
        pandas_agent_requests = re.findall(r'\[PANDAS_AGENT: ([^\]]+)\]', content)
        for request in pandas_agent_requests:
            result = execute_pandas_agent(request)
            content = content.replace(f"[PANDAS_AGENT: {request}]", f"Pandas Agent Result: {result}")
        
        steps.append((f"Step {step_count}: {step_data['title']}", content, thinking_time))
        
        messages.append({"role": "assistant", "content": json.dumps({"title": step_data['title'], "content": content, "next_action": step_data['next_action']})})
        
        yield steps, None  # Yield intermediate steps
        
        if step_data['next_action'] == 'final_answer' or step_count > 25:
            break
        
        step_count += 1

    messages.append({"role": "user", "content": """Please provide the final answer based solely on your reasoning above and the provided dataset. 
                     Ensure your answer is consistent with the intermediate steps and results.
                     Do not use JSON formatting. Only provide the text response without any titles or preambles. 
                     Retain any formatting as instructed by the original prompt, 
                     such as exact formatting for free response or multiple choice."""})
    
    start_time = time.time()
    final_data = make_api_call(messages, 1200, is_final_answer=True)
    end_time = time.time()
    thinking_time = end_time - start_time
    total_thinking_time += thinking_time
    
    steps.append(("Final Answer", final_data, thinking_time))
    yield steps, total_thinking_time

def handle_user_input(user_question: str):
    if st.session_state.df is None:
        st.error("Please upload a CSV file before asking questions.")
        return

    if user_question:
        with st.spinner("Analyzing..."):
            thought_process = st.empty()
            
            try:
                # Generate response
                thought_process.markdown("### Thought Process")
                
                response_generator = generate_response(user_question)
                
                reasoning_steps = []
                for steps, _ in response_generator:
                    for step in steps:
                        reasoning_steps.append(step)
                        thought_process.markdown(f"**{step[0]}**\n{step[1]}\n")
                
                # Extract final answer
                final_answer = reasoning_steps[-1][1] if reasoning_steps else "No final answer generated."
                
                # Add the response to chat history
                st.session_state.chat_history.append({"role": "human", "content": user_question})
                st.session_state.chat_history.append({
                    "role": "ai",
                    "content": final_answer,
                    "thought_process": "### Detailed Thought Process:\n" + 
                                       "\n".join([f"**{step[0]}**\n{step[1]}" for step in reasoning_steps])
                })

                # Display the updated chat history
                display_chat_history()

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.error(f"Error details: {type(e).__name__}, {str(e)}")

def display_chat_history():
    for message in reversed(st.session_state.chat_history):
        if message["role"] == "human":
            st.write(user_template.replace("{{MSG}}", message['content']), unsafe_allow_html=True)
        else:
            bot_response = bot_template.replace("{{MSG}}", message['content'])
            if 'thought_process' in message:
                bot_response = bot_response.replace("{{THOUGHT_PROCESS}}", f"{message['thought_process']}")
            else:
                bot_response = bot_response.replace("{{THOUGHT_PROCESS}}", "")
            st.write(bot_response, unsafe_allow_html=True)

def display_map():
    if st.session_state.df_processed is not None:
        st.subheader("Store Locations")
        fig = create_map(st.session_state.df_processed)
        st.plotly_chart(fig, use_container_width=True)

def main():
    st.set_page_config(page_title="McDonald's Store Analysis", layout="wide")
    st.write(css, unsafe_allow_html=True)
    st.title("McDonald's Store Performance Analysis")

    initialize_session_state()
    uploaded_file, model = setup_sidebar()

    if st.session_state.llm is None or st.session_state.llm.model_name != model:
        st.session_state.llm = load_llm(model)

    handle_file_upload(uploaded_file)

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