# ideas for next iteration, instead of using rag, have the agent use the dataframe to perform the analysis directly.

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
from langchain.agents import AgentExecutor
from dotenv import load_dotenv
import os
import tempfile
import time
import json
import groq

# Import HTML templates and utility functions (assuming they're in separate files)
from htmlTemplates import css, bot_template, user_template
from utils import preprocess_dataframe, create_map, get_conversational_response

load_dotenv()

# Constants
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_OPTIONS = ['llama-3.1-70b-versatile','llama-3.2-90b-text-preview','llama-3.1-8b-instant']

@st.cache_resource
def load_llm(model_name: str) -> ChatGroq:
    return ChatGroq(groq_api_key=GROQ_API_KEY, model_name=model_name)

@st.cache_data
def load_and_process_data(file):
    df = pd.read_csv(file)
    return preprocess_dataframe(df)

def initialize_session_state():
    default_states = {
        "df": None,
        "df_processed": None,
        "pandas_agent": None,
        "rag_chain": None,
        "chat_history": [],
        "llm": None,
        "vectors": None,
        "embeddings_initialized": False,
        "groq_client": groq.Groq(),
    }
    for key, default_value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def setup_sidebar():
    st.sidebar.title("McDonald's Store Analysis")
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type="csv")
    model = st.sidebar.selectbox('Choose a model', MODEL_OPTIONS, key='model_choice')
    st.session_state.chunk_size = st.sidebar.slider('Chunk size:', 1000, 8000, value=2000, step=500)
    st.session_state.chunk_overlap = st.sidebar.slider('Chunk overlap:', 0, 1000, value=200, step=100)
    return uploaded_file, model

def handle_file_upload(uploaded_file):
    if uploaded_file is not None and st.session_state.df is None:
        try:
            with st.spinner("Processing file..."):
                st.session_state.df = pd.read_csv(uploaded_file)
                st.session_state.df_processed = preprocess_dataframe(st.session_state.df)
                
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ','})
                st.session_state.documents = loader.load()
                os.unlink(tmp_file_path)
                
            st.sidebar.success("File uploaded and processed successfully!")
            st.sidebar.subheader("Data Preview (Raw)")
            st.sidebar.dataframe(st.session_state.df.head())
            
            create_pandas_agent()
            create_rag_chain()
        except Exception as e:
            st.sidebar.error(f"Error processing file: {str(e)}")

def create_pandas_agent():
    if st.session_state.df is not None and st.session_state.llm is not None:
        st.session_state.pandas_agent = create_pandas_dataframe_agent(
            st.session_state.llm,
            st.session_state.df_processed,
            verbose=True,
            agent_type="zero-shot-react-description",
            return_intermediate_steps=True,
            allow_dangerous_code=True,
        )

def vector_embedding():
    if not st.session_state.embeddings_initialized:
        st.sidebar.write("Initializing embeddings and processing documents...")
        progress_bar = st.sidebar.progress(0)
        progress_text = st.sidebar.empty()
        
        try:
            st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=st.session_state.chunk_size, chunk_overlap=st.session_state.chunk_overlap)
            texts = text_splitter.split_documents(st.session_state.documents)
            
            # Batch processing
            batch_size = 100  # Maximum allowed by Google Generative AI
            vectors = None
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                if vectors is None:
                    vectors = FAISS.from_documents(batch, st.session_state.embeddings)
                else:
                    batch_vectors = FAISS.from_documents(batch, st.session_state.embeddings)
                    vectors.merge_from(batch_vectors)
                progress = min(1.0, (i + batch_size) / len(texts))
                progress_bar.progress(progress)
                progress_text.text(f"Processing embeddings... {progress:.0%}")
            
            st.session_state.vectors = vectors
            st.session_state.embeddings_initialized = True
            progress_bar.empty()
            progress_text.empty()
            st.sidebar.success("Vector embeddings created successfully!")
        except Exception as e:
            progress_bar.empty()
            progress_text.empty()
            st.sidebar.error(f"An error occurred during embedding: {str(e)}")

def create_rag_chain():
    if st.session_state.vectors is not None and st.session_state.llm is not None:
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
        retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 200})
        st.session_state.rag_chain = ConversationalRetrievalChain.from_llm(
            llm=st.session_state.llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True
        )

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
        
def generate_response(prompt, approach, explanation, initial_response, thought_process):
    messages = [
        {"role": "system", "content": """You are an expert AI assistant that explains your reasoning step by step.  
         For each step, provide a title that describes what you're doing in that step, along with the content. 
         Decide if you need another step or if you're ready to give the final answer. 
         Respond in JSON format with 'title', 'content', and 'next_action' (either 'continue' or 'final_answer') keys. 
         USE AS MANY REASONING STEPS AS POSSIBLE. AT LEAST 3. 
             BE AWARE OF YOUR LIMITATIONS AS AN LLM AND WHAT YOU CAN AND CANNOT DO. IN YOUR REASONING, 
             INCLUDE EXPLORATION OF ALTERNATIVE ANSWERS. CONSIDER YOU MAY BE WRONG, AND IF YOU ARE WRONG IN YOUR REASONING, 
             WHERE IT WOULD BE. FULLY TEST ALL OTHER POSSIBILITIES. YOU CAN BE WRONG. WHEN YOU SAY YOU ARE RE-EXAMINING, 
             ACTUALLY RE-EXAMINE, AND USE ANOTHER APPROACH TO DO SO. DO NOT JUST SAY YOU ARE RE-EXAMINING. 
             USE AT LEAST 3 METHODS TO DERIVE THE ANSWER. USE BEST PRACTICES."""},
        
        {"role": "user", "content": f"Approach: {approach}
         \nExplanation: {explanation}
         \nInitial response: {initial_response}
         \nThought process: {thought_process}
         \n
         \nBased on this information, {prompt}"},
        
        {"role": "assistant", "content": "Thank you! I will now think step by step following my instructions, \
         starting at the beginning after decomposing the problem and considering the provided thought process."}
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
        
        steps.append((f"Step {step_count}: {step_data['title']}", step_data['content'], thinking_time))
        
        messages.append({"role": "assistant", "content": json.dumps(step_data)})
        
        if step_data['next_action'] == 'final_answer' or step_count > 25:
            break
        
        step_count += 1
        yield steps, None

    messages.append({"role": "user", "content": "Please provide the final answer based solely on your reasoning above. \
                     Do not use JSON formatting. Only provide the text response without any titles or preambles. \
                     Retain any formatting as instructed by the original prompt, \
                     such as exact formatting for free response or multiple choice."})
    
    start_time = time.time()
    final_data = make_api_call(messages, 1200, is_final_answer=True)
    end_time = time.time()
    thinking_time = end_time - start_time
    total_thinking_time += thinking_time
    
    steps.append(("Final Answer", final_data, thinking_time))
    yield steps, total_thinking_time

def determine_approach(question: str) -> str:
    prompt = f"""
    Determine the best approach to answer this question about McDonald's store data:
    1. RAG: For general information, customer reviews, or qualitative data spread across multiple entries.
    2. PANDAS: For specific numerical analysis, statistics, or data manipulation on structured data.
    
    Question: {question}
    
    Respond with "RAG" or "PANDAS" followed by a brief explanation.
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
                    thought_process = response.get('source_documents', [])
                    formatted_thought_process = "Used RAG approach to answer the question.\n" + "\n".join([
                        f"Source {i+1}: {doc.page_content[:200]}..."  # Truncate long content
                        for i, doc in enumerate(thought_process)
                    ])

                # Use the modified reasoning chain to generate a detailed response
                reasoning_steps = []
                total_thinking_time = 0
                for steps, time in generate_response(
                    "provide a detailed analysis",
                    approach,
                    explanation,
                    response_content,
                    formatted_thought_process
                ):
                    reasoning_steps = steps
                    if time is not None:
                        total_thinking_time = time
                    
                    # Update the live output with the current steps
                    live_output.text("\n".join([f"{step[0]}\n{step[1]}" for step in steps]))

                final_answer = reasoning_steps[-1][1]
                
                full_thought_process = (
                    f"Approach chosen: {approach}\nReasoning: {explanation}\n\n"
                    f"Initial Thought Process:\n{formatted_thought_process}\n\n"
                    "Detailed Reasoning:\n" +
                    "\n".join([f"{step[0]}\n{step[1]}" for step in reasoning_steps[:-1]]) +
                    f"\n\nFinal Answer: {final_answer}\n\n"
                    f"Total processing time: {total_thinking_time:.2f} seconds"
                )
                
                # Clear the live output
                live_output.empty()

                st.session_state.chat_history.append({"role": "human", "content": user_question})
                st.session_state.chat_history.append({
                    "role": "ai",
                    "content": final_answer,
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

    if st.sidebar.button("Process Data"):
        vector_embedding()
        create_rag_chain()

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