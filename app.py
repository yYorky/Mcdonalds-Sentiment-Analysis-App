import streamlit as st
import os
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tempfile
from langchain_groq import ChatGroq
from langchain_community.document_loaders import CSVLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Import HTML templates for chat messages
from htmlTemplates import css, bot_template, user_template

# Load environment variables from .env file
load_dotenv()

# Load API key
groq_api_key = os.getenv('GROQ_API_KEY')

# Function to load and process the CSV file
def load_csv(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ','})
    data = loader.load()

    # Load as DataFrame for visualization and preprocessing
    df = pd.read_csv(tmp_file_path)

    # Preprocess the DataFrame
    df = preprocess_dataframe(df)

    # Clean up the temporary file
    os.unlink(tmp_file_path)

    return data, df

# Function to preprocess the dataframe
def preprocess_dataframe(df):
    # Convert 'rating' to numeric, coercing errors to NaN
    df['rating'] = pd.to_numeric(df['rating'].str.replace(' star', '').str.replace(' stars', ''), errors='coerce')
    
    # Drop rows with NaN ratings
    df = df.dropna(subset=['rating'])
    
    # Convert latitude and longitude to numeric
    df['latitude'] = pd.to_numeric(df['latitude '], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    
    # Drop rows with invalid coordinates
    df = df.dropna(subset=['latitude', 'longitude'])
    
    # Group by store and calculate average rating and mode sentiment
    df_grouped = df.groupby('store_address').agg({
        'latitude': 'first',
        'longitude': 'first',
        'rating_count': 'first',
        'rating': 'mean',
        'sentiment': lambda x: x.mode().iloc[0] if not x.empty else None
    }).reset_index()
    
    return df_grouped

# Function to create an interactive map
def create_map(df):
    fig = go.Figure()

    fig.add_trace(go.Scattermapbox(
        lat=df['latitude'],
        lon=df['longitude'],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=15,
            color='red',
            opacity=0.7,
            sizemin=5,  # Minimum size of the marker
            sizemode='area'  # Size scales with zoom
        ),
        text=df['store_address'],
        hoverinfo='text',
        hovertemplate=
        "<b>%{text}</b><br>" +
        "Rating Count: %{customdata[0]}<br>" +
        "Average Rating: %{customdata[1]:.2f}<br>" +
        "Sentiment: %{customdata[2]}<extra></extra>",
        customdata=df[['rating_count', 'rating', 'sentiment']]
    ))

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(
            zoom=3,
            center=go.layout.mapbox.Center(lat=39.8283, lon=-98.5795)  # Center of USA
        ),
        showlegend=False,
        height=600,
        margin={"r":0,"t":0,"l":0,"b":0}
    )

    return fig

def vector_embedding(documents):
    if "vectors" not in st.session_state:
        st.session_state.embeddings_initialized = False
        st.sidebar.write("Initializing embeddings...")

        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        st.sidebar.write("Embeddings initialized.")

        st.sidebar.write("Processing documents...")
        
        st.sidebar.write(f"{len(documents)} documents processed.")

        st.sidebar.write("Splitting text into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=st.session_state.chunk_size, chunk_overlap=st.session_state.chunk_overlap)
        texts = text_splitter.split_documents(documents)
        st.sidebar.write(f"{len(texts)} chunks created.")

        st.sidebar.write("Creating vector embeddings...")
        
        # Process in batches of 100
        batch_size = 100
        vectors = None
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            if vectors is None:
                vectors = FAISS.from_documents(batch, st.session_state.embeddings)
            else:
                batch_vectors = FAISS.from_documents(batch, st.session_state.embeddings)
                vectors.merge_from(batch_vectors)
            
            st.sidebar.write(f"Processed batch {i//batch_size + 1} of {(len(texts)-1)//batch_size + 1}")
        
        st.session_state.vectors = vectors
        st.sidebar.write("Vector embeddings created.")
        st.session_state.embeddings_initialized = True

# Function to get the conversational retrieval chain
def get_conversation_chain(vectorstore, model_name):
    llm = ChatGroq(groq_api_key=groq_api_key, model_name=model_name)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
    
    # Increase the number of retrieved documents
    retriever = vectorstore.as_retriever(search_kwargs={"k": 200})
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )
    return conversation_chain

# Streamlit app
st.set_page_config(page_title="McDonald's Store Analysis", layout="wide")

# Sidebar
st.sidebar.title("McDonald's Store Analysis")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type="csv")

if uploaded_file is not None:
    documents, df = load_csv(uploaded_file)
    st.sidebar.success("File uploaded and processed successfully!")

    # Data preview
    st.sidebar.subheader("Data Preview")
    st.sidebar.dataframe(df.head())

    # Customization options
    st.sidebar.title('Customization')
    model = st.sidebar.selectbox(
        'Choose a model',
        ['llama-3.1-70b-versatile','llama-3.1-8b-instant' ],
        key='model_choice',
    )
    st.session_state.chunk_size = st.sidebar.slider('Chunk size:', 1000, 8000, value=2000, step=500)
    st.session_state.chunk_overlap = st.sidebar.slider('Chunk overlap:', 0, 1000, value=200, step=100)

    if st.sidebar.button("Process Data"):
        vector_embedding(documents)
        st.sidebar.write("Vector Store is Ready")
        st.session_state.conversation_chain = get_conversation_chain(st.session_state.vectors, model)

# Main content
st.title("McDonald's Store Performance Analysis")

# Split the screen into two columns
col1, col2 = st.columns(2)

# Left column: Data visualization
with col1:
    st.subheader("Store Locations")
    if 'df' in locals():
        fig = create_map(df)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Please upload a CSV file to view the map.")

# Right column: Chatbot
with col2:
    st.subheader("Ask for Analysis")

    # Session state variable
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    def handle_userinput():
        user_question = st.session_state.user_question
        if not st.session_state.get("embeddings_initialized", False):
            st.error("Please process the data first.")
            return
        
        if "conversation_chain" not in st.session_state:
            st.error("Please initialize the conversation chain first.")
            return

        start = time.process_time()
        
        response = st.session_state.conversation_chain({'question': user_question, 'chat_history': st.session_state.chat_history})
        ai_response = response['answer']

        # Process retrieved documents for potential analysis
        retrieved_docs = response['source_documents']
        df_retrieved = pd.DataFrame([doc.page_content.split('\n') for doc in retrieved_docs])
        if len(df_retrieved.columns) == 11:  # Ensure we have the expected number of columns
            df_retrieved.columns = ['reviewer_id', 'store_name', 'category', 'store_address', 'latitude', 'longitude', 'rating_count', 'review_time', 'review', 'rating', 'sentiment']
            
            # Convert rating to numeric, removing 'stars' and handling potential errors
            df_retrieved['rating'] = pd.to_numeric(df_retrieved['rating'].str.replace(' stars', '').str.replace(' star', ''), errors='coerce')
            
            # Add retrieved data statistics to session state for potential use in follow-up questions
            st.session_state.retrieved_data = {
                'avg_rating': df_retrieved['rating'].mean(),
                'common_comments': df_retrieved['review'].value_counts().head(5).to_dict(),
                'sentiment_distribution': df_retrieved['sentiment'].value_counts().to_dict(),
                'store_distribution': df_retrieved['store_name'].value_counts().to_dict()
            }
        else:
            st.session_state.retrieved_data = None

        # Add processing time information
        end = time.process_time()
        processing_time = end - start
        ai_response += f"\n\nProcessing time: {processing_time:.2f} seconds"

        st.session_state.chat_history.append({'human': user_question, 'AI': ai_response})
        st.session_state.user_question = ""
        st.session_state.response = response

    def display_chat_history():
        st.write(css, unsafe_allow_html=True)
        for message in reversed(st.session_state.chat_history):
            st.write(user_template.replace("{{MSG}}", message['human']), unsafe_allow_html=True)
            st.write(bot_template.replace("{{MSG}}", message['AI']), unsafe_allow_html=True)

    # Chat input
    st.text_input("Ask for analysis or suggestions:", key="user_question", on_change=handle_userinput)

    # Chat history
    chat_container = st.container()
    with chat_container:
        display_chat_history()

    # Relevant Data expander
    if 'response' in st.session_state:
        with st.expander("Relevant Data"):
            if "source_documents" in st.session_state.response:
                for i, doc in enumerate(st.session_state.response["source_documents"]):
                    st.write(f"Document {i+1}:")
                    st.write(doc.page_content)
                    st.write("--------------------------------")
            else:
                st.write("No relevant data found in the response.")

# Run the Streamlit app
if __name__ == '__main__':
    st.sidebar.markdown("---")
    st.sidebar.write("Powered by Groq and Streamlit")