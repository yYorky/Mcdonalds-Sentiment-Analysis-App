import pandas as pd
import plotly.graph_objects as go
from langchain.schema import HumanMessage, SystemMessage

def preprocess_dataframe_grouped(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataframe by cleaning and aggregating data.
    
    Args:
        df (pd.DataFrame): The input dataframe.
    
    Returns:
        pd.DataFrame: The preprocessed dataframe.
    """
    # Create a copy of the dataframe to avoid SettingWithCopyWarning
    df = df.copy()
    
    # # Define a mapping for star ratings
    # rating_map = {
    #     '1 star': 1.0,
    #     '2 stars': 2.0,
    #     '3 stars': 3.0,
    #     '4 stars': 4.0,
    #     '5 stars': 5.0
    # }
    
    # # Map ratings using the defined dictionary
    # df['rating'] = df['rating'].replace(rating_map)
    
    # # Drop rows with NaN ratings
    # df = df.dropna(subset=['rating'])
    
    # Convert latitude and longitude to numeric
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    
    # # Drop rows with NaN latitude or longitude
    # df = df.dropna(subset=['latitude', 'longitude'])
    
    # # Recalculate rating_count for each store_address
    # df['rating_count'] = df.groupby('store_address')['rating'].transform('count')
    
    # Group by store_address and calculate weighted average rating
    df_grouped = df.groupby('store_address').agg({
        'latitude': 'first',
        'longitude': 'first',
        'store_no': 'first',
        # 'rating_count': 'first',
        # 'rating': lambda x: round((x * df.loc[x.index, 'rating_count']).sum() / df.loc[x.index, 'rating_count'].sum(), 2) if df.loc[x.index, 'rating_count'].sum() != 0 else round(x.mean(), 2),
        'actual_sentiment': lambda x: x.mode().iloc[0] if not x.empty else None
    }).reset_index()
    
    return df_grouped

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataframe by cleaning and aggregating data. Works for csv file 'dashboard_data'
    
    Args:
        df (pd.DataFrame): The input dataframe.
    
    Returns:
        pd.DataFrame: The preprocessed dataframe.
    """
    # Create a copy of the dataframe to avoid SettingWithCopyWarning
    df = df.copy()
       
    # Convert latitude and longitude to numeric
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
            
    return df



def create_map(df: pd.DataFrame) -> go.Figure:
    """
    Create an interactive map using Plotly with sentiment-based marker colors.
    
    Args:
        df (pd.DataFrame): The preprocessed dataframe.
    
    Returns:
        go.Figure: The Plotly figure object for the map.
    """
    # Define color mapping for sentiments
    color_map = {
        'positive': 'green',
        'neutral': 'yellow',
        'negative': 'red'
    }
    
    # Map sentiments to colors
    colors = df['actual_sentiment'].map(color_map)
    
    fig = go.Figure()
    fig.add_trace(go.Scattermapbox(
        lat=df['latitude'],
        lon=df['longitude'],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=15,
            color=colors,
            opacity=0.7,
            sizemin=5,
            sizemode='area'
        ),
        text=df['store_address'],
        hoverinfo='text',
        hovertemplate=
        "<b>%{text}</b><br>" +
        "Store No: %{customdata[0]}<br>" +
        "General Sentiment: Mostly %{customdata[1]}<extra></extra>",
        customdata=df[['store_no', 'actual_sentiment']]
    ))
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(
            zoom=3,
            center=go.layout.mapbox.Center(lat=39.8283, lon=-98.5795)
        ),
        showlegend=False,
        height=600,
        margin={"r":0,"t":0,"l":0,"b":0}
    )
    
    # Add a legend
    for sentiment, color in color_map.items():
        fig.add_trace(go.Scattermapbox(
            lat=[None],
            lon=[None],
            mode='markers',
            marker=dict(size=10, color=color),
            name=sentiment.capitalize(),
            showlegend=True
        ))
    
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    
    return fig

def get_conversational_response(llm, original_response: str, user_question: str) -> str:
    """
    Generate a conversational response using the language model.
    
    Args:
        llm: The language model instance.
        original_response (str): The original analysis response.
        user_question (str): The user's question.
    
    Returns:
        str: The conversational response.
    """
    system_message = SystemMessage(content=
                                   "You are a helpful assistant. \
                                    Take the following analysis and rephrase it into a friendly, conversational response. \
                                    Make sure to address the user's question directly. \
                                    Make sure to use information from the original analysis only.")
    human_message = HumanMessage(content=f"User question: {user_question}\n\nOriginal analysis: {original_response}\n\nPlease rephrase this into a conversational response:")
    response = llm.invoke([system_message, human_message])
    return response.content

def calculate_store_metrics(df: pd.DataFrame) -> dict:
    """
    Calculate various metrics for the stores.
    
    Args:
        df (pd.DataFrame): The preprocessed dataframe.
    
    Returns:
        dict: A dictionary containing calculated metrics.
    """
    metrics = {
        "total_stores": len(df),
        "average_rating": df['rating'].mean(),
        "highest_rated_store": df.loc[df['rating'].idxmax(), 'store_address'],
        "lowest_rated_store": df.loc[df['rating'].idxmin(), 'store_address'],
        "most_reviewed_store": df.loc[df['rating_count'].idxmax(), 'store_address'],
        "sentiment_distribution": df['sentiment'].value_counts().to_dict()
    }
    return metrics

def generate_insights(df: pd.DataFrame) -> str:
    """
    Generate insights based on the store data.
    
    Args:
        df (pd.DataFrame): The preprocessed dataframe.
    
    Returns:
        str: A string containing insights about the data.
    """
    metrics = calculate_store_metrics(df)
    
    insights = f"""
    Based on the analysis of {metrics['total_stores']} McDonald's stores:

    1. Overall Performance:
       - The average rating across all stores is {metrics['average_rating']:.2f} stars.
       - The highest-rated store is located at: {metrics['highest_rated_store']}
       - The lowest-rated store is located at: {metrics['lowest_rated_store']}

    2. Customer Engagement:
       - The store with the most reviews is located at: {metrics['most_reviewed_store']}

    3. Sentiment Analysis:
       - Sentiment distribution across stores:
         {', '.join([f"{k}: {v}" for k, v in metrics['sentiment_distribution'].items()])}

    These insights provide a high-level overview of the McDonald's store performance and customer sentiment.
    For more detailed analysis or specific questions about the data, please feel free to ask.
    """
    
    return insights