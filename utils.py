import pandas as pd
import plotly.graph_objects as go
from langchain.schema import HumanMessage, SystemMessage

def preprocess_dataframe_grouped(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataframe by cleaning and aggregating data,
    displaying the mode for each aspect ignoring zeros and giving the second highest mode.
    
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

    # Function to calculate the second highest mode, ignoring zeros
    def second_highest_mode(series):
        # Calculate the value counts, including zeros
        mode_counts = series.value_counts()
        
        # If the most common value is zero, we need to take the second highest mode
        if not mode_counts.empty and mode_counts.index[0] == 0:
            # If there's more than one unique value, take the second most frequent
            if len(mode_counts) > 1:
                return mode_counts.index[1], mode_counts.iloc[1]  # Second highest mode and its count
            else:
                return None, 0  # No valid second mode available
        else:
            # If the most common value is not zero, return it
            return mode_counts.index[0], mode_counts.iloc[0]  # Highest mode and its count


    # Group by store_address and calculate aggregates
    df_grouped = df.groupby('store_address').agg({
        'latitude': 'first',
        'longitude': 'first',
        'store_no': 'first',
        'actual_sentiment': [
            ('mode', lambda x: x.mode().iloc[0] if not x.empty else None),
            ('total_positive', lambda x: (x == 'positive').sum()),
            ('total_negative', lambda x: (x == 'negative').sum()),
            ('total_neutral', lambda x: (x == 'neutral').sum())
        ],
        'food': [('second_mode', second_highest_mode)],
        'service': [('second_mode', second_highest_mode)],
        'cleanliness': [('second_mode', second_highest_mode)],
        'price': [('second_mode', second_highest_mode)],
        'others': [('second_mode', second_highest_mode)]
    }).reset_index()
    
    # Flatten the column names
    df_grouped.columns = ['_'.join(col).strip() for col in df_grouped.columns.values]
    
    # Rename the columns for clarity
    df_grouped = df_grouped.rename(columns={
        'store_address_': 'store_address',  # Keep store_address as a column
        'latitude_first': 'latitude',
        'longitude_first': 'longitude',
        'store_no_first': 'store_no',
        'actual_sentiment_mode': 'actual_sentiment_mode',
        'actual_sentiment_total_positive': 'positive_review_count',
        'actual_sentiment_total_negative': 'negative_review_count',
        'actual_sentiment_total_neutral': 'neutral_review_count',
        'food_second_mode': 'food_mode_sentiment',
        'service_second_mode': 'service_mode_sentiment',
        'cleanliness_second_mode': 'cleanliness_mode_sentiment',
        'price_second_mode': 'price_mode_sentiment',
        'others_second_mode': 'others_mode_sentiment'
    })
    
    # Split the mode and count values for aspect sentiments
    for aspect in ['food_mode_sentiment', 'service_mode_sentiment', 'cleanliness_mode_sentiment', 'price_mode_sentiment', 'others_mode_sentiment']:
        df_grouped[[f'{aspect}_value', f'{aspect}_count']] = pd.DataFrame(df_grouped[aspect].tolist(), index=df_grouped.index)
        df_grouped.drop(columns=[aspect], inplace=True)

    # Map sentiment integers or strings to descriptive text
    sentiment_map = {
        1: 'Mostly Negative',
        2: 'Mostly Neutral',
        3: 'Mostly Positive',
        'negative': 'Mostly Negative',
        'neutral': 'Mostly Neutral',
        'positive': 'Mostly Positive'
    }

    # Apply mapping to each aspect sentiment column
    for aspect in ['food_mode_sentiment_value', 'service_mode_sentiment_value', 'cleanliness_mode_sentiment_value', 'price_mode_sentiment_value', 'others_mode_sentiment_value']:
        df_grouped[aspect] = df_grouped[aspect].map(sentiment_map).fillna('Not Available')
    
    return df_grouped

def create_map(df: pd.DataFrame) -> go.Figure:
    """
    Create an interactive map using Plotly with sentiment-based marker colors.
    
    Args:
        df (pd.DataFrame): The preprocessed dataframe.
    
    Returns:
        go.Figure: The Plotly figure object for the map.
    """
    # Map the general sentiment to text
    sentiment_map = {
        'negative': 'Mostly Negative',
        'neutral': 'Mostly Neutral',
        'positive': 'Mostly Positive'
    }
    
    # Replace the actual_sentiment_mode with descriptive text
    df['actual_sentiment_text'] = df['actual_sentiment_mode'].map(sentiment_map).fillna('Undefined')
    
    # Define color mapping for general sentiments
    color_map = {
        'Mostly Positive': 'green',
        'Mostly Neutral': 'yellow',
        'Mostly Negative': 'red',
        'Undefined': 'gray'  # Default color for missing or undefined values
    }
    
    # Map descriptive sentiments to colors
    colors = df['actual_sentiment_text'].map(color_map)
    
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
        hovertemplate=(
            "<b>%{text}</b><br>" +
            "Store No: %{customdata[0]}<br>" +
            "General Sentiment: %{customdata[1]}<br>" +
            "Food: %{customdata[2]} (%{customdata[3]} reviews)<br>" +
            "Service: %{customdata[4]} (%{customdata[5]} reviews)<br>" +
            "Cleanliness: %{customdata[6]} (%{customdata[7]} reviews)<br>" +
            "Price: %{customdata[8]} (%{customdata[9]} reviews)<br>" +
            "Others: %{customdata[10]} (%{customdata[11]} reviews)<extra></extra>"
        ),
        customdata=df[['store_no', 'actual_sentiment_text',
                       'food_mode_sentiment_value', 'food_mode_sentiment_count',
                       'service_mode_sentiment_value', 'service_mode_sentiment_count',
                       'cleanliness_mode_sentiment_value', 'cleanliness_mode_sentiment_count',
                       'price_mode_sentiment_value', 'price_mode_sentiment_count',
                       'others_mode_sentiment_value', 'others_mode_sentiment_count']].fillna('Not Available')
    ))
    
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(
            zoom=3,
            center=go.layout.mapbox.Center(lat=39.8283, lon=-98.5795)
        ),
        showlegend=False,
        height=600,
        margin={"r": 0, "t": 0, "l": 0, "b": 0}
    )
    
    # Add a legend for general sentiments
    for sentiment, color in color_map.items():
        fig.add_trace(go.Scattermapbox(
            lat=[None],  # Empty lat/lon to only show legend entries
            lon=[None],
            mode='markers',
            marker=dict(size=10, color=color),
            name=sentiment,
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