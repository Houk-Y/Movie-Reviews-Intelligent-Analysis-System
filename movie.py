import streamlit as st
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import matplotlib.pyplot as plt

# Download NLTK resources
nltk.download("vader_lexicon")


# Function to assign sentiment
def assign_sentiment(row):
    score = sid.polarity_scores(row)["compound"]
    if score > 0.05:
        return "Positive"
    elif score < -0.05:
        return "Negative"
    else:
        return "Neutral"


# Function to summarize positive and negative sentiments
def summarize_sentiments(df):
    positive_count = (df["Sentiment"] == "Positive").sum()
    negative_count = (df["Sentiment"] == "Negative").sum()
    neutral_count = (df["Sentiment"] == "Neutral").sum()

    st.write("## Sentiment Summary")
    st.write(f"Total Positive Sentiments: {positive_count}")
    st.write(f"Total Negative Sentiments: {negative_count}")
    st.write(f"Total Neutral Sentiments: {neutral_count}")

    # Plot pie chart
    labels = ["Positive", "Negative", "Neutral"]
    sizes = [positive_count, negative_count, neutral_count]
    explode = (0, 0, 0)  # explode 1st slice
    colors = ["red", "blue", "green"]  # Set custom colors
    fig1, ax1 = plt.subplots()
    ax1.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        shadow=True,
        startangle=140,
        wedgeprops={"edgecolor": "black"},
    )
    ax1.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig1)


# Load the data
@st.cache
def load_data():
    df = pd.read_csv("Test.csv")
    return df


# Sidebar to select number of rows to display
st.sidebar.title("Choose Dataset Size")
rows_to_display = st.sidebar.slider(
    "Select number of rows to display",
    min_value=100,
    max_value=1000,
    value=500,
    step=100,
)

# Main content
st.title("Sentiment Analysis with NLTK and Streamlit")

# Load the data
df = load_data()

# Subset the data based on user input
df_subset = df.head(rows_to_display)

# Perform sentiment analysis
sid = SentimentIntensityAnalyzer()
df_subset["Sentiment"] = df_subset["text"].apply(assign_sentiment)

# Display the DataFrame
st.write("## Data with Sentiment Analysis")
st.write(df_subset)

# Summarize sentiments
summarize_sentiments(df_subset)
