# Import necessary libraries
import pandas as pd
import numpy as np
import nltk
import spacy
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from collections import Counter

# Sentiment analysis libraries
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# Alternatively, you can use TextBlob or transformers for more advanced analysis

# Download NLTK data files
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('vader_lexicon')  # For VADER sentiment analyzer

# Load English tokenizer, POS tagger, etc.
nlp = spacy.load('en_core_web_sm')

# Read the CSV file
df = pd.read_csv('../data/reviews_for_classification.csv')

# Preview the data
print(df.head())

# Combine 'review_head' and 'review_body' into a single 'review_text' column
df['review_text'] = df['review_body'].fillna('')

# def preprocess_text(text):
#     # Lowercase
#     text = text.lower()
#     # Remove URLs
#     text = re.sub(r'http\S+', '', text)
#     # Remove HTML tags
#     text = re.sub(r'<.*?>', '', text)
#     # Remove non-alphabetic characters
#     text = re.sub(r'[^a-zA-Z\s]', '', text)
#     # Tokenize
#     tokens = nltk.word_tokenize(text)
#     # Remove stopwords and lemmatize
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]
#     # Rejoin into string
#     text = ' '.join(tokens)
#     return text

# Apply preprocessing to the 'review_text' column
df['cleaned_review'] = df['review_text']

# Sentiment Analysis using VADER
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    sentiment_score = sia.polarity_scores(text)
    compound_score = sentiment_score['compound']
    df['sentiment_score'] = compound_score  # Store the actual score
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Apply sentiment analysis
df['sentiment'] = df['cleaned_review'].apply(analyze_sentiment)

# Visualize Sentiment Distribution
plt.figure(figsize=(8,6))
sns.countplot(x='sentiment', data=df, order=['Positive', 'Neutral', 'Negative'])
plt.title('Sentiment Distribution')
plt.show()

def add_random_dates(df, start='2023-01-01', end='2023-12-31'):
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    
    # Generate random dates
    date_range = (end_date - start_date).days
    random_days = np.random.randint(0, date_range, size=len(df))
    random_dates = start_date + pd.to_timedelta(random_days, unit='D')
    
    df['date'] = random_dates
    return df.sort_values('date')

# Apply the function
df = add_random_dates(df)

def create_sentiment_dashboard(df):
    # 1. Time Series of Sentiment Distribution
    sentiment_over_time = df.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
    sentiment_over_time_pct = sentiment_over_time.div(sentiment_over_time.sum(axis=1), axis=0)

    # 2. Extract top themes/keywords for each sentiment
    def get_top_terms(text_series, n=10):
    # Tokenize and clean the words
        adjectives = []
        for text in text_series:
            # Process text with spaCy
            doc = nlp(text.lower())
            # Only keep adjectives (ADJ tag)
            adj_tokens = [token.text for token in doc if token.pos_ == 'ADJ' 
                            and token.text not in stopwords.words('english')]
            adjectives.extend(adj_tokens)

        # Count word frequencies
        word_freq = Counter(adjectives)

        # Get the top N words and their counts
        top_words = pd.DataFrame(word_freq.most_common(n), columns=['term', 'count'])
        return top_words

# Optional: Add this debug code to see what adjectives are being foun

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Sentiment Trends Over Time', 'Top Themes by Sentiment',
                       'Sentiment Distribution'),
        specs=[[{"type": "scatter"}, {"type": "bar"}],
               [{"type": "pie"}, None]]
    )

    # 1. Sentiment Trends Over Time
    for sentiment in sentiment_over_time_pct.columns:
        fig.add_trace(
            go.Scatter(x=sentiment_over_time_pct.index, 
                        y=sentiment_over_time_pct[sentiment],
                        name=sentiment,
                        mode='lines'),
            row=1, col=1
        )

    # 2. Top Themes Bar Chart
    for sentiment in ['Positive', 'Negative', 'Neutral']:
        top_terms = get_top_terms(df[df['sentiment'] == sentiment]['cleaned_review'])
        fig.add_trace(
            go.Bar(x=top_terms['term'][:5],
                    y=top_terms['count'][:5],
                    name=f'{sentiment} themes'),
            row=1, col=2
        )

    # 3. Overall Sentiment Distribution (Pie Chart)
    sentiment_counts = df['sentiment'].value_counts()
    fig.add_trace(
        go.Pie(labels=sentiment_counts.index,
                values=sentiment_counts.values,
                hole=0.3),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        height=800,
        width=1200,
        showlegend=True,
        title_text="Sentiment Analysis Dashboard",
        template="plotly_white"
    )

    return fig

# Create and display the dashboard
dashboard = create_sentiment_dashboard(df)
dashboard.show()