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

# Generate Word Cloud for each sentiment
sentiments = ['Positive', 'Neutral', 'Negative']
for sentiment in sentiments:
    text = ' '.join(df[df['sentiment'] == sentiment]['cleaned_review'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(15,7.5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for {sentiment} Reviews')
    plt.show()

# Topic Modeling using LDA
# Vectorize the text data
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
tf = tf_vectorizer.fit_transform(df['cleaned_review'])

# Fit the LDA model
num_topics = 5  # You can adjust the number of topics
lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda.fit(tf)

# Display the top words for each topic
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx+1}:")
        print(", ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))
        print()

tf_feature_names = tf_vectorizer.get_feature_names_out()
display_topics(lda, tf_feature_names, no_top_words=10)

# Assign topics to reviews
topic_values = lda.transform(tf)
df['topic'] = topic_values.argmax(axis=1) + 1  # Adding 1 to start topics from 1

# Visualize Topics Distribution
plt.figure(figsize=(8,6))
sns.countplot(x='topic', data=df)
plt.title('Topics Distribution')
plt.show()

# Save the processed data for the dashboard
df.to_csv('processed_reviews.csv', index=False)

# Create a Dashboard (Optional)
# You can use Streamlit to create an interactive dashboard
# Save the following code in a file named 'streamlit_app.py'

'''
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load processed data
df = pd.read_csv('processed_reviews.csv')

st.title('Customer Reviews Dashboard')

# Sentiment Distribution
st.subheader('Sentiment Distribution')
fig1, ax1 = plt.subplots()
sns.countplot(x='sentiment', data=df, order=['Positive', 'Neutral', 'Negative'], ax=ax1)
st.pyplot(fig1)

# Topics Distribution
st.subheader('Topics Distribution')
fig2, ax2 = plt.subplots()
sns.countplot(x='topic', data=df, ax=ax2)
st.pyplot(fig2)

# Word Clouds
st.subheader('Word Clouds by Sentiment')
sentiment = st.selectbox('Select Sentiment', ['Positive', 'Neutral', 'Negative'])
from wordcloud import WordCloud

text = ' '.join(df[df['sentiment'] == sentiment]['cleaned_review'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
fig3, ax3 = plt.subplots(figsize=(10, 5))
ax3.imshow(wordcloud, interpolation='bilinear')
ax3.axis('off')
st.pyplot(fig3)

# Display sample reviews
st.subheader('Sample Reviews')
num_reviews = st.slider('Number of Reviews to Display', min_value=1, max_value=20, value=5)
sample_reviews = df[['name', 'country', 'date_time', 'sentiment', 'review_text']].sample(num_reviews)
st.write(sample_reviews)
'''

# Instructions:
# 1. Save the above Streamlit code in a file named 'streamlit_app.py'.
# 2. Run the dashboard using the command: streamlit run streamlit_app.py
