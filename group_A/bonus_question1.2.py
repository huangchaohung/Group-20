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
