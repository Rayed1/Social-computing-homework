import sqlite3
import pandas as pd
import os
from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS
import re

os.system('clear')

DB_FILE = "database.sqlite"

try:
    conn = sqlite3.connect(DB_FILE)
    print("SQLite Database connection successful")
except Exception as e:
    print(f"Uh oh '{e}'")

posts_df = pd.read_sql_query("SELECT id, content FROM posts", conn)

print(f'\nTotal posts in database: {len(posts_df)}')
print(f'\n{"="*80}')
print('Sample Posts from Database:')
print(f'{"="*80}')
for idx, row in posts_df.head(20).iterrows():
    print(f"\nPost {row['id']}: {row['content'][:100]}...")  # Show first 100 chars

print(f'\n{"="*80}\n')

conn.close()

#Preprocess
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOPWORDS and len(word) > 3]
    return tokens

documents = posts_df['content'].apply(preprocess_text).tolist()

documents = [doc for doc in documents if len(doc) > 0]

print(f'Documents after preprocessing: {len(documents)}')

#dictionary and corpus
dictionary = corpora.Dictionary(documents)

dictionary.filter_extremes(no_below=5, no_above=0.5)

corpus = [dictionary.doc2bow(doc) for doc in documents]

print(f'Dictionary size: {len(dictionary)}')
print(f'Corpus size: {len(corpus)}')

#LDA
lda_model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=10,
    random_state=42,
    passes=10,
    alpha='auto'
)

print(f'\n{"="*80}')
print(f'Top 10 Most Popular Topics on the Platform')
print(f'{"="*80}\n')

#topic distribution across documents
topic_counts = [0] * 10
for doc_bow in corpus:
    doc_topics = lda_model.get_document_topics(doc_bow)
    if doc_topics:
        #dominant topic for this document
        dominant_topic = max(doc_topics, key=lambda x: x[1])[0]
        topic_counts[dominant_topic] += 1

#Sort by frequency
topic_freq = [(i, topic_counts[i]) for i in range(10)]
topic_freq.sort(key=lambda x: x[1], reverse=True)

#Display topics sorted by frequency
for rank, (topic_id, count) in enumerate(topic_freq, 1):
    topic_words = lda_model.show_topic(topic_id, topn=5)
    words = [word for word, prob in topic_words]
    print(f'{rank}. Topic {topic_id} ({count} posts): {", ".join(words)}')

    # ==================== Exercise 4.2: Sentiment Analysis ====================

print('\nExercise 4.2: Sentiment Analysis\n')

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

try:
    nltk.download('vader_lexicon', quiet=True)
    print("VADER lexicon loaded successfully\n")
except:
    print("Error loading VADER\n")

#Initialize VADER
sia = SentimentIntensityAnalyzer()

post_topics = {}
for idx, doc_bow in enumerate(corpus):
    doc_topics = lda_model.get_document_topics(doc_bow)
    if doc_topics:
        dominant_topic = max(doc_topics, key=lambda x: x[1])[0]
        post_id = posts_df.iloc[idx]['id']
        post_topics[post_id] = dominant_topic

print(f"Mapped {len(post_topics)} posts to topics\n")

post_sentiments = []

for idx, row in posts_df.iterrows():
    post_id = row['id']
    content = row['content']
    
    #vader sentiment scores
    sentiment = sia.polarity_scores(content)
    
    #topic for this post
    topic = post_topics.get(post_id, None)
    
    post_sentiments.append({
        'post_id': post_id,
        'content': content,
        'topic': topic,
        'compound': sentiment['compound'],
        'positive': sentiment['pos'],
        'negative': sentiment['neg'],
        'neutral': sentiment['neu']
    })

posts_sentiment_df = pd.DataFrame(post_sentiments)
print(f"Analyzed {len(posts_sentiment_df)} posts\n")

#analyze comments
conn = sqlite3.connect(DB_FILE)
comments_df = pd.read_sql_query("SELECT id, post_id, content FROM comments", conn)
conn.close()

comment_sentiments = []

for idx, row in comments_df.iterrows():
    content = row['content']
    sentiment = sia.polarity_scores(content)
    
    comment_sentiments.append({
        'comment_id': row['id'],
        'post_id': row['post_id'],
        'compound': sentiment['compound'],
        'positive': sentiment['pos'],
        'negative': sentiment['neg'],
        'neutral': sentiment['neu']
    })

comments_sentiment_df = pd.DataFrame(comment_sentiments)
print(f"Analyzed {len(comments_sentiment_df)} comments\n")


print('\nOverall Platform Sentiment\n')


all_compound_scores = list(posts_sentiment_df['compound']) + list(comments_sentiment_df['compound'])
overall_mean = sum(all_compound_scores) / len(all_compound_scores)

if overall_mean > 0.05:
    tone = "Positive"
elif overall_mean < -0.05:
    tone = "Negative"
else:
    tone = "Neutral"

print(f"Total Posts: {len(posts_sentiment_df)}")
print(f"Total Comments: {len(comments_sentiment_df)}")
print(f"Average Sentiment Score: {overall_mean:.4f}")
print(f"Overall Platform Tone: {tone}")

def categorize_sentiment(score):
    if score > 0.05:
        return "Positive"
    elif score < -0.05:
        return "Negative"
    else:
        return "Neutral"

posts_sentiment_df['category'] = posts_sentiment_df['compound'].apply(categorize_sentiment)
comments_sentiment_df['category'] = comments_sentiment_df['compound'].apply(categorize_sentiment)

print(f"\nPosts Distribution:")
post_dist = posts_sentiment_df['category'].value_counts()
for cat, count in post_dist.items():
    percentage = (count / len(posts_sentiment_df)) * 100
    print(f"  {cat}: {count} ({percentage:.1f}%)")

print(f"\nComments Distribution:")
comment_dist = comments_sentiment_df['category'].value_counts()
for cat, count in comment_dist.items():
    percentage = (count / len(comments_sentiment_df)) * 100
    print(f"  {cat}: {count} ({percentage:.1f}%)")


print('\nSentiment Analysis by Topic (from Exercise 4.1)\n')

#average sentiment per topic
topic_sentiments = posts_sentiment_df.groupby('topic')['compound'].agg(['mean', 'count', 'std'])

for rank, (topic_id, post_count) in enumerate(topic_freq, 1):
    if topic_id in topic_sentiments.index:
        topic_words = lda_model.show_topic(topic_id, topn=5)
        words = [word for word, prob in topic_words]

        avg_sentiment = topic_sentiments.loc[topic_id, 'mean']
        std_sentiment = topic_sentiments.loc[topic_id, 'std']
        
        if avg_sentiment > 0.05:
            category = "Positive"
        elif avg_sentiment < -0.05:
            category = "Negative"
        else:
            category = "Neutral"
        
        print(f"Topic {rank}: {', '.join(words)}")
        print(f"  Posts: {post_count}")
        print(f"  Avg Sentiment: {avg_sentiment:.4f} ({category})")
        print(f"  Std Dev: {std_sentiment:.4f}")
        print()
