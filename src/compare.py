from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def compare_questions_responses(df):
    if 'question' not in df.columns or 'response' not in df.columns:
        raise ValueError("Missing required columns: 'question' and/or 'response'")

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['question'].fillna('') + " " + df['response'].fillna(''))
    similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Self-comparison for simplicity, real use would compare with reference answers
    df['self_similarity'] = similarities.diagonal()
    return df
