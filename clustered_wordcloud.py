import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from wordcloud import WordCloud
from keybert import KeyBERT
from streamlit_option_menu import option_menu
import re

# === Function to Generate Enhanced Clustered WordCloud ===
def generate_enhanced_clustered_wordcloud(df, n_clusters=5):
    st.title('📌 Clustered WordClouds with Insights')
    
    # Vectorize text using TF-IDF
    text_data = df['question'].tolist()
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(text_data)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)

    # Extract top terms for each cluster
    terms = vectorizer.get_feature_names_out()
    clustered_text = {i: [] for i in range(n_clusters)}
    
    for i, label in enumerate(kmeans.labels_):
        clustered_text[label].append(text_data[i])

    # Initialize KeyBERT for topic labeling
    kw_model = KeyBERT()
    cluster_keywords = {}

    for cluster_id, texts in clustered_text.items():
        combined_text = ' '.join(texts)
        keywords = kw_model.extract_keywords(combined_text, top_n=3)
        label = ', '.join([kw[0] for kw in keywords])
        cluster_keywords[cluster_id] = label

    # Sidebar Navigation for Clusters
    selected_cluster = option_menu(
        'Select Cluster',
        [f'Cluster {i}: {cluster_keywords[i]}' for i in range(n_clusters)],
        icons=['box'] * n_clusters,
        menu_icon='cast',
        default_index=0,
        orientation='horizontal'
    )

    # Extract selected cluster index
    selected_index = int(re.search(r'\d+', selected_cluster).group())

    # Display WordCloud and Insights
    st.subheader(f'🌐 WordCloud for {selected_cluster}')
    combined_text = ' '.join(clustered_text[selected_index])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(combined_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

    # Display Cluster Insights
    st.subheader('📊 Cluster Insights')
    st.write(f'**Top Keywords:** {cluster_keywords[selected_index]}')
    st.write(f'**Total Messages:** {len(clustered_text[selected_index])}')
    st.write('**Sample Messages:**')
    for msg in clustered_text[selected_index][:5]:
        st.write(f'- {msg}')

    return clustered_text, cluster_keywords