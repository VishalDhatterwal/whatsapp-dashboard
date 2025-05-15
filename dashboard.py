import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.run(asyncio.sleep(0))
    
import streamlit as st
import pandas as pd
from io import BytesIO
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from src.analyzer import add_sentiment, add_emotions, classify_message_type,extract_response_emojis
from src.cleaner import clean_chat_keep_emojis as clean_chat
from clustered_wordcloud import generate_enhanced_clustered_wordcloud
from src.parser import parse_uploaded_excel
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from keybert import KeyBERT
from collections import Counter
import re
from src.emoji_utils import get_emoji_stats
import plotly.express as px
from streamlit_plotly_events import plotly_events
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# ✅ Specify the local cache path of the model
model_name = "distilbert-base-uncased-finetuned-sst-2-english"

# ✅ Initialize tokenizer and model manually
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# ✅ Create pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


st.set_page_config(page_title="AI Coach Dashboard", layout="wide")
st.title("🤖 AI Expert Coach Feedback Dashboard")

@st.cache_data
def fig_to_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return buf

#generate wordcloud
def generate_wordcloud(text, title="Word Cloud", ngram_range=(1, 1), stopwords=None):
    if not text.strip():
        return None

    # Use CountVectorizer to extract word/phrase frequencies
    vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words=stopwords)
    X = vectorizer.fit_transform([text])
    word_freq = dict(zip(vectorizer.get_feature_names_out(), X.toarray()[0]))

    # Generate word cloud
    wordcloud = WordCloud(
        width=900,
        height=400,
        background_color='white',
        colormap='plasma'
    ).generate_from_frequencies(word_freq)

    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title)
    ax.axis('off')
    return fig


uploaded_file = st.file_uploader("Upload the Excel file (AI Expert Coach Feedback)", type="xlsx")


if uploaded_file:
    df = parse_uploaded_excel(uploaded_file)

    # Apply your logic
    df = clean_chat(df)

    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df = df.dropna(subset=['timestamp'])



    df = add_sentiment(df, text_column='question')
    df = add_emotions(df, text_column='question')
    df = classify_message_type(df)
    df = extract_response_emojis(df)

    st.success("✅ File processed successfully!")
else:
    st.warning("Please upload an Excel or .XLSX file to begin.")
    st.stop()

# --- Sidebar Filters (Optimized with Caching) ---
@st.cache_data
def get_sidebar_data(df):
    type_options = df['type'].dropna().unique().tolist()
    min_date, max_date = df['timestamp'].min().date(), df['timestamp'].max().date()
    all_users = sorted(df['user'].dropna().unique())
    return type_options, min_date, max_date, all_users

# Retrieve cached sidebar data
type_options, min_date, max_date, all_users = get_sidebar_data(df)

# Sidebar for Message Type
st.sidebar.subheader('🧺 Filter by Message Type')
selected_types = st.sidebar.multiselect('Select type(s)', options=type_options, default=type_options)

# Sidebar for Date Range
st.sidebar.subheader('📆 Date Range')
start_date, end_date = st.sidebar.date_input('Select range', [min_date, max_date])

# Sidebar for User Selection with Search
st.sidebar.subheader('👤 Select User')
user_search = st.sidebar.text_input('🔍 Search User')
filtered_users = ['All'] + [u for u in all_users if user_search.lower() in u.lower()]
selected_user = st.sidebar.selectbox('User', filtered_users)  

# Apply filters
filtered_df = df[(df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)]
if selected_user != "All":
    filtered_df = filtered_df[filtered_df['user'] == selected_user]

filtered_df = filtered_df[filtered_df['type'].isin(selected_types)]


try:
    st.subheader("📈 User Interaction Summary")

   
    
    @st.cache_data
    def extract_emojis(text):
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"
            u"\U0001F1E0-\U0001F1FF"
            u"\U00002500-\U00002BEF"
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        return ''.join(emoji_pattern.findall(str(text)))

    # Apply to question column (user inputs)
    emoji_series = filtered_df['question'].dropna().map(extract_emojis)
    total_emojis = sum(len(e) for e in emoji_series)

    total_questions = len(filtered_df)
    total_users = filtered_df['user'].nunique()
    avg_qs_per_user = round(total_questions / total_users, 2) if total_users else 0

    total_links = filtered_df['question'].str.contains("http", na=False).sum()
    top_user = filtered_df['user'].value_counts().idxmax()
    top_user_qs = filtered_df['user'].value_counts().max()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("❓ Questions", total_questions)
    col2.metric("😄 Emojis", total_emojis)
    col3.metric("🔗 Links", total_links)
    col4.metric("👥 Users", total_users)
    col5.metric("📊 Avg Questions/User", avg_qs_per_user)

    st.markdown(f"**🏆 Most Active User:** `{top_user}` with `{top_user_qs}` questions")

except Exception as e:
    st.error(f"⚠️ Error in summary block: {e}")

#gnrate wordcloud
# === Word Cloud Generator with Lazy Loading ===
st.subheader("☁️ Word Cloud Generator")

# --- UI Elements for WordCloud options ---
text_source = st.selectbox("Select message type", ["User Questions", "AI Responses", "Both"])
ngram_choice = st.selectbox("Display as", ["Single Words", "2-word Phrases", "3-word Phrases"])

# --- Stopwords Setup ---
custom_stopwords = list(set(STOPWORDS).union({
    "please", "thanks", "hi", "okay", "ok", "hello", "yeah", "sure", "hey"
}))

# --- Function to generate word cloud (Cached for performance) ---
@st.cache_data
def generate_wordcloud(text, title="Word Cloud", ngram_range=(1, 1), stopwords=None):
    if not text.strip():
        return None
    vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words=stopwords)
    X = vectorizer.fit_transform([text])
    word_freq = dict(zip(vectorizer.get_feature_names_out(), X.toarray()[0]))

    wordcloud = WordCloud(
        width=900,
        height=400,
        background_color='white',
        colormap='plasma'
    ).generate_from_frequencies(word_freq)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title)
    ax.axis('off')
    return fig

# --- Lazy Loading Logic: Only generate when button is clicked ---
if st.button("Generate WordCloud 🌐"):
    st.write("Button Clicked!")  # For debugging
    # Select text column(s)
    if text_source == "User Questions":
        messages = df['question'].dropna()
    elif text_source == "AI Responses":
        messages = df['response'].dropna()
    else:
        messages = pd.concat([df['question'], df['response']]).dropna()

    # Set n-gram range
    ngram_range = (1, 1) if ngram_choice == "Single Words" else (2, 2) if ngram_choice == "2-word Phrases" else (3, 3)

    # Build the text and generate word cloud
    text = " ".join(messages)
    fig = generate_wordcloud(text, f"{text_source} - {ngram_choice}", ngram_range, custom_stopwords)

    if fig:
        with st.spinner("Generating word cloud..."):
            st.pyplot(fig)
        if st.download_button("📥 Download Word Cloud as PNG", data=fig_to_bytes(fig), file_name="wordcloud.png"):
            st.success("✅ Downloaded successfully!")
    else:
        st.info("No valid text found for generating the word cloud.")
else:
    st.info("Click 'Generate WordCloud 🌐' to display the word cloud.")



#word explorer from wordcloud
st.subheader("🧠 Word Context Explorer")

# Combine all text from user questions
all_text = " ".join(filtered_df['question'].dropna()).lower()
all_words = list(set(all_text.split()))
selected_word = st.selectbox("Choose a word to explore", sorted(all_words))

# Filter rows where the selected word appears in the question
context_df = filtered_df[filtered_df['question'].str.contains(fr'\b{selected_word}\b', case=False, na=False)]

st.write(f"Showing user questions containing the word: **{selected_word}**")
st.dataframe(context_df[['timestamp', 'user', 'question', 'response']])

#--------------------------------------------------------------------------
# === Emotion Distribution ===
st.subheader("🎉 Top Emojis Used by AI")

emoji_counts = get_emoji_stats(filtered_df['response'])

if emoji_counts:
    emoji_df = pd.DataFrame(emoji_counts.items(), columns=['Emoji', 'Count']).sort_values(by='Count', ascending=False)
    st.dataframe(emoji_df)
    st.bar_chart(emoji_df.set_index('Emoji'))
else:
    st.info("No emojis found in the AI response.")

# === Sentiment Distribution ===
st.subheader("😊 Sentiment Distribution")
if 'sentiment' in filtered_df.columns:
    sentiment_counts = filtered_df['sentiment'].value_counts()
    st.bar_chart(sentiment_counts)
else:
    st.info("No sentiment data available.")


# --- Simple Conversation Drill-Down ---
st.subheader("🧵 Conversation Drill-Down")

if selected_user != "All":
    convo_df = df[df['user'] == selected_user].sort_values(by='timestamp')
    
    if not convo_df.empty:
        for _, row in convo_df.iterrows():
            with st.container():
                st.markdown(f"**🕒 {row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}**")
                st.markdown(f"**🙋‍♂️ User:** {row['question']}")
                st.markdown(f"**🤖 Bot:** {row['response']}")
                st.markdown("---")
    else:
        st.info("No conversation history found for the selected user.")
else:
    st.info("Please select a specific user to view their conversation history.")





# === Top Users by Question Volume ===
st.subheader("👤 Top Users (By Question Count)")
top_users = filtered_df['user'].value_counts().head(10)
st.bar_chart(top_users)

# === Daily Activity ===
st.subheader("📈 Daily Question Volume")
daily_activity = filtered_df.groupby(filtered_df['timestamp'].dt.date).size()
st.line_chart(daily_activity)


# Sentiment analysis
st.subheader("📈 Sentiment Trends")
daily_sentiment = filtered_df.groupby([filtered_df['timestamp'].dt.date, 'sentiment']).size().unstack().fillna(0)
st.line_chart(daily_sentiment)


# --- Topic Extraction with KeyBERT ---
st.subheader("🎯 Top Topics and Keywords")

# Combine all user questions into a single string
messages = " ".join(filtered_df['question'].dropna().tolist())

# Initialize KeyBERT model
kw_model = KeyBERT()

# Extract top 10 keywords
keywords = kw_model.extract_keywords(messages, top_n=10)

# Display extracted keywords with scores
st.markdown("### 🗣️ Key Topics in Patient Messages")
for kw, score in keywords:
    st.write(f"- {kw} ({score:.2f})")

# --- Visualization Section ---
visualization_choice = st.radio("How would you like to visualize the key topics?", ["Word Cloud", "Bar Chart"])
custom_stopwords = {"please", "thanks", "hi", "okay", "ok", "hello", "yeah", "sure", "hey"}

# Word cloud generator

def generate_wordcloud(text, title="Word Cloud", stopwords=None):
    if not text.strip():
        return None

    wordcloud = WordCloud(
        width=900,
        height=400,
        background_color='white',
        colormap='plasma',
        stopwords=stopwords
    ).generate(text)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title, fontsize=18)
    ax.axis('off')
    return fig

# Convert plot to downloadable bytes


def fig_to_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return buf

# Generate visualization
if visualization_choice == "Word Cloud":
    text = " ".join([kw[0] for kw in keywords])
    fig = generate_wordcloud(text, title="Top Keywords in Questions", stopwords=custom_stopwords)
    st.pyplot(fig)
    st.download_button("📥 Download Word Cloud", data=fig_to_bytes(fig), file_name="keywords_wordcloud.png")
else:
    keywords_df = pd.DataFrame(keywords, columns=["Keyword", "Score"])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(keywords_df["Keyword"], keywords_df["Score"], color='skyblue')
    ax.set_title("Top Keywords in Patient Messages")
    ax.set_xlabel("Score")
    st.pyplot(fig)
    st.download_button("📥 Download Bar Chart", data=fig_to_bytes(fig), file_name="keywords_barchart.png")



st.subheader("🔍 Clickable Drill-Down by User")

# Count top users
top_users = df['user'].value_counts().reset_index()
top_users.columns = ['user', 'question_count']

# Plot clickable bar chart
fig = px.bar(top_users, x='user', y='question_count')
fig.update_layout(clickmode='event+select')

# Render once and assign a key
plotly_chart = st.plotly_chart(fig, use_container_width=True, key="top_users_chart")

# Access selected data via session_state or st.session_state workaround
click_data = plotly_chart  # There's no selected_data directly returned, use events via st.session_state

# Handle click event manually via workaround
if 'clicked_user' not in st.session_state:
    st.session_state.clicked_user = None

# Optional workaround for persistent selection: let user manually select a user
selected_user = st.selectbox("Or select a user manually to drill down:", top_users['user'])

if selected_user:
    st.session_state.clicked_user = selected_user

if st.session_state.clicked_user:
    selected_user = st.session_state.clicked_user
    st.markdown(f"### 🔍 Conversation Drilldown for **{selected_user}**")

    # Filter data for that user
    user_df = df[df['user'] == selected_user].reset_index(drop=True)
     
    # ✅ Show phone and question count info
    if not user_df.empty:
        phone_number = user_df['user_phone'].iloc[0] if 'user_phone' in user_df.columns else 'N/A'
        question_count = len(user_df)

        col1, col2 = st.columns(2)
        col1.metric("📱 Phone Number", phone_number)
        col2.metric("💬 Total Questions", question_count)

    # Create a dropdown for selecting a message
    question_list = user_df['question'].tolist()
    selected_question = st.selectbox("Select a message", question_list)

    # Get the response for that question
    selected_row = user_df[user_df['question'] == selected_question].iloc[0]
    response = selected_row['response']
    timestamp = selected_row['timestamp']

    # Show conversation
    st.markdown(f"🕒 **{timestamp}**")
    st.markdown(f"🧑‍💬 **User:** {selected_question}")
    st.markdown(f"🤖 **Bot:** {response}")


st.subheader("Clusters")

# Define categories with associated keywords
categories = {
    'Enquiry': ['how', 'what', 'can', 'please', 'when', 'why'],
    'Feedback': ['feedback', 'suggestion', 'review', 'rate', 'comment'],
    'Support Issue': ['help', 'issue', 'problem', 'trouble', 'support'],
    'General Conversation': ['hello', 'hi', 'thanks', 'ok', 'bye', 'good', 'great']
}

# Function to clean and preprocess text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Function to classify text based on predefined categories
def classify_text(text):
    for category, keywords in categories.items():
        if any(keyword in text for keyword in keywords):
            return category
    return 'General Conversation'  # Default category if no match found

# Preprocess and classify the data
df['processed_question'] = df['question'].apply(clean_text)
df['category'] = df['processed_question'].apply(classify_text)

# --- Display Cluster Buttons in the Middle ---
st.subheader("🔹 **Clusters**")
clicked_category = None

# Calculate counts for each category
category_counts = df['category'].value_counts().to_dict()

# Display buttons in columns (middle-aligned)
cols = st.columns(len(categories))
for idx, (category, keywords) in enumerate(categories.items()):
    count = category_counts.get(category, 0)
    with cols[idx]:
        if st.button(f"{category} ({count})"):
            clicked_category = category

# --- Plot the Category Distribution ---
st.subheader("📊 **Category Distribution**")
category_counts_df = pd.DataFrame(list(category_counts.items()), columns=['Category', 'Count'])
fig = px.bar(category_counts_df, x='Category', y='Count', title="Category Distribution")
st.plotly_chart(fig, use_container_width=True)

# --- Display Filtered Data ---
if clicked_category:
    st.subheader(f"📝 Showing Messages for Category: **{clicked_category}**")
    filtered_category_df = df[df['category'] == clicked_category]
    st.dataframe(filtered_category_df[['timestamp', 'user', 'question', 'response']])
else:
    st.info("")


# Call the enhanced clustered wordcloud function
clustered_text, cluster_keywords = generate_enhanced_clustered_wordcloud(filtered_df)

# Display the clusters with enhanced UI
st.markdown("## 🗂️ **Cluster Insights**")
for cluster_num, texts in clustered_text.items():
    with st.expander(f"📌 **Cluster {cluster_num} — Topics: {cluster_keywords[cluster_num]}**", expanded=False):
        st.markdown(f"**Top Messages:**")
        for msg in texts[:5]:  # Display the first 5 messages for each cluster
            st.write(f"- {msg}")




# Display data
st.subheader("📄 Filtered Feedback Data")
st.dataframe(filtered_df)
