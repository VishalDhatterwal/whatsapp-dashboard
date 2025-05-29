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
from src.parser import parse_uploaded_excel
from keybert import KeyBERT
from collections import Counter
import re
import emoji
from src.emoji_utils import get_emoji_stats
import plotly.express as px
from streamlit_plotly_events import plotly_events
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from src.utils import show_brand_drilldown, show_interaction_summary

st.set_page_config(page_title="AI Coach Dashboard", layout="wide")
st.title("🤖 AI Expert Coach Feedback Dashboard")

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = parse_uploaded_excel(uploaded_file)
    else:
        df = parse_uploaded_excel("data/feedback1.xlsx")
    return df

uploaded_file = st.file_uploader("Upload Excel File or XLSX File", type="xlsx")
df = load_data(uploaded_file)

# ✅ Process DataFrame in all cases (uploaded or default)
@st.cache_data
def preprocess_data(df):
    df = clean_chat(df)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df = df.dropna(subset=['timestamp'])

    # Remove meaningless greetings/fillers
    useless_keywords = {"hi", "Hii","hlo", "hello", "hey", "ok", "okay", "yo", "thanks", "thank you", "good morning", "good night", "bye", "yes", "no", "hmm", "nice"}

    def is_useful(msg):
        if not isinstance(msg, str) or not msg.strip():
            return False
        msg_clean = msg.lower().strip()
        return not all(word in useless_keywords for word in msg_clean.split())

    df = df[df['question'].apply(is_useful)]
    df = add_sentiment(df, text_column='question')
    df = add_emotions(df, text_column='question')
    df = classify_message_type(df)
    df = extract_response_emojis(df)
    return df
df = preprocess_data(df)

st.success("✅ File processed successfully!")


# --- Sidebar Filters (Optimized with Caching) ---
with st.sidebar:
    st.header("📌 Filters")
    min_date, max_date = df['timestamp'].min().date(), df['timestamp'].max().date()
    types = df['type'].dropna().unique().tolist()
    users = sorted(df['user'].dropna().unique())

    selected_types = st.multiselect("Message Type", types, default=types)
    start_date, end_date = st.date_input("Date Range", [min_date, max_date])
    search_user = st.text_input("🔍 Search User")
    filtered_users = ['All'] + [u for u in users if search_user.lower() in u.lower()]
    selected_user = st.selectbox("User", filtered_users)

@st.cache_data
def filter_dataframe(df, start_date, end_date, selected_user, selected_types):
    df_filtered = df[(df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)]
    if selected_user != 'All':
        df_filtered = df_filtered[df_filtered['user'] == selected_user]
    df_filtered = df_filtered[df_filtered['type'].isin(selected_types)]
    return df_filtered

filtered_df = filter_dataframe(df, start_date, end_date, selected_user, selected_types)


# --- Brand Setup ---
brand_themes = {
    'Centrum': ['centrum'],
    'Paradontax': ['paradontax'],
    'Sensodyne': ['sensodyne'],
    'Otrivin': ['otrivin'],
    'Eno': ['eno'],
    'Crocin': ['crocin']
}

subthemes = [
    "Pricing Enquiry",
    "Product Information",
    "Composition/Ingredients",
    "Usage/Benefits, Dosage",
    "Safety/Side Effects"
]

def clean_text(text):
    import re
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def classify_brand_theme(text):
    for brand, keywords in brand_themes.items():
        if any(keyword in text for keyword in keywords):
            return brand
    return 'Other'

def assign_subtheme(question):
    text = question.lower()
    if any(kw in text for kw in ["price", "cost", "affordable"]):
        return "Pricing Enquiry"
    elif any(kw in text for kw in ["where", "stock", "available", "buy"]):
        return "Product Information"
    elif any(kw in text for kw in ["ingredients", "composition", "formulation"]):
        return "Composition/Ingredients"
    elif any(kw in text for kw in ["use", "benefit", "dosage", "how much"]):
        return "Usage/Benefits, Dosage"
    elif any(kw in text for kw in ["side effect", "safe", "safety", "reaction"]):
        return "Safety/Side Effects"
    else:
        return "Other"

# --- Apply brand logic ---
df['processed_question'] = df['question'].apply(clean_text)
df['brand_theme'] = df['processed_question'].apply(classify_brand_theme)
df['subtheme'] = df['question'].apply(assign_subtheme)

# --- Streamlit UI ---
st.subheader("🔹 Brand Themes")

if 'selected_brand' not in st.session_state:
    st.session_state.selected_brand = None
if 'selected_subtheme' not in st.session_state:
    st.session_state.selected_subtheme = None

brand_counts = df['brand_theme'].value_counts().to_dict()
cols = st.columns(len(brand_themes))

# --- Brand Buttons with Highlight ---
for idx, brand in enumerate(brand_themes.keys()):
    count = brand_counts.get(brand, 0)
    label = f"{brand} ({count})"
    with cols[idx]:
        if st.session_state.selected_brand == brand:
            st.markdown(f"<button style='background-color:#4CAF50;color:white;border:none;padding:0.5em 1em;border-radius:5px'>{label}</button>", unsafe_allow_html=True)
        else:
            if st.button(label, key=f"top_brand_btn_{idx}_{brand}"):
                st.session_state.selected_brand = brand
                st.session_state.selected_subtheme = None

# --- Sub-Themes Section ---
if st.session_state.selected_brand:
    st.markdown(f"### Sub-Themes for {st.session_state.selected_brand}")

    sub_cols = st.columns(2)
    for idx, sub in enumerate(subthemes):
        with sub_cols[idx % 2]:
            if st.session_state.selected_subtheme == sub:
                st.markdown(f"<button style='background-color:#FF6347;color:white;border:none;padding:0.5em 1em;border-radius:5px'>{sub}</button>", unsafe_allow_html=True)
            else:
                if st.button(sub, key=f"subtheme_btn_{idx}"):
                    st.session_state.selected_subtheme = sub

# --- Filtered Output Section ---
if st.session_state.selected_brand and st.session_state.selected_subtheme:
    st.markdown(f"### 🔍 Questions for **{st.session_state.selected_brand}** - **{st.session_state.selected_subtheme}**")
    filtered_df = df[
        (df['brand_theme'] == st.session_state.selected_brand) &
        (df['subtheme'] == st.session_state.selected_subtheme)
    ]

    if not filtered_df.empty:
        st.dataframe(filtered_df[['question', 'response']])
    else:
        st.info("No matching data found.")



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

# --- Topic Extraction with KeyBERT ---
st.subheader("🎯 Top Topics and Keywords")

# Combine all user questions into a single string
messages = " ".join(filtered_df['question'].dropna().tolist())

# Initialize KeyBERT model
kw_model = KeyBERT()

# Extract top 10 keywords
keywords = kw_model.extract_keywords(messages, top_n=10)

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


# === Top Users by Question Volume ===
st.subheader("👤 Top Users (By Question Count)")
top_users = filtered_df['user'].value_counts().head(10)
st.bar_chart(top_users)

# === Daily Activity ===
st.subheader("📈 Daily Question Volume")
daily_activity = filtered_df.groupby(filtered_df['timestamp'].dt.date).size()
st.line_chart(daily_activity)


# === Sentiment Distribution ===
st.subheader("😊 Sentiment Distribution")
if 'sentiment' in filtered_df.columns:
    sentiment_counts = filtered_df['sentiment'].value_counts()
    st.bar_chart(sentiment_counts)
else:
    st.info("No sentiment data available.")


# Sentiment analysis
st.subheader("📈 Sentiment Trends")
daily_sentiment = filtered_df.groupby([filtered_df['timestamp'].dt.date, 'sentiment']).size().unstack().fillna(0)
st.line_chart(daily_sentiment)


# Display data
st.subheader("📄 Filtered Feedback Data")
st.dataframe(filtered_df)


# Define themes with associated keywords
themes = {
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

# Function to classify text based on predefined themes
def classify_text(text):
    for theme, keywords in themes.items():
        if any(keyword in text for keyword in keywords):
            return theme
    return 'General Conversation'  # Default theme if no match found

# Preprocess and classify the data
df['processed_question'] = df['question'].apply(clean_text)
df['theme'] = df['processed_question'].apply(classify_text)

# --- Display Theme Buttons in the Middle ---
st.subheader("🔹 **Themes**")
clicked_theme = None

# Calculate counts for each theme
theme_counts = df['theme'].value_counts().to_dict()

# Display buttons in columns (middle-aligned)
cols = st.columns(len(themes))
for idx, (theme, keywords) in enumerate(themes.items()):
    count = theme_counts.get(theme, 0)
    with cols[idx]:
        if st.button(f"{theme} ({count})"):
            clicked_theme = theme

# --- Plot the Theme Distribution ---
st.subheader("📊 **Theme Distribution**")
theme_counts_df = pd.DataFrame(list(theme_counts.items()), columns=['Theme', 'Count'])
fig = px.bar(theme_counts_df, x='Theme', y='Count', title="Theme Distribution")
st.plotly_chart(fig, use_container_width=True)


#clickable drill down 
st.subheader("🔍 Overall Visualization Clickable Drill-Down by User")

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