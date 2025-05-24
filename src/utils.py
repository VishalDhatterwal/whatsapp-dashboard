import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import re
from wordcloud import WordCloud
from keybert import KeyBERT


def show_brand_drilldown(df, brand_name):
    filtered_df = df[df['brand_theme'] == brand_name]

    st.markdown(f"### 📌 Messages mentioning **{brand_name}**")
    st.metric("Total Messages", len(filtered_df))

    # Convert timestamp to datetime and extract date
    filtered_df['timestamp'] = pd.to_datetime(filtered_df['timestamp'], errors='coerce')
    filtered_df['DateOnly'] = filtered_df['timestamp'].dt.date

    # Active Days
    st.metric("Active Days", filtered_df['DateOnly'].nunique())

    # Show Table
    st.dataframe(filtered_df[['question', 'timestamp','response']])

    # ✅ USER INTERACTION SUMMARY
    st.markdown("### 👥 User Interaction Summary")
    msg_by_day = filtered_df.groupby('DateOnly').size().reset_index(name='Message Count')

    if not msg_by_day.empty:
        peak_day = msg_by_day.loc[msg_by_day['Message Count'].idxmax()]
        avg_msgs = msg_by_day['Message Count'].mean()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📅 Peak Day", str(peak_day['DateOnly']))
        with col2:
            st.metric("📈 Peak Msgs", peak_day['Message Count'])
        with col3:
            st.metric("📊 Avg Daily Msgs", f"{avg_msgs:.2f}")

        fig_interact = px.line(msg_by_day, x='DateOnly', y='Message Count',
                               title='📈 Message Volume Over Time', markers=True)
        st.plotly_chart(fig_interact, use_container_width=True)
    else:
        st.info("No messages found for interaction summary.")

    # ✅ WORDCLOUD (KeyBERT)
    st.markdown("### 🔍 Keyword WordCloud (KeyBERT)")
    try:
        kw_model = KeyBERT()
        all_text = ' '.join(filtered_df['question'].dropna().tolist())
        keywords = kw_model.extract_keywords(all_text, stop_words='english', top_n=50)
        keyword_dict = {kw[0]: kw[1] for kw in keywords if kw[1] > 0.3}
        if keyword_dict:
            wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(keyword_dict)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.info("No strong keywords extracted.")
    except Exception as e:
        st.warning(f"Keyword extraction failed: {e}")


def show_interaction_summary(df):
    st.markdown("## 📈 User Interaction Summary")

    # Total Questions
    total_questions = df['question'].dropna().shape[0]

    # Emoji Count
    emoji_pattern = re.compile("[\U0001F600-\U0001F64F\U0001F300-\U0001F6FF\U0001F1E0-\U0001F1FF]+", flags=re.UNICODE)
    emoji_count = df['question'].dropna().apply(lambda x: len(emoji_pattern.findall(x))).sum()

    # Link Count
    link_pattern = re.compile(r"http[s]?://\S+")
    link_count = df['question'].dropna().apply(lambda x: len(link_pattern.findall(x))).sum()

    # Unique Users
    user_count = df['user'].nunique()

    # Avg Questions per User
    avg_questions = total_questions / user_count if user_count > 0 else 0

    # 🟨 Peak Day (Most Active Date)
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    peak_day = df['date'].value_counts().idxmax() if not df.empty else "N/A"

    # Display in Columns
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric(label="❓ Questions", value=total_questions)
    with col2:
        st.metric(label="😄 Emojis", value=emoji_count)
    with col3:
        st.metric(label="🔗 Links", value=link_count)
    with col4:
        st.metric(label="👥 Users", value=user_count)
    with col5:
        st.metric(label="📊 Avg Questions/User", value=f"{avg_questions:.1f}")
    with col6:
        st.metric(label="📅 Peak Day", value=str(peak_day))