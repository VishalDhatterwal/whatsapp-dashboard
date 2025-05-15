import pandas as pd
import re
import string

# === Cleaning Utility ===
def clean_text_keep_emojis(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # Normalize spaces
    return text

# === Emoji-safe cleaner for question and response columns ===
def clean_questions_responses_keep_emojis(df):
    if 'question' in df.columns:
        df['question'] = df['question'].apply(clean_text_keep_emojis)
    if 'response' in df.columns:
        df['response'] = df['response'].apply(clean_text_keep_emojis)
    return df

# === Emoji-safe cleaner for full chat context ===
def clean_chat_keep_emojis(df):
    def basic_clean_emojis(text):
        if pd.isna(text):
            return ""
        text = str(text)
        text = re.sub(r"http\S+", "", text)  # Remove URLs
        text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
        text = re.sub(r"\s+", " ", text).strip()  # Normalize spaces
        return text

    if 'question' in df.columns:
        df['question'] = df['question'].apply(basic_clean_emojis)
    if 'response' in df.columns:
        df['response'] = df['response'].apply(basic_clean_emojis)
    return df
