from transformers import pipeline
import pandas as pd
import re
import torch

# Sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

# Emotion classification pipeline (can use a specific model)
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=False)

def add_sentiment(df, text_column='message'):
    if text_column not in df.columns:
        raise ValueError(f"'{text_column}' column not found in DataFrame")
    
    sentiments = sentiment_analyzer(df[text_column].fillna('').tolist())
    df['sentiment'] = [result['label'] for result in sentiments]
    return df

# Optional: cache model to prevent reloading
# Ensure CPU is being used correctly
if torch.cuda.is_available():
    torch.device('cuda')
else:
    torch.device('cpu')

def get_emotion_model():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", device=-1)

def add_emotions(df, text_column='question'):
    emotion_pipeline = get_emotion_model()

    emotions = []
    for text in df[text_column]:
        if pd.isna(text) or not isinstance(text, str):
            emotions.append(None)
        else:
            try:
                result = emotion_pipeline(text)
                if isinstance(result, list) and len(result) > 0:
                    emotions.append(result[0]['label'])
                else:
                    emotions.append(None)
            except:
                emotions.append(None)

    df['emotion'] = emotions
    return df

def classify_message_type(df):
    def classify(text):
        if pd.isna(text): return "unknown"
        text = text.lower()
        if "?" in text or text.strip().endswith("?"):
            return "question"
        elif len(text.split()) < 4:
            return "short"
        return "statement"

    df["type"] = df["question"].apply(classify)
    return df

# === Define classify_message_type ===
def classify_message_type(df):
    import re

    def classify(msg):
        msg = str(msg).lower()

        if re.search(r"\b(book|appointment|schedule|availability|consultation|see the dentist)\b", msg):
            return 'appointment'
        elif re.search(r"\b(price|cost|charges|how much|fee|rate|affordable|package)\b", msg):
            return 'pricing'
        elif re.search(r"\b(braces|root canal|implant|filling|extraction|crown|bridge|dentures|whitening|scaling|treatment|procedure)\b", msg):
            return 'treatment inquiry'
        elif re.search(r"\b(pain|swelling|bleeding|sensitivity|emergency|urgent|infection|toothache)\b", msg):
            return 'emergency/issue'
        elif re.search(r"\b(report|results|checkup|follow up|again|revisit|next visit)\b", msg):
            return 'follow-up/checkup'
        elif re.search(r"\b(location|timing|working hours|open|clinic address|map|directions)\b", msg):
            return 'logistics/info'
        elif re.search(r"\b(insur|cashless|claim|policy|coverage)\b", msg):
            return 'insurance query'
        elif re.search(r"\b(cancel|reschedule|change|modify|missed|late)\b", msg):
            return 'rescheduling'
        else:
            return 'other'

    df['type'] = df['question'].apply(classify)
    return df


def extract_response_emojis(df):
    # Extract emojis from AI responses and store them
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        "]+", flags=re.UNICODE)

    def get_emojis(text):
        return ''.join(emoji_pattern.findall(str(text)))

    df['response_emojis'] = df['response'].apply(get_emojis)
    return df