import pandas as pd
from src.parser import parse_uploaded_excel
from src.cleaner import clean_questions_responses
from src.analyzer import add_sentiment, add_emotions
from src.compare import compare_questions_responses
from src.db import store_to_db

# === CONFIG ===
EXCEL_FILE = "data/AI Expert Coach Feedback.xlsx"
DB_CONFIG = {
    'user': 'root',
    'password': 'your_password',
    'host': 'localhost',
    'port': 3306,
    'db_name': 'ai_feedback',
    'table_name': 'chat_feedback'
}

# === PIPELINE ===
if __name__ == "__main__":
    print("📥 Loading and parsing Excel file...")
    df = parse_uploaded_excel(EXCEL_FILE)

    print("🧹 Cleaning text fields...")
    df = clean_questions_responses(df)

    print("🔍 Analyzing sentiment and emotion...")
    df = add_sentiment(df, text_column='question')
    df = add_emotions(df, text_column='question')

    print("📊 Comparing question-response pairs...")
    df = compare_questions_responses(df)

    print("💾 Saving to MySQL database...")
    store_to_db(df, **DB_CONFIG)

    print("✅ Pipeline completed successfully.")
