import pandas as pd

def parse_uploaded_excel(file):
    df = pd.read_excel(file)
    df.rename(columns={
        "TimeStamp": "timestamp",
        "User Phone": "user_phone",
        "User Name": "user",
        "Question": "question",
        "AI Response": "response"
    }, inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df
