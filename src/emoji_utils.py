import re
from collections import Counter

def get_emoji_stats(series):
    emoji_pattern = re.compile(
        r"["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002700-\U000027BF"  # dingbats
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\u2600-\u26FF"          # misc
        "]+", flags=re.UNICODE
    )

    emoji_counter = Counter()
    for text in series.dropna():
        found = emoji_pattern.findall(str(text))
        emoji_counter.update(found)

    return dict(emoji_counter)