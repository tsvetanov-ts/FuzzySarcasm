import re

def clean_text(text, remove_hashtags=False):
    """
    Basic text cleaning:
      - Converts text to lowercase,
      - Removes URLs, mentions, hashtags (optional),
      - Removes non-alphabet characters.
    If the input is not a string, it is converted to one.
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove mentions and hashtags
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    # Remove punctuation and numbers (keep only letters and spaces)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_texts(texts, remove_hashtags=False):
    """
    Apply cleaning to a list of texts.
    """
    return [clean_text(text,remove_hashtags) for text in texts]