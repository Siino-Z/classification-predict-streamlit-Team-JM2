

# Function to preprocess text (remove emojis, links, mentions, hashtags, expand contractions, and normalize case)
def preprocess_text(text):
    # Change the case of all the words in the text to lowercase
    text = text.lower()

    # Remove links from the text
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    text = url_pattern.sub(r'', text)

    # Remove emojis
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F700-\U0001F77F"  # alchemical symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                               u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                               u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                               u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                               u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                               u"\U00002702-\U000027B0"  # Dingbat symbols
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    # Remove punctuations and weird characters
    text = re.sub(r'[^\w\s]', '', re.sub(r'[^a-zA-Z0-9\s]', '', text))

    return text

# Function to tokenize text
def tokenize_data(text):
    # Use the word_tokenize function for tokenization
    tokens = word_tokenize(text)
    return tokens

# Function to lemmatize text
def lemmatize_data(list_of_words):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(x) for x in list_of_words]
