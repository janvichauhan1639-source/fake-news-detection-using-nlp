import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

import string
nltk.download("stopwords")

def clean_text(text):
    text = str(text)

    text = text.lower()

    text = re.sub(r"http\S+|www\S+", "", text)

    text = re.sub(r"<.*?>", "", text)

    text = re.sub(r"\d+", "", text)

    text = text.translate(str.maketrans("", "", string.punctuation))

    text = re.sub(r"\s+", " ", text).strip()

    return text
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
def clean_text(text):

    text = str(text)

    text = text.lower()

    text = re.sub(r"http\S+|www\S+", "", text)

    text = re.sub(r"<.*?>", "", text)

    text = re.sub(r"\d+", "", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize
    words = text.split()

    # Remove Stopwords + Stemming
    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    text = " ".join(words)

    return text
