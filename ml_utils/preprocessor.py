"""Text preprocessing module for news clustering"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')


class TextPreprocessor:
    """Preprocess text for clustering"""

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # Add common words that don't add meaning
        self.stop_words.update([
            'said', 'would', 'could', 'also', 'may', 'might', 'must',
            'should', 'this', 'that', 'these', 'those', 'one', 'two',
            'like', 'even', 'see', 'get', 'make', 'go', 'know', 'take',
            'come', 'think', 'look', 'want', 'give', 'use', 'find',
            'tell', 'ask', 'work', 'seem', 'feel', 'try', 'leave',
            'call', 'new', 'year', 'time', 'day', 'people', 'way'
        ])

    def clean_text(self, text):
        """Basic cleaning"""
        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove numbers
        text = re.sub(r'\d+', '', text)

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def tokenize_and_lemmatize(self, text):
        """Tokenize and lemmatize text"""
        # Tokenize using simple regex to avoid NLTK hang issues that can occur
        # with word_tokenize() on certain input texts containing malformed Unicode or special characters
        tokens = re.findall(r'\b\w+\b', text.lower())

        # Remove stopwords and lemmatize
        lemmatized = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]

        return ' '.join(lemmatized)

    def preprocess(self, text):
        """Complete preprocessing pipeline"""
        # print("  Cleaning...")
        text = self.clean_text(text)
        # print("  Tokenizing...")
        text = self.tokenize_and_lemmatize(text)
        # print("  Done.")
        return text

    def preprocess_documents(self, documents):
        """Preprocess multiple documents"""
        print("Preprocessing documents...")
        processed = []
        for i, doc in enumerate(documents):
            print(f"  Processing document {i+1}/{len(documents)} (len={len(doc)})...")
            processed.append(self.preprocess(doc))
        print("Preprocessing complete!")
        return processed
