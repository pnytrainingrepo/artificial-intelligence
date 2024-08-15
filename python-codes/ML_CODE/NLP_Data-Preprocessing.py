# pip install nltk scikit-learn

import nltk
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import download

# Download necessary NLTK data
download('punkt')
download('stopwords')
download('wordnet')

# Sample text
text = """
The quick brown fox jumps over the lazy dog. The dog barked loudly at the fox ! word# to $.
"""

# 1. Text Normalization
def normalize_text(text):
    text = text.lower()  # Lowercasing
    text = text.translate(str.maketrans('', '', string.punctuation))  # Removing punctuation
    return text

# 2. Tokenization
def tokenize_text(text):
    words = word_tokenize(text)  # Word Tokenization
    return words

# 3. Stop Words Removal
def remove_stop_words(words):
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    return filtered_words

# 4. Stemming
def stem_words(words):
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in words]
    return stemmed_words

# 5. Lemmatization
def lemmatize_words(words):
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return lemmatized_words

# 6. Vectorization
def vectorize_text(text):
    vectorizer = TfidfVectorizer()  # Using TF-IDF Vectorizer
    vectors = vectorizer.fit_transform([text])
    return vectors, vectorizer.get_feature_names_out()

# Pre-processing pipeline
text = normalize_text(text)
words = tokenize_text(text)
words = remove_stop_words(words)
words = stem_words(words)  # You can use lemmatize_words(words) instead of stem_words(words) if preferred

# Vectorization
vectors, feature_names = vectorize_text(' '.join(words))

print("Normalized Text:", text)
print("Tokenized Words:", words)
print("TF-IDF Matrix:\n", vectors.toarray())
print("Feature Names:", feature_names)
