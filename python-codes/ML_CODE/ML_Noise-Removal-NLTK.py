import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import download

# Download necessary NLTK data
download('punkt')
download('stopwords')

# Sample text
text = "The quick brown fox jumps over the lazy dog. The dog barked loudly at the fox! #fox #dog"

# 1. Remove Punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

# 2. Remove Stopwords
def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word.lower() not in stop_words]

# 3. Tokenization
tokens = word_tokenize(remove_punctuation(text))

# 4. Remove Stopwords
filtered_tokens = remove_stopwords(tokens)

print("Original Text:", text)
print("Tokens:", tokens)
print("Filtered Tokens (No Stopwords):", filtered_tokens)
