import spacy

# Load a pre-trained SpaCy model
nlp = spacy.load('en_core_web_sm')

# Sample text
text = "The quick brown fox jumps over the lazy dog. The dog barked loudly at the fox! #fox #dog"

# Process the text
doc = nlp(text)

# 1. Remove Punctuation and Stopwords
filtered_tokens = [token.text for token in doc if not token.is_punct and not token.is_stop]

print("Original Text:", text)
print("Filtered Tokens (No Punctuation and Stopwords):", filtered_tokens)
