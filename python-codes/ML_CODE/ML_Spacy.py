import spacy

# Load a pre-trained SpaCy model
nlp = spacy.load('en_core_web_sm')

text = "The quick brown fox jumps over the lazy dog."

# Process the text
doc = nlp(text)

# Tokenization and Stop Words Removal
tokens = [token.text for token in doc]
filtered_tokens = [token.text for token in doc if not token.is_stop]

# Lemmatization
lemmatized_tokens = [token.lemma_ for token in doc]

# Named Entity Recognition
entities = [(ent.text, ent.label_) for ent in doc.ents]

print("Original Text:", text)
print("Tokens:", tokens)
print("Filtered Tokens:", filtered_tokens)
print("Lemmatized Tokens:", lemmatized_tokens)
print("Entities:", entities)
