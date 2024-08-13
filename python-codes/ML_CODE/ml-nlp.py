# Libraries Installation

# pip install spacy gensim textblob
# (Semantics: Gensim's Word2Vec is used to generate word embeddings for semantic analysis.)
# (Pragmatics: TextBlob is used for basic sentiment analysis, which can help understand sentiments and nuances in text.)

# python -m spacy download en_core_web_sm
# (Syntax: The SpaCy model en_core_web_sm provides dependency parsing for syntax analysis.)


# python -m spacy download en_coref_md
# (Discourse: Coreference resolution is demonstrated with SpaCy's en_coref_md model.)


# 1. Syntax: Dependency Parsing with SpaCy

import spacy

# Load the SpaCy model
nlp = spacy.load('en_core_web_sm')

# Process the text
text = "The cat sat on the mat."
doc = nlp(text)

# Print the dependency parse
for token in doc:
    print(f'{token.text:10} {token.dep_:10} {token.head.text:10}')

# Output structure (word, dependency relation, head word)


# 2. Semantics: Word Embeddings with Gensim

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# Sample sentences
sentences = [
    "The cat sat on the mat.",
    "Dogs are great pets.",
    "I love programming in Python."
]

# Tokenize the sentences
tokenized_sentences = [simple_preprocess(sentence) for sentence in sentences]

# Train a Word2Vec model
model = Word2Vec(sentences=tokenized_sentences, vector_size=50, window=3, min_count=1, sg=1)

# Find similar words
similar_words = model.wv.most_similar('cat')
print(similar_words)


# 3. Pragmatics: Sentiment Analysis with TextBlob

from textblob import TextBlob

# Sample text
text = "I love this movie! It's fantastic."
blob = TextBlob(text)

# Get the sentiment
sentiment = blob.sentiment
print(f'Sentiment polarity: {sentiment.polarity}')
print(f'Sentiment subjectivity: {sentiment.subjectivity}')


# 4. Discourse: Coreference Resolution with SpaCy

import spacy
import neuralcoref

# Load the SpaCy model
nlp = spacy.load('en_core_web_md')

# Add neuralcoref to the SpaCy pipeline
neural_coref = neuralcoref.NeuralCoref(nlp.vocab)
nlp.add_pipe(neural_coref, name='neural_coref', last=True)

# Process the text
text = "John went to the store. He bought some apples."
doc = nlp(text)

# Print coreferences
if doc._.has_coref:
    for cluster in doc._.coref_clusters:
        print(f"Cluster: {[mention.text for mention in cluster.mentions]}")
else:
    print("No coreferences found.")


# Output the coreference resolution
