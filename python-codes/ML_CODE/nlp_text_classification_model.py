# Step 1: Train, Test, and Save the NLP Model

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn import metrics
import pickle


# Sample data (replace with your own dataset)
texts = [
    "I love programming in Python",
    "I hate doing the dishes",
    "Python is a great programming language",
    "I dislike bad weather",
    "I enjoy machine learning",
    "Bad weather makes me sad",
]
labels = [1, 0, 1, 0, 1, 0]  # 1 for positive sentiment, 0 for negative sentiment

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Create a pipeline with a TfidfVectorizer and a LogisticRegression model
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

# Train the model
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)
print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{metrics.classification_report(y_test, y_pred)}")

# Save the model to a file
with open('text_classification_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# OR SAVE The Model to a file.
# import joblib



#joblib.dump(model, 'text_classification_model.pkl')


# print('Model saved as text_classification_model.pkl')
