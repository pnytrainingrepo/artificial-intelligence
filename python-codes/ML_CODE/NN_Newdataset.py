# Import necessary libraries
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load the 20 newsgroups dataset
newsgroups = fetch_20newsgroups(subset='all', categories=['rec.sport.hockey', 'sci.space'], remove=('headers', 'footers', 'quotes'))

# Prepare the features (text data) and labels
X = newsgroups.data
y = newsgroups.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert text data into TF-IDF features
vectorizer = TfidfVectorizer(max_features=2000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Create a Neural Network model using MLPClassifier
nn_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=20, random_state=42)

# Train the model
nn_model.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = nn_model.predict(X_test_tfidf)

# Evaluate the model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
