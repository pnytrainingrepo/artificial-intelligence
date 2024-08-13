# Step 1: Install Necessary Libraries
# pip install streamlit pandas scikit-learn matplotlib seaborn

# Step 2: Import Required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
import streamlit as st

# Step 3: Load and Prepare the Spam Dataset
df = pd.read_csv('./datasets/spam.csv')

# Preprocess the data (assuming the columns are 'Category' and 'Message')
df['label'] = df['Category'].map({'spam': 1, 'ham': 0})  # Convert labels to binary (spam=1, ham=0)

# Features and target
X = df['Message']
y = df['label']

# Convert the text data into a bag of words model
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Step 4: Streamlit App for Logistic Regression Model

# Streamlit app
st.title("Spam Detection with Logistic Regression")

# Sidebar for logistic regression parameters
st.sidebar.header("Model Parameters")
C = st.sidebar.slider("Regularization strength (C)", 0.01, 10.0, 1.0)
max_iter = st.sidebar.slider("Maximum Iterations", 100, 10000, 1000)

# Train the logistic regression model
model = LogisticRegression(C=C, max_iter=max_iter, solver='liblinear')
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluate the model
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, output_dict=True)
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

# Plot confusion matrix
st.subheader("Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_title('Confusion Matrix')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
st.pyplot(fig)

# Plot ROC curve
st.subheader("ROC Curve")
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic (ROC) Curve')
ax.legend(loc="lower right")
st.pyplot(fig)

# Display classification report
st.subheader("Classification Report")
st.text(pd.DataFrame(class_report).transpose().to_string())

# Step 5: Run the Streamlit App

# streamlit run app.py