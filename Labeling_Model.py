# Import libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the data
df = pd.read_csv("incidents.csv")

# Split the data into the input and output variables
X = df['description']
y = df['label']

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Transform the input variable into a TF-IDF matrix
X = vectorizer.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = clf.predict(X_test)

# Check the Models Accurracy and print it to the user
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Load the new data
new_data = pd.read_csv("new_incidents.csv")

# Transform the new data
new_data = vectorizer.transform(new_data['description'])

# Predict the labels for the new data
new_data_pred = clf.predict(new_data)
