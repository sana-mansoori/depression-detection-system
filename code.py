import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Define sentences expressing different emotions
sentences = [
    "I'm feeling really sad and lonely today.",
    "Today was a great day, feeling happy!",
    "I can't stop crying, everything is terrible.",
    "Feeling optimistic about the future.",
    "I'm feeling overwhelmed and stressed.",
    "I just want to be left alone.",
    "Feeling grateful for the little things in life.",
    "I feel like a failure and can't do anything right.",
]

while len(sentences) < 1000:
    sentences.append("Random sentence.")

# Generate labels randomly (1 for depressed, 0 for not depressed)
labels = [random.choice([0, 1]) for _ in range(1000)]

# Create the dataset
data = {"text": sentences[:1000], "label": labels[:1000]}

# Convert to DataFrame
dataset = pd.DataFrame(data)
dataset.to_csv("depression_dataset.csv", index=False)
dataset = pd.read_csv("depression_dataset.csv")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    dataset["text"], dataset["label"], test_size=0.2, random_state=42
)


vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Logistic Regression classifier
lr_classifier = LogisticRegression()
lr_classifier.fit(X_train_tfidf, y_train)

# Train Random Forest classifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train_tfidf, y_train)

# Make predictions
lr_predictions = lr_classifier.predict(X_test_tfidf)
rf_predictions = rf_classifier.predict(X_test_tfidf)

# Calculate accuracy
lr_accuracy = accuracy_score(y_test, lr_predictions)
rf_accuracy = accuracy_score(y_test, rf_predictions)

# Plot the results
algorithms = ["Logistic Regression", "Random Forest"]
accuracy_scores = [lr_accuracy, rf_accuracy]

plt.bar(algorithms, accuracy_scores)
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Depression Detection Accuracy")
plt.show()

# Calculate percentage of depressed and non-depressed instances
percentage_depressed = dataset["label"].sum() / len(dataset) * 1000
percentage_non_depressed = 1000 - percentage_depressed

# Plot the percentage graph
plt.bar(["Depressed", "Non-Depressed"], [percentage_depressed, percentage_non_depressed])
plt.ylim(0, 1000)
plt.ylabel("Percentage")
plt.title("Percentage of Depressed vs. Non-Depressed Instances ")
plt.show()

