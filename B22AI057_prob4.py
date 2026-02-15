# Sport vs Politics Text Classifier
# Uses 3 ML models:
# 1. Multinomial Naive Bayes
# 2. Logistic Regression
# 3. Support Vector Machine (Linear SVM)

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report


# --------------------------------------------------
# 1. DATA COLLECTION
# --------------------------------------------------
# Sports category → rec.sport.*
# Politics category → talk.politics.*

categories = [
    "rec.sport.baseball",
    "rec.sport.hockey",
    "talk.politics.guns",
    "talk.politics.mideast"
]

dataset = fetch_20newsgroups(
    subset="all",
    categories=categories,
    remove=("headers", "footers", "quotes")
)

X = dataset.data
y = dataset.target

# Convert labels:
# sport → 0, politics → 1
y_binary = []
for label in y:
    if label in [0, 1]:
        y_binary.append(0)  # Sport
    else:
        y_binary.append(1)  # Politics

y_binary = np.array(y_binary)


# --------------------------------------------------
# 2. TRAIN–TEST SPLIT
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.25, random_state=42
)


# --------------------------------------------------
# 3. FEATURE EXTRACTION (TF-IDF with n-grams)
# --------------------------------------------------
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=10000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# --------------------------------------------------
# 4. MODEL DEFINITIONS
# --------------------------------------------------
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC()
}


# --------------------------------------------------
# 5. TRAINING & EVALUATION
# --------------------------------------------------
for name, model in models.items():
    print("\n" + "=" * 50)
    print(f"Model: {name}")
    print("=" * 50)

    model.fit(X_train_tfidf, y_train)
    predictions = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Sport", "Politics"]
    ))


# --------------------------------------------------
# 6. CUSTOM TEXT TEST
# --------------------------------------------------
def classify_text(text):
    text_vec = vectorizer.transform([text])
    prediction = models["Linear SVM"].predict(text_vec)[0]
    return "Politics" if prediction == 1 else "Sport"


print("\nSample Predictions:")
print(classify_text("The government passed a new law on gun control"))
print(classify_text("The team won the hockey championship last night"))
