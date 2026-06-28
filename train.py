import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("=" * 60)
print("           MODEL TRAINING")
print("=" * 60)

# ==========================================================
# Load Dataset
# ==========================================================

news = pd.read_csv("data/clean_news.csv")

print("\nDataset Shape :", news.shape)

# ==========================================================
# Keep Required Columns
# ==========================================================

news = news[["text", "label"]]

# ==========================================================
# Remove Missing Values
# ==========================================================

news.dropna(inplace=True)

news["text"] = news["text"].fillna("").astype(str)

news = news[news["text"].str.strip() != ""]

print("\nDataset Shape After Cleaning :", news.shape)

# ==========================================================
# Features and Labels
# ==========================================================

X = news["text"]
y = news["label"]

# ==========================================================
# TF-IDF
# ==========================================================

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X = vectorizer.fit_transform(X)

print("\nTF-IDF Shape :", X.shape)

# ==========================================================
# Train Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Shape :", X_train.shape)
print("Testing Shape :", X_test.shape)

# ==========================================================
# Logistic Regression
# ==========================================================

print("\nTraining Logistic Regression...")

lr = LogisticRegression(max_iter=1000)

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

lr_acc = accuracy_score(y_test, lr_pred)

print("Accuracy :", round(lr_acc * 100, 2), "%")

print(classification_report(y_test, lr_pred))

# ==========================================================
# Naive Bayes
# ==========================================================

print("\nTraining Naive Bayes...")

nb = MultinomialNB()

nb.fit(X_train, y_train)

nb_pred = nb.predict(X_test)

nb_acc = accuracy_score(y_test, nb_pred)

print("Accuracy :", round(nb_acc * 100, 2), "%")

# ==========================================================
# Linear SVM
# ==========================================================

print("\nTraining Linear SVM...")

svm = LinearSVC()

svm.fit(X_train, y_train)

svm_pred = svm.predict(X_test)

svm_acc = accuracy_score(y_test, svm_pred)

print("Accuracy :", round(svm_acc * 100, 2), "%")

# ==========================================================
# Random Forest
# ==========================================================

print("\nTraining Random Forest...")

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)

print("Accuracy :", round(rf_acc * 100, 2), "%")

# ==========================================================
# Model Comparison
# ==========================================================

print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(f"Logistic Regression : {lr_acc:.4f}")
print(f"Naive Bayes         : {nb_acc:.4f}")
print(f"Linear SVM          : {svm_acc:.4f}")
print(f"Random Forest       : {rf_acc:.4f}")

# ==========================================================
# Save Best Model
# ==========================================================

best_model = svm
best_name = "Linear SVM"

joblib.dump(best_model, "models/model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("\nBest Model :", best_name)
print("Model Saved Successfully")

