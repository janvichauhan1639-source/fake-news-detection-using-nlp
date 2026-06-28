import pandas as pd
import joblib

from preprocessing import clean_text

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("=" * 60)
print("        FAKE NEWS MODEL TRAINING")
print("=" * 60)



news = pd.read_csv("data/clean_news.csv")

print("\nDataset Shape :", news.shape)



news = news[["text", "label"]]



news.dropna(inplace=True)

news["text"] = news["text"].fillna("").astype(str)

news = news[news["text"].str.strip() != ""]



print("\nCleaning Text...")

news["text"] = news["text"].apply(clean_text)

print("Text Cleaning Completed")



X = news["text"]

y = news["label"]



vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X = vectorizer.fit_transform(X)

print("\nTF-IDF Shape :", X.shape)



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Shape :", X_train.shape)

print("Testing Shape :", X_test.shape)



print("\nTraining Logistic Regression...")

lr = LogisticRegression(max_iter=1000)

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

lr_acc = accuracy_score(y_test, lr_pred)

print("Accuracy :", round(lr_acc * 100, 2), "%")


print("\nTraining Naive Bayes...")

nb = MultinomialNB()

nb.fit(X_train, y_train)

nb_pred = nb.predict(X_test)

nb_acc = accuracy_score(y_test, nb_pred)

print("Accuracy :", round(nb_acc * 100, 2), "%")


print("\nTraining Linear SVM...")

svm = LinearSVC()

svm.fit(X_train, y_train)

svm_pred = svm.predict(X_test)

svm_acc = accuracy_score(y_test, svm_pred)

print("Accuracy :", round(svm_acc * 100, 2), "%")



print("\nTraining Random Forest...")

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)

print("Accuracy :", round(rf_acc * 100, 2), "%")



print("\nClassification Report (Best Baseline - Logistic Regression)\n")

print(classification_report(y_test, lr_pred))



print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(f"Logistic Regression : {lr_acc:.4f}")
print(f"Naive Bayes         : {nb_acc:.4f}")
print(f"Linear SVM          : {svm_acc:.4f}")
print(f"Random Forest       : {rf_acc:.4f}")



models = {
    "Logistic Regression": (lr, lr_acc),
    "Naive Bayes": (nb, nb_acc),
    "Linear SVM": (svm, svm_acc),
    "Random Forest": (rf, rf_acc)
}

best_name = max(models, key=lambda x: models[x][1])

best_model = models[best_name][0]

best_accuracy = models[best_name][1]



joblib.dump(best_model, "models/model.pkl")

joblib.dump(vectorizer, "models/vectorizer.pkl")

print("\n")
print("=" * 60)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 60)

print("Best Model :", best_name)

print("Accuracy :", round(best_accuracy * 100, 2), "%")