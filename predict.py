import joblib
from preprocessing import clean_text

print("="*50)
print("FAKE NEWS DETECTOR")
print("="*50)

model = joblib.load("models/model.pkl")

vectorizer = joblib.load("models/vectorizer.pkl")

news = input("\nEnter News : ")

news = clean_text(news)

news_vector = vectorizer.transform([news])

prediction = model.predict(news_vector)

if prediction[0] == 0:
    print("\nPrediction : Fake News")
else:
    print("\nPrediction : Real News")