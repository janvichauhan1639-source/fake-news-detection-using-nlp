# 📰 Fake News Detection System using NLP & Machine Learning

An end-to-end **Fake News Detection System** developed using **Python, Natural Language Processing (NLP), TF-IDF Vectorization, and Machine Learning**. The system classifies news articles as **Fake** or **Real** using multiple supervised machine learning algorithms and provides real-time prediction through a **Streamlit Web Application**.

---

# 📌 Project Overview

This project demonstrates a complete Machine Learning pipeline from raw data to deployment.

### Workflow

* Dataset Loading
* Exploratory Data Analysis (EDA)
* Data Cleaning
* NLP Text Preprocessing
* TF-IDF Feature Engineering
* Model Training
* Model Comparison
* Model Evaluation
* Model Serialization
* Real-time Prediction
* Streamlit Web Application

---

# 🚀 Features

* ✅ Fake & Real News Classification
* ✅ Data Cleaning and Text Preprocessing
* ✅ Stopword Removal & Stemming
* ✅ TF-IDF Feature Extraction
* ✅ Multiple Machine Learning Models
* ✅ Automatic Best Model Selection
* ✅ Model Comparison
* ✅ Saved Trained Model (.pkl)
* ✅ Interactive Streamlit Web App
* ✅ Modular Python Project Structure

---

# 🤖 Machine Learning Models

The following machine learning algorithms were implemented and compared:

| Model                         | Accuracy   |
| ----------------------------- | ---------- |
| Logistic Regression           | **98.66%** |
| Multinomial Naive Bayes       | **93.26%** |
| Linear Support Vector Machine | **99.41%** |
| Random Forest Classifier ⭐    | **99.74%** |

**Best Model Selected:** Random Forest Classifier

---

# 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Natural Language Processing (NLP)
* NLTK
* TF-IDF Vectorization
* Joblib
* Streamlit
* Git
* GitHub

---

# 📂 Project Structure

```text
Fake-News-Detection/
│
├── data/
│   ├── Fake.csv
│   ├── True.csv
│   └── clean_news.csv
│
├── models/
│   ├── model.pkl
│   └── vectorizer.pkl
│
├── reports/
│
├── app.py
├── fake_news.py
├── preprocessing.py
├── train.py
├── predict.py
├── requirements.txt
├── runtime.txt
└── README.md
```

---

# ⚙ Installation

Clone the repository

```bash
git clone https://github.com/janvichauhan1639-source/fake-news-detection-using-nlp.git
```

Move into project directory

```bash
cd fake-news-detection-using-nlp
```

Install required packages

```bash
pip install -r requirements.txt
```

---

# ▶ Run the Project

Train the model

```bash
python train.py
```

Predict news from terminal

```bash
python predict.py
```

Run Streamlit App

```bash
streamlit run app.py
```

---

# 📊 Model Evaluation

Evaluation metrics used:

* Accuracy Score
* Precision
* Recall
* F1 Score
* Classification Report
* Model Comparison

---

# 🌐 Streamlit Web App

The application allows users to:

* Paste a news article
* Automatically preprocess text
* Predict whether the news is Fake or Real
* Display cleaned text
* Provide instant prediction

---

# 🔮 Future Improvements

* BERT-based Fake News Detection
* LSTM / BiLSTM Models
* Explainable AI (SHAP / LIME)
* News API Integration
* Docker Deployment
* CI/CD Pipeline
* Cloud Deployment

---

# 👩‍💻 Author

**Janvi Chauhan**

GitHub:
https://github.com/janvichauhan1639-source

LinkedIn:
(Add your LinkedIn Profile URL)

---

# ⭐ Support

If you found this project useful, please give this repository a ⭐ on GitHub.

---

# 📜 License

This project is licensed under the MIT License.
