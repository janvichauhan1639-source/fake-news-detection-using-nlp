# 📰 Fake News Detection System using NLP & Machine Learning

An end-to-end **Fake News Detection System** developed using **Python, Natural Language Processing (NLP), TF-IDF Vectorization, and Machine Learning**. The system classifies news articles as **Fake** or **Real** using multiple supervised machine learning algorithms and provides real-time prediction through an interactive **Streamlit Web Application**.

---

# 📌 Project Overview

This project demonstrates a complete Machine Learning workflow from raw news articles to a deployed web application.

### Project Workflow

* 📥 Dataset Loading
* 📊 Exploratory Data Analysis (EDA)
* 🧹 Data Cleaning
* 📝 NLP Text Preprocessing
* 🔤 TF-IDF Feature Engineering
* 🤖 Machine Learning Model Training
* 📈 Model Comparison
* 📊 Model Evaluation
* 💾 Model Serialization
* 🌐 Streamlit Web Deployment

---

# 📷 Application Preview

> Replace `app.png` with your latest application screenshot.

![Application Screenshot](screenshots/app.png)

---

# 🚀 Features

* ✅ Fake & Real News Classification
* ✅ NLP Text Preprocessing
* ✅ Stopword Removal
* ✅ Porter Stemming
* ✅ TF-IDF Feature Extraction
* ✅ Multiple Machine Learning Models
* ✅ Automatic Best Model Selection
* ✅ Saved Model (.pkl)
* ✅ Interactive Streamlit Web App
* ✅ Modular Python Project Structure

---

# 🤖 Machine Learning Models

| Model                   |   Accuracy |
| ----------------------- | ---------: |
| Logistic Regression     | **98.66%** |
| Multinomial Naive Bayes | **93.26%** |
| Linear SVM              | **99.41%** |
| ⭐ Random Forest         | **99.74%** |

### 🏆 Best Model

**Random Forest Classifier**

---

# 🛠 Technologies Used

| Technology   | Purpose             |
| ------------ | ------------------- |
| Python       | Programming         |
| Pandas       | Data Analysis       |
| NumPy        | Numerical Computing |
| Scikit-learn | Machine Learning    |
| NLTK         | NLP                 |
| TF-IDF       | Feature Engineering |
| Joblib       | Model Saving        |
| Streamlit    | Web Application     |
| Git          | Version Control     |
| GitHub       | Project Hosting     |

---

# 📂 Project Structure

```text
fake-news-detection-using-nlp/
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
├── screenshots/
│   └── app.png
│
├── reports/
│
├── app.py
├── fake_news.py
├── preprocessing.py
├── train.py
├── predict.py
├── README.md
├── requirements.txt
├── runtime.txt
└── LICENSE
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/janvichauhan1639-source/fake-news-detection-using-nlp.git
```

Move into the project

```bash
cd fake-news-detection-using-nlp
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Project

### Train the Model

```bash
python train.py
```

### Predict News

```bash
python predict.py
```

### Run Streamlit App

```bash
streamlit run app.py
```

---

# 📊 Model Evaluation

Evaluation Metrics Used:

* Accuracy Score
* Precision
* Recall
* F1 Score
* Classification Report
* Model Comparison

---

# 🌐 Live Demo

**Streamlit App**

https://fake-news-detector-janvi.streamlit.app

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

**GitHub**

https://github.com/janvichauhan1639-source

**LinkedIn**

(Add your LinkedIn Profile URL)

---

# ⭐ Support

If you found this project useful, please consider giving this repository a ⭐ on GitHub.

---

# 📜 License

This project is licensed under the MIT License.
