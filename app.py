import streamlit as st
import joblib
from preprocessing import clean_text


st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)



model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")



st.sidebar.title("📌 About Project")

st.sidebar.write("""
This Fake News Detection System is developed using:

✅ Python

✅ Natural Language Processing (NLP)

✅ TF-IDF Feature Extraction

✅ Machine Learning

✅ Linear SVM
""")

st.sidebar.markdown("---")

st.sidebar.success("Developed by Janvi Chauhan")


st.markdown(
"""
<h1 style='text-align:center;color:#1f77b4;'>
📰 Fake News Detection System
</h1>
""",
unsafe_allow_html=True
)

st.markdown(
"""
<h4 style='text-align:center;color:gray;'>
Detect whether a news article is Fake or Real using Machine Learning & NLP
</h4>
""",
unsafe_allow_html=True
)

st.markdown("---")



news = st.text_area(
    "📝 Paste News Article Here",
    placeholder="Paste your news article here...",
    height=220
)



if st.button("🔍 Predict", use_container_width=True):

    if news.strip() == "":

        st.warning("⚠ Please enter some news text.")

    else:

        clean_news = clean_text(news)

        vector = vectorizer.transform([clean_news])

        prediction = model.predict(vector)

        st.markdown("---")

        st.subheader("Prediction Result")

        if prediction[0] == 0:

            st.error("🚨 FAKE NEWS")

        else:

            st.success("✅ REAL NEWS")

        st.markdown("---")

        st.subheader("Processed Text")

        st.write(clean_news)



st.markdown("---")

st.caption(
"© 2026 Fake News Detection System | Developed by Janvi Chauhan"
)