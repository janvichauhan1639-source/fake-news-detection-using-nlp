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
This Fake News Detection System is built using:

✅ Python

✅ Natural Language Processing (NLP)

✅ TF-IDF Feature Extraction

✅ Machine Learning

✅ Random Forest Classifier

✅ Streamlit
""")

st.sidebar.markdown("---")

st.sidebar.info("🎓 Developed by Janvi Chauhan")



st.markdown("""
<h1 style='text-align:center;color:#1f77b4;'>
📰 Fake News Detection System
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<h4 style='text-align:center;color:gray;'>
Detect whether a news article is Fake or Real using Machine Learning & NLP
</h4>
""", unsafe_allow_html=True)

st.markdown("---")



news = st.text_area(
    "📝 Paste News Article Here",
    placeholder="Paste any news article here...",
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

        st.subheader("📊 Prediction Result")

        if prediction[0] == 0:
            st.error("🚨 FAKE NEWS")
        else:
            st.success("✅ REAL NEWS")

        
        st.markdown("### 🎯 Confidence Score")

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(vector)

            confidence = probability.max() * 100

            st.progress(int(confidence))

            st.write(f"**{confidence:.2f}% Confidence**")

        else:
            st.info("Confidence score is not available for this model.")

        
        with st.expander("🔍 View Processed Text"):

            st.write(clean_news)


st.markdown("---")

st.caption(
    "© 2026 Fake News Detection System | Developed by Janvi Chauhan"
)