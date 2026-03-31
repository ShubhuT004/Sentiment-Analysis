import streamlit as st
import pickle
import re
import string
import nltk
import os
nltk_data_path = "/tmp/nltk_data"
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

nltk.download('stopwords', download_dir=nltk_data_path)
nltk.download('wordnet', download_dir=nltk_data_path)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

st.set_page_config(
    page_title="Review AI",
    page_icon="📊",
    layout="centered"
)

@st.cache_resource
def load_models():
    model = pickle.load(open("model.pkl", "rb"))
    tfidf = pickle.load(open("tfidf.pkl", "rb"))
    
    # Optional Naive Bayes
    try:
        nb_model = pickle.load(open("nb_model.pkl", "rb"))
    except:
        nb_model = None

    return model, tfidf, nb_model

model, tfidf, nb_model = load_models()

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"(.)\1+", r"\1\1", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    cleaned_words = []

    for w in words:
        if w not in stop_words:
            w = lemmatizer.lemmatize(w)
            cleaned_words.append(w)

    return " ".join(cleaned_words)

st.title("📊 Customer Review Analyzer")

st.write("Analyze customer sentiment across any platform 🚀")

st.markdown("**Try Examples:**")
st.code("Delivery was very slow and bad")
st.code("Amazing product, very fast and smooth")

user_input = st.text_area(
    "Enter Review:",
    placeholder="Type your review here...",
    height=150
)

if st.button("Analyze Sentiment"):

    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        with st.spinner("Analyzing..."):
            cleaned = clean_text(user_input)
            vector = tfidf.transform([cleaned])

            if model_choice == "Naive Bayes" and nb_model:
                pred = nb_model.predict(vector)[0]
                prob = nb_model.predict_proba(vector).max()
            else:
                pred = model.predict(vector)[0]
                prob = model.predict_proba(vector).max()

        st.divider()

        if str(pred).lower() == "positive":
            st.success(f"😊 Positive ({round(prob*100,1)}%)")
        else:
            st.error(f"😡 Negative ({round(prob*100,1)}%)")

        st.progress(float(prob))

        if prob > 0.75:
            st.success("High confidence prediction")
        elif prob > 0.5:
            st.info("Moderate confidence")
        else:
            st.warning("Low confidence - uncertain prediction")
