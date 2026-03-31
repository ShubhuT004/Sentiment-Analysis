import streamlit as st
import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Review AI",
    page_icon="📊",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_models():
    model = pickle.load(open("model.pkl", "rb"))
    tfidf = pickle.load(open("tfidf.pkl", "rb"))
    return model, tfidf

model, tfidf = load_models()

# ---------------- NLP SETUP ----------------
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

# ---------------- UI STYLING ----------------
st.markdown("""
<style>
.main {
    background-color: #f5f7f9;
}
.stButton>button {
    width: 100%;
    border-radius: 8px;
    height: 3em;
    background-color: #ff4b4b;
    color: white;
}
.sentiment-box {
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- UI ----------------
st.title("📊 Customer Review Analyzer")

st.write(
    "Enter a customer review below to determine if the sentiment is **Positive** or **Negative**."
)

user_input = st.text_area(
    "Review Text:",
    placeholder="Type your review here...",
    height=150
)

# ---------------- PREDICTION ----------------
if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        cleaned = clean_text(user_input)
        vector = tfidf.transform([cleaned])

        prediction = model.predict(vector)[0]
        confidence = model.predict_proba(vector).max()

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Predicted Sentiment",
                value=str(prediction).upper()
            )

        with col2:
            st.metric(
                label="Confidence Score",
                value=f"{round(confidence * 100, 1)}%"
            )

        if confidence > 0.75:
            st.success(f"The model is very confident this is {prediction}.")
        else:
            st.info("The model has moderate confidence in this result.")

        st.progress(float(confidence))