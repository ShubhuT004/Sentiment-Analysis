## 📊 Dataset

This project uses the **Sentiment140 dataset**, a widely used benchmark dataset for sentiment analysis in NLP.

🔗 Dataset Link: https://www.kaggle.com/datasets/kazanova/sentiment140

---

### 📌 Dataset Description

The dataset contains **1.6 million real-world tweets** labeled for sentiment analysis.

Each record includes:

* **target** → Sentiment label

  * `0` → Negative
  * `4` → Positive

* **text** → The actual tweet (customer-like review)

---

### ⚠️ Note on Dataset Usage

Due to GitHub file size limitations, the dataset is **not included** in this repository.

👉 To use this project:

1. Download the dataset from the link above
2. Place the CSV file inside a `data/` folder
3. Update the file path in the notebook if needed

---

### 🧠 Why This Dataset?

* Contains **real-world noisy text data** (slang, emojis, informal language)
* Covers **multiple domains** (e-commerce, transport, services, etc.)
* Helps build a **domain-independent sentiment analysis model**

---

## ⚙️ Techniques and Tools Used

### 📊 Exploratory Data Analysis (EDA)

* Analyzed dataset structure, class distribution, and text length
* Identified data imbalance and noise patterns
* Visualized word distributions and review lengths

---

### 🧹 Text Preprocessing

* Lowercasing text
* Removing URLs, numbers, and punctuation
* Stopword removal using NLTK
* Lemmatization using WordNetLemmatizer
* Handling repeated characters (e.g., "gooooood" → "good")

---

### 🔢 Feature Engineering

* **TF-IDF (Term Frequency–Inverse Document Frequency)**
* Converts text into numerical vectors based on word importance
* Limits vocabulary size for efficiency

---

### 🤖 Machine Learning Models

#### 1. Logistic Regression

* Main classification model
* Provides strong performance on text data

#### 2. Naive Bayes

* Baseline model for NLP tasks
* Fast and efficient for high-dimensional data

---

### 📈 Model Evaluation

* Accuracy Score
* Precision, Recall, F1-score
* Confusion Matrix

---

### 🌐 Deployment

* Built an interactive web app using **Streamlit**
* Enables real-time sentiment prediction

---

### 🧠 Key Learning

This project demonstrates:

* Handling **large-scale text data**
* Building a **complete NLP pipeline**
* Applying **ML models for classification**
* Creating a **domain-independent system** for real-world use cases

---
