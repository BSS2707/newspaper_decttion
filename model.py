import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

@st.cache_data
def load_data():
    df = pd.read_csv("fake_or_real_news.csv")
    return df

df = load_data()

# Ensure labels are numeric (0 = Real, 1 = Fake)
df['label'] = df['label'].map({'REAL': 0, 'FAKE': 1})

X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=200)
model.fit(X_train_tfidf, y_train)

st.title("📰 Fake News Detector")
st.write("Paste any news article text below and check if it's Fake or Real.")

user_input = st.text_area("Enter news article text:")

if st.button("Check"):
    if user_input.strip() != "":
        input_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(input_tfidf)[0]
        st.subheader("Result:")
        if prediction == 0:
            st.success("✅ Real News")
        else:
            st.error("🚨 Fake News")
    else:
        st.warning("Please enter some text to analyze.")
