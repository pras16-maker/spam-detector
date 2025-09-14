import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Training dataset
emails = [
    "Congratulations! You won a free lottery ticket",   # spam
    "Claim your free prize now!!!",                     # spam
    "Can we reschedule our meeting tomorrow?",          # not spam
    "Project deadline is next week, please prepare",    # not spam
    "Limited time offer, buy now and save 50%",         # spam
    "Let's catch up over lunch tomorrow",               # not spam,
]
labels = ["spam", "spam", "not spam", "not spam", "spam", "not spam"]

# Train model
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(emails)
model = MultinomialNB()
model.fit(X, labels)

# Streamlit UI
st.set_page_config(page_title="Spam Detector", page_icon="📧")
st.title("📧 Spam Detector")
st.write("Enter an email message below to check if it's **Spam** or **Not Spam**.")

user_input = st.text_area("Email Text", height=150)

if st.button("Check Spam"):
    if user_input.strip() != "":
        test_vec = vectorizer.transform([user_input])
        prediction = model.predict(test_vec)[0]
        if prediction == "spam":
            st.error("🚨 Prediction: SPAM")
        else:
            st.success("✅ Prediction: NOT SPAM")
    else:
        st.warning("Please enter some text first.")
