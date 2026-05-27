import os
import json
import random
import pickle
import streamlit as st

# --------------------------------
# Create model if not exists
# --------------------------------

if not os.path.exists("model.pkl"):

    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB

    # Load intents
    with open("intents.json") as file:
        data = json.load(file)

    texts = []
    labels = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            texts.append(pattern)
            labels.append(intent["tag"])

    # Vectorize text
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)

    # Train model
    model = MultinomialNB()
    model.fit(X, labels)

    # Save files
    pickle.dump(model, open("model.pkl", "wb"))
    pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

# --------------------------------
# Load model
# --------------------------------

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Load intents
with open("intents.json") as file:
    data = json.load(file)

# --------------------------------
# Streamlit UI
# --------------------------------

st.title("🎓 College Enquiry Chatbot")

user_input = st.text_input("Ask your question")

if user_input:

    X = vectorizer.transform([user_input])

    prediction = model.predict(X)[0]

    response = "Sorry, I don't understand."

    for intent in data["intents"]:
        if intent["tag"] == prediction:
            response = random.choice(intent["responses"])

    st.success(response)