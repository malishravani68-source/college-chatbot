import os
import pickle
import json
import random
import streamlit as st

# -----------------------------------
# Train model automatically
# -----------------------------------

if not os.path.exists("model.pkl"):
    import train

# -----------------------------------
# Load model
# -----------------------------------

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("intents.json") as f:
    data = json.load(f)

# -----------------------------------
# Streamlit UI
# -----------------------------------

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