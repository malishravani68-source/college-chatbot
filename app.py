import os
import pickle
import json
import random
import streamlit as st

# -----------------------------
# Load files
# -----------------------------

BASE_DIR = os.path.dirname(__file__)

model_path = os.path.join(BASE_DIR, "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")
intents_path = os.path.join(BASE_DIR, "intents.json")

# Load ML model
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Load vectorizer
with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

# Load intents
with open(intents_path, "r") as f:
    intents = json.load(f)

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="College Chatbot")

st.title("🎓 College Enquiry Chatbot")

st.write("Ask any college-related question.")

# User input
user_input = st.text_input("Enter your question")

# -----------------------------
# Chatbot Logic
# -----------------------------

if user_input:

    # Convert user text into vector
    X = vectorizer.transform([user_input])

    # Predict tag
    prediction = model.predict(X)[0]

    response = "Sorry, I don't understand."

    # Find matching response
    for intent in intents["intents"]:
        if intent["tag"] == prediction:
            response = random.choice(intent["responses"])
            break

    # Show response
    st.success(response)