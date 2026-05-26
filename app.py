import streamlit as st
import pickle
import json
import random
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model safely
model_path = os.path.join(BASE_DIR, "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")

model = pickle.load(open(model_path, "rb"))
vectorizer = pickle.load(open(vectorizer_path, "rb"))

# Load dataset
json_path = os.path.join(BASE_DIR, "intents.json")
with open(json_path) as file:
    data = json.load(file)

st.title("🎓 College Enquiry Chatbot")

user_input = st.text_input("Ask your question:")

if st.button("Send"):

    X = vectorizer.transform([user_input])
    prediction = model.predict(X)[0]

    response = "Sorry, I don't understand."

    for intent in data["intents"]:
        if intent["tag"] == prediction:
            response = random.choice(intent["responses"])

    st.success(response)