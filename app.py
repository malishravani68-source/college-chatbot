import streamlit as st
import pickle
import json
import random

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Load dataset
with open("intents.json") as file:
    data = json.load(file)

# Page title
st.title("🎓 College Enquiry Chatbot")

st.write("Ask any college related questions")

# User input
user_input = st.text_input("You:")

# Button
if st.button("Send"):

    # Convert input into vector
    X = vectorizer.transform([user_input])

    # Predict tag
    prediction = model.predict(X)[0]

    # Find response
    response = "Sorry, I don't understand."

    for intent in data["intents"]:
        if intent["tag"] == prediction:
            response = random.choice(intent["responses"])

    st.success(response)