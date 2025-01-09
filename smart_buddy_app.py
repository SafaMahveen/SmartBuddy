import os
import re
import json
import random
import datetime
import csv
import streamlit as st
import nltk
import ssl
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Eq, solve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Fix SSL issue for nltk downloads
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download("punkt")

# Load intents from the JSON file
file_path = os.path.abspath("intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Initialize the vectorizer and classifier
vectorizer = TfidfVectorizer(ngram_range=(1, 4))
clf = LogisticRegression(random_state=0, max_iter=10000)

# Prepare the data for training
tags = []
patterns = []
for intent in intents:
    for pattern in intent["patterns"]:
        tags.append(intent["tag"])
        patterns.append(pattern)

# Train the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Function to evaluate basic arithmetic
def evaluate_arithmetic(expression):
    try:
        result = eval(expression)  
        return f"The result is: {result}"
    except Exception:
        return "Sorry, I couldn't evaluate that expression."

# Function to solve quadratic equations
def solve_quadratic(equation):
    try:
        x = symbols("x")
        eq = Eq(eval(equation.replace("=", "-(") + ")"), 0)
        solutions = solve(eq, x)
        return f"The solutions are: {', '.join(map(str, solutions))}"
    except Exception:
        return "Please enter a valid quadratic equation in the form ax^2 + bx + c = 0."

# Function to solve linear equations
def solve_linear(equation):
    try:
        x = symbols("x")
        eq = Eq(eval(equation.replace("=", "-(") + ")"), 0)
        solution = solve(eq, x)
        return f"The solution is: {solution}"
    except Exception:
        return "Please enter a valid linear equation in the form ax + b = c."

# Function to plot equations
def plot_equation(equation):
    try:
        x_vals = np.linspace(-10, 10, 400)
        y_vals = eval(equation.replace("^", "**"))
        plt.figure(figsize=(6, 4))
        plt.plot(x_vals, y_vals, label=equation)
        plt.axhline(0, color="blue", linewidth=0.5)
        plt.axvline(0, color="blue", linewidth=0.5)
        plt.grid(color="gray", linestyle="--", linewidth=0.5)
        plt.title("Graph of the Equation")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        st.pyplot(plt)
    except Exception:
        st.write("Unable to plot the equation. Please ensure it's a valid mathematical expression.")

# Main chatbot function
def chatbot(input_text):
    input_text = input_text.lower().strip()

    # Detect quadratic equations
    if re.search(r"\bx\^2\b", input_text):
        return solve_quadratic(input_text)

    # Detect linear equations
    elif re.search(r"\bx\b", input_text):
        return solve_linear(input_text)

    # Detect basic arithmetic
    elif re.match(r"^[\d\s\+\-\*/\(\)\.]+$", input_text):
        return evaluate_arithmetic(input_text)

    # Process intents if no arithmetic expression or equation found
    input_text_transformed = vectorizer.transform([input_text])
    tag = clf.predict(input_text_transformed)[0]
    for intent in intents:
        if intent["tag"] == tag:
            response = random.choice(intent["responses"])
            return response

    # Fallback for unrecognized input
    return "I'm sorry, I didn't understand that. Can you try rephrasing?"

# Main Streamlit app
def main():
    st.title("SmartBuddy: Your Friendly AI Chatbot")

    # Sidebar menu options
    menu = ["Talk With SmartBuddy", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Talk With SmartBuddy":
        st.write("Welcome to SmartBuddy. Please type a message to get started.")

        # Initialize chat log if it doesn't exist
        if not os.path.exists("chat_log.csv"):
            with open("chat_log.csv", "w", newline="", encoding="utf-8") as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(["User Input", "SmartBuddy Response", "Timestamp"])

        user_input = st.text_input("You:")

        if user_input:
            response = chatbot(user_input)
            st.text_area("SmartBuddy:", value=response, height=120)

            # Record the current timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Log the interaction to the CSV file
            with open("chat_log.csv", "a", newline="", encoding="utf-8") as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input, response, timestamp])

            # Option to plot equations
            if "solutions are" in response or "solution is" in response:
                plot = st.checkbox("Plot the equation?")
                if plot:
                    plot_equation(user_input)

    elif choice == "Conversation History":
        st.header("Conversation History")
        with open("chat_log.csv", "r", encoding="utf-8") as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  # Skip the header row
            for row in csv_reader:
                st.text(f"User: {row[0]}")
                st.text(f"SmartBuddy: {row[1]}")
                st.text(f"Timestamp: {row[2]}")
                st.markdown("---")

    elif choice == "About":
        st.write("SmartBuddy is designed to help with math problems and answer queries.")
if __name__ == "__main__":
    main()
