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
        # Remove spaces for easier parsing
        expression = expression.replace(" ", "")
        math_pattern = r'[-+]?\d*\.\d+|[-+]?\d+|[+\-*/()]'
        tokens = re.findall(math_pattern, expression)
        expression_2 = "".join(tokens)
        result = eval(expression_2)
        return str(result)
    except Exception:
        return "Sorry, I couldn't evaluate that expression."

# Function to solve quadratic equations
def solve_quadratic(equation):
    try:
        # Replace ^ with ** for Python compatibility
        equation = equation.replace("^", "**")

        # Split into LHS and RHS
        lhs, rhs = equation.split("=")
        rhs = rhs.strip()  # Strip spaces

        # Form the symbolic equation
        x = symbols("x")
        eq = Eq(eval(lhs), eval(rhs))

        # Solve and format solutions
        solutions = solve(eq, x)
        return f"The solutions are: {', '.join(map(str, solutions))}"
    except Exception:
        return "Please enter a valid quadratic equation in the form ax^2 + bx + c = 0."

# Function to solve linear equations
def solve_linear(equation):
    try:
        # Replace ^ with ** for Python compatibility
        equation = equation.replace("^", "**")

        # Split into LHS and RHS
        lhs, rhs = equation.split("=")
        rhs = rhs.strip()  # Strip spaces

        # Form the symbolic equation
        x = symbols("x")
        eq = Eq(eval(lhs), eval(rhs))

        # Solve and format the solution
        solution = solve(eq, x)
        return f"The solution is: {', '.join(map(str, solution))}"
    except Exception:
        return "Please enter a valid linear equation in the form ax + b = c."

# Function to extract and classify mathematical expressions
def extract_math_expression(input_text):
    """
    Extract and classify the mathematical expression or equation.
    Returns a dictionary with type ('arithmetic', 'linear', 'quadratic') and the cleaned expression.
    """
    try:
        # Clean input by replacing "^" with "**" for Python compatibility
        input_text = input_text.replace("^", "**")

        # Regex patterns for different types of math expressions
        arithmetic_pattern = r"[\d\+\-\*/\(\)\.\s]+$"  # Matches basic arithmetic
        linear_pattern = r"[-+]?\d*x[-+x\d\s]*=[-+\d\s]*$"  # Matches linear equations
        quadratic_pattern = r"[-+]?\d*x\*\*2[-+x\d\s]*=[-+\d\s]*$"  # Matches quadratic equations

        # Check for quadratic equations
        if re.search(quadratic_pattern, input_text):
            return {"type": "quadratic", "expression": input_text.strip()}

        # Check for linear equations
        elif re.search(linear_pattern, input_text):
            return {"type": "linear", "expression": input_text.strip()}

        # Check for basic arithmetic
        elif re.search(arithmetic_pattern, input_text):
            return {"type": "arithmetic", "expression": input_text.strip()}

        # If no match, return None
        return None
    except Exception as e:
        return None

# Function to process the extracted math expression
def process_math_expression(expression_data):
    """
    Processes the extracted math expression or equation based on its type.
    """
    if not expression_data:
        return "I couldn't find a valid mathematical expression in your input."

    expr_type = expression_data["type"]
    expr = expression_data["expression"]

    try:
        if expr_type == "arithmetic":
            return evaluate_arithmetic(expr)
        elif expr_type == "linear":
            return solve_linear(expr)
        elif expr_type == "quadratic":
            return solve_quadratic(expr)
        else:
            return "I'm sorry, I couldn't understand the type of math query."
    except Exception as e:
        return f"An error occurred while processing the expression: {e}"

# Main chatbot function
def chatbot(input_text):
    # Try extracting and solving a math expression
    math_data = extract_math_expression(input_text)
    if math_data:
        return process_math_expression(math_data)

    # Process intents if no arithmetic expression or equation found
    input_text_transformed = vectorizer.transform([input_text])
    tag = clf.predict(input_text_transformed)[0]
    for intent in intents:
        if intent["tag"] == tag:
            response = random.choice(intent["responses"])
            return response

    # Fallback for unrecognized input
    return "I'm sorry, I didn't understand that. Can you try rephrasing?"

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
        st.write("This project aims to develop a chatbot that comprehends and responds to user inquiries based on predefined intents. The chatbot utilizes Natural Language Processing (NLP) techniques and Logistic Regression to interpret user input effectively.")

        st.subheader("Project Overview:")
        st.write("""
        The project consists of two main components:
        1. The chatbot is trained using NLP methods and a Logistic Regression model on a set of labeled intents.
        2. Streamlit is used to create a user-friendly web interface for the chatbot, allowing users to interact seamlessly.
        """)

        st.subheader("Dataset:")
        st.write("""
        The dataset comprises a collection of labeled intents and corresponding entities. It is structured as follows:
        - Intents: Categories representing user intent (e.g., "greeting", "movies", "math").
        - Entities: Specific phrases extracted from user input (e.g., "Hi", "What is 23+4-5?", etc..)
        - Text: The actual user input.
        """)

        st.subheader("Streamlit Chatbot Interface:")
        st.write("The chatbot interface is designed using Streamlit, featuring a text input box for user queries and a display area for chatbot responses. The trained model generates replies based on user input.")

        st.subheader("Conclusion:")
        st.write("This project successfully creates a chatbot capable of understanding and responding to user inquiries based on predefined intents. By leveraging NLP and Logistic Regression, along with Streamlit for the interface, the chatbot can be further developed with additional data and advanced NLP techniques.")
if __name__ == "__main__":
    main()
