# SmartBuddy
**SmartBuddy: Your Friendly AI Chatbot** is a Python-based chatbot that uses NLP (nltk) and machine learning (scikit-learn) to understand user intents and deliver relevant responses. It features conversation history tracking and an interactive web interface built with Streamlit.
<br>
Author - Safa Mahveen

## OVERVIEW
## Features

- **Mathematical Problem Solving**: SmartBuddy can solve arithmetic expressions, linear equations, and quadratic equations.
- **Intent Classification**: The chatbot can classify user input into predefined categories (intents) and respond accordingly.
- **Conversation History**: All conversations are logged and can be reviewed later.
- **Interactive Interface**: The user interface is built using Streamlit, making it intuitive and easy to interact with.

## Technologies Used

- **Streamlit**: For building the web-based user interface.
- **Scikit-learn**: For machine learning, specifically using Logistic Regression for text classification.
- **NLTK**: For Natural Language Processing (NLP) tasks such as tokenization.
- **SymPy**: For solving algebraic equations, including linear and quadratic equations.
- **NumPy** and **Matplotlib**: For numerical operations and plotting (though not used directly in this version, they are imported in the code).
- **Regular Expressions**: For parsing and extracting math expressions from user input.

## Installation Instructions

To run this project, you need to have Python installed on your system, along with the necessary libraries. You can install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

### How to Run

1. Clone this repository to your local machine:

```bash
git clone https://github.com/SafaMahveen/SmartBuddy.git
```

2. Navigate to the project directory:

```bash
cd smartbuddy
```

3. Install the required libraries:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit application:

```bash
streamlit run smart_buddy_app.py
```

This will open the chatbot interface in your web browser where you can start interacting with SmartBuddy.

## Project Structure

```
smartbuddy-chatbot/
│
├── smart_buddy_app.py               # Main Streamlit application
├── chat_log.csv         # Stores conversation history
├── intents.json         # JSON file containing intents and patterns for training the chatbot
├── requirements.txt     # List of dependencies
└── README.md            # Project documentation (this file)
```

### Explanation of the Code

- **smart_buddy_app.py**: This is the main file that runs the Streamlit app. It defines the chatbot’s logic, including processing user input, training the intent classifier, and displaying the chatbot's responses.
  
- **chat_log.csv**: All interactions between the user and SmartBuddy are logged here, including the user input, chatbot response, and timestamp.

- **intents.json**: This file contains predefined intents, each with patterns (examples of user input) and corresponding responses. These are used to train the intent classifier.

- **requirements.txt**: Contains the list of Python dependencies required to run the project.

## How SmartBuddy Works

1. **Input Processing**: The user inputs a message through the text box on the Streamlit interface. The message is then processed by the chatbot.
  
2. **Mathematical Expression Parsing**: If the input contains a mathematical expression (arithmetic, linear, or quadratic equation), the chatbot identifies and solves it.
  
3. **Intent Classification**: If the input is not a mathematical query, the chatbot classifies it using a Logistic Regression model trained on the intents defined in `intents.json`.
  
4. **Response Generation**: Based on the intent classification or solved mathematical expression, the chatbot generates a response.

5. **Conversation Logging**: Every user interaction and chatbot response is logged in `chat_log.csv` for future reference.

## Contributing

If you would like to contribute to the development of SmartBuddy, feel free to fork the repository and submit a pull request with your changes. 
