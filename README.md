# Healthcare Chatbot

A **Healthcare Chatbot** designed to assist users with information regarding various diseases, their symptoms, and possible consultation suggestions. Built with **Python**, this chatbot provides easy access to healthcare-related information through an interactive interface. It covers a wide range of diseases including common cold, fever, diabetes, depression, asthma, and more.

## Features

- **Disease Information**: Get detailed descriptions of diseases like common cold, fever, diabetes, depression, asthma, etc.
- **Symptom Checker**: Input your symptoms to receive guidance on potential conditions.
- **Consultation Suggestions**: Based on the symptoms, the chatbot suggests consulting with healthcare professionals.
- **Interactive and User-Friendly**: Provides easy-to-understand responses to user queries, helping users navigate health concerns.
- **AI Integration**: Uses AI to understand and respond to queries in a natural language format.

---

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Code Overview](#code-overview)
- [Future Enhancements](#future-enhancements)
- [Contributors](#contributors)
- [License](#license)

---

## Technologies Used

- **Python**: Main programming language for chatbot development.
- **NLTK**: Natural Language Toolkit used for text processing and tokenization.
- **Tkinter**: Python's built-in library for developing a graphical user interface (GUI).
- **JSON**: Data format to store and access disease information.
- **Machine Learning**: Used for text classification and understanding user intent.

---

## Setup & Installation

### Prerequisites

Ensure you have Python installed on your machine. You can download it from [here](https://www.python.org/downloads/).

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/healthcare-chatbot.git
   
2. Navigate to the project directory:

   ```bash
   cd healthcare-chatbot

3. Install the required dependencies:

   ```bash
   pip install nltk

4. (Optional) Download necessary NLTK data:

   ```bash
   import nltk
   nltk.download('punkt')

5. Run the chatbot:
   ```bash
   python main.py

## Usage

1. **Start the Chatbot**: Once the program runs, a GUI window will open.
2. **Enter Your Query**: Type in a health-related question or symptom in the input box (e.g., "I have a cold").
3. **Receive Information**: The chatbot will respond with relevant details about the condition, its symptoms, and consultation suggestions.
4. **Consultation Suggestions**: If the symptoms match a particular disease, the bot will suggest you to consult a healthcare professional.

---
## Concepts used to train our model

- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **Receive Information**
- **Support Vector Classification (SVC)**
- **Decision Trees**
- **Gaussian Naive Bayes (GNB)**
- **Random Forest**
- **XGBoost**
---


## Code Overview

### Main Components:

1. **Disease Data**: 
   - Stored in `diseases.json`, containing various diseases, their symptoms, and consultation information.

2. **Chatbot Logic**: 
   - The core of the chatbot is handled in `chatbot.py`, where user input is processed using NLP to extract meaningful information.
   - It matches the input with the stored data to provide responses.

3. **GUI with Tkinter**:
   - The `gui.py` file contains the code to create a user interface using Tkinter.
   - It includes an input box for user queries, a display area for responses, and buttons for user interaction.

4. **Machine Learning**:
   - The chatbot uses basic ML techniques to classify user input and map it to the relevant disease.
   - It leverages tokenization and other NLP techniques to understand and process the text.

---

## Future Enhancements

- **Expanded Disease Coverage**: Adding more diseases and symptoms to improve the chatbot's knowledge base.
- **Voice Integration**: Implementing voice-to-text functionality for a hands-free experience.
- **Improved ML Models**: Enhancing the chatbot's ability to diagnose with more advanced AI models.
- **Medication Suggestions**: Along with consultation, suggesting over-the-counter medications for minor illnesses.
- **Multi-Language Support**: Implementing multi-language support to cater to non-English speaking users.

---

## Contributors

- **Gaurav kumar**[23bcs031] (Team Leader) - Developed the core logic. 
- **Akash kumar chaurasiya**[23bcs010]  - NLP integration. 
- **Babul kumar**[23bcs023]  - GUI interface.
- **Pranav kumar**[23bcs067]  - Data collector. 
- **Dipenshu Deep Bhat**[21bcs034] (Mentor) - Additional support and guidance in the project development.

---

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.

---

## Screenshots
![Chatbot Interface](img.jpeg)




