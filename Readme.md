# Eman's Chatbot using Markov Chain

## 1. Project Overview
This project implements a **text-based chatbot** using the **Markov Chain algorithm**.  
The chatbot learns from a preloaded text corpus and the user’s recent messages to generate sentences based on word probabilities.

**Key Features:**
- Learns from a **predefined text base** to generate better initial responses.
- Updates its word mapping in **real-time** from the last few user messages.
- Generates sentences using a **first-order Markov chain**.
- Maintains **conversation context** using the last 5 user messages.

---

## 2. Technologies Used
- **Python 3.x** – Programming language  
- **random** – To randomly select words  
- **re (Regular Expressions)** – For cleaning and preprocessing text  
- **collections.deque** – For maintaining chat history (last 5 messages)  

---

## 3. Installation & Setup
1. Make sure **Python 3.x** is installed on your system.
2. Clone or download this project folder.
3. No additional external libraries are required.
4. Optionally, update the `text_base` variable with a larger text corpus to improve responses.

---

## 4. How It Works

### Preprocessing
- Converts all text to lowercase.
- Removes punctuation and numbers.
- Splits text into individual words (tokens).

### Learning (Markov Model)
- Creates a **word_map dictionary** where each word maps to a list of possible next words.
- Updates `word_map` from both the **preloaded text base** and **recent user messages**.

### Reply Generation
- Picks a random starting word from the dictionary.
- Generates a sentence of up to 10 words by selecting the next word from possible options in `word_map`.
- Stops if no next word is available.

---

## 5. Usage Instructions
1. Run the Python script:

```bash
python chatbot.py