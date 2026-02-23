from rapidfuzz import process, fuzz
import json
from collections import deque
import re

# Optional: memory for last 5 messages
chat_history = deque(maxlen=5)

# Load responses from JSON file
with open("responses.json", "r") as f:
    responses = json.load(f)

# Preprocess input: lowercase + remove punctuation
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return " ".join(text.split())

# Chat loop
print("Chatbot is running... Type 'bye' to exit.\n")

while True:
    user_text = input("You: ")
    if user_text.lower() == "bye":
        print("Bot:", responses.get("bye", "See you later!"))
        break

    user_clean = preprocess(user_text)
    chat_history.append(user_clean)

    # Efficient lookup using RapidFuzz
    match, score, _ = process.extractOne(
        user_clean, responses.keys(), scorer=fuzz.ratio
    )

    if score > 80:  # threshold for similarity
        bot_answer = responses[match]
    else:
        bot_answer = "Sorry, I don't understand that yet."

    print("Bot:", bot_answer)