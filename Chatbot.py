import random
import re
from collections import deque

# This dictionary stores word connections (Markov model)
word_map = {}

# This remembers the last 5 user inputs for chat context
chat_history = deque(maxlen=5)

# 2. Preprocessing function
def preprocess(sentence):
    """
    Converts text to lowercase, removes punctuation,
    and splits into a list of words (tokens)
    """
    sentence = sentence.lower()  # convert all letters to lowercase
    sentence = re.sub(r'[^a-zA-Z\s]', '', sentence)  # remove symbols/numbers
    tokens = sentence.split()  # split into words
    return tokens

# 3. Learning function
def learn(data):
    """
    Updates the word_map dictionary with word pairs
    from the given text data
    """
    tokens = preprocess(data)  # clean and tokenize text

    # Loop through each word and its next word
    for index in range(len(tokens) - 1):
        first_word = tokens[index]
        second_word = tokens[index + 1]

        # If the first_word is not in dictionary, add it with empty list
        if first_word not in word_map:
            word_map[first_word] = []

        # Append the next word to the list of the first word
        word_map[first_word].append(second_word)

# 4. Reply / sentence generation
def reply():
    """
    Generates a random sentence using the word_map
    """
    if not word_map:
        return "I have no knowledge yet."  # if dictionary empty

    # Pick a random starting word from the dictionary
    starting_point = random.choice(list(word_map.keys()))
    sentence_builder = [starting_point]  # start the sentence

    # Build sentence of max 10 words
    for step in range(10):
        last_word = sentence_builder[-1]  # get the last word in sentence

        if last_word in word_map:
            # Pick a random next word from possible choices
            next_choice = random.choice(word_map[last_word])
            sentence_builder.append(next_choice)  # add it to the sentence
        else:
            break  # if no next word, stop

    # Join words into a single string
    return " ".join(sentence_builder)


# 5. Preload text base for better responses
text_base = """
Hello how are you doing today I hope you are having a good day
i love programming and learnning new things every day
the weather is nice and suuny outside perfect for a walk
"""

# Learn from the text base before starting the chat
learn(text_base)

# -------------------------------
# 6. Chat loop
# -------------------------------
print("Chatbot is running... Type 'bye' to exit.\n")

while True:
    # Get user input
    user_text = input("Welcome to the Eman's Chatbot! how can i help you? ")

    # Exit condition
    if user_text.lower() == "bye":
        print("Bot: See you later!")
        break

    # Add user message to chat history
    chat_history.append(user_text)

    # Combine last few messages into a single string
    combined_text = " ".join(chat_history)

    # Learn from the latest user messages
    learn(combined_text)

    # Generate bot reply
    bot_answer = reply()
    print("Bot:", bot_answer)
