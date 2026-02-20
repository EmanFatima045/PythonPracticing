import numpy as np
from collections import defaultdict, deque

text = "hello how are you i am fine What About You? how are you doing today Have you done your assigned task of python? is supervisor satisfied with your work?how is your performance in office ? i am doing my tasks perfectly"
words = text.split(); markov = defaultdict(list); memory = deque(maxlen=5)
for i in range(len(words)-1): markov[words[i]].append(words[i+1])

while True:
    user = input("You: "); memory.append(user)
    start = user.split()[-1] if user.split()[-1] in markov else np.random.choice(list(markov.keys()))
    response = start
    for _ in range(5): response += " " + np.random.choice(markov[response.split()[-1]])
    print("Bot:", response)