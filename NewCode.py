# ── Libraries ─────────────────────────────
from openai import OpenAI

# ── Configuration ─────────────────────────
API_KEY = "Your Api key here"  # Your Router AI key
BASE_URL = "Router Base Url added here"  # Router AI base URL

# Create a client (new OpenAI 1.0+)
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# Optional: keep last few messages for context
chat_history = []

# ── Chat Function ─────────────────────────
def chat():
    print("="*50)
    print("Welcome to Eman's Chatbot! Type 'quit' to exit")
    print("="*50)

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["quit", "exit"]:
            print("Bot: Goodbye! 👋")
            break

        # Add user message to conversation memory
        chat_history.append({"role": "user", "content": user_input})

        try:
            # Call Router AI / OpenAI 1.0+ API
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",     # Efficient chat model
                messages=chat_history,
                max_tokens=150,
                temperature=0.7
            )

            # Extract bot reply
            bot_reply = response.choices[0].message.content.strip()
            print("Bot:", bot_reply)

            # Add bot reply to memory
            chat_history.append({"role": "assistant", "content": bot_reply})

            # Keep only last 10 messages
            if len(chat_history) > 10:
                chat_history[:] = chat_history[-10:]

        except Exception as e:
            print("Bot: Sorry, something went wrong. Try again.")
            print("Error:", e)

# ── Run Program ───────────────────────────
if __name__ == "__main__":
    chat()