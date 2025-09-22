class Chatbot:
    def __init__(self, model="gpt-4o-mini", client=None, prompt_path="prompt.txt"):
        # Load SYSTEM_PROMPT from file
        with open(prompt_path, "r", encoding="utf-8") as f:
            SYSTEM_PROMPT = f.read().strip()
        self.model = model
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.client = client

    def generate_response(self, user_input: str) -> str:
        self.history.append({"role": "user", "content": user_input})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.history,
            max_tokens=200,
        )
        reply = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": reply})
        return reply