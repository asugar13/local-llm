from openai import OpenAI

# Points to your local mlx-lm server instead of Ollama
client = OpenAI(base_url="http://localhost:8080/v1", api_key="none")

MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"


class ConversationManager:
    def __init__(self, model: str = MODEL, system_prompt: str = None):
        self.model = model
        self.history = []
        if system_prompt:
            self.history.append({
                "role": "system",
                "content": system_prompt
            })

    def chat(self, user_message: str) -> str:
        """Send a message and get a response. History is maintained."""
        self.history.append({
            "role": "user",
            "content": user_message
        })

        response = client.chat.completions.create(
            model=self.model,
            messages=self.history
        )

        assistant_message = response.choices[0].message.content
        self.history.append({
            "role": "assistant",
            "content": assistant_message
        })
        return assistant_message

    def stream_chat(self, user_message: str):
        """Generator version: yields tokens as they are produced."""
        self.history.append({
            "role": "user",
            "content": user_message
        })

        full_response = ""
        for chunk in client.chat.completions.create(
            model=self.model,
            messages=self.history,
            stream=True
        ):
            token = chunk.choices[0].delta.content or ""
            full_response += token
            yield token

        self.history.append({
            "role": "assistant",
            "content": full_response
        })

    def reset(self):
        """Clear conversation history (keep system prompt if set)."""
        self.history = [m for m in self.history if m["role"] == "system"]
