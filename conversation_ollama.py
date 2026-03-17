import ollama

class ConversationManager:
    def __init__(self, model: str = "qwen2.5:7b", system_prompt: str = None):
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

        response = ollama.chat(
            model=self.model,
            messages=self.history
        )

        assistant_message = response["message"]["content"]
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
        for chunk in ollama.chat(
            model=self.model,
            messages=self.history,
            stream=True
        ):
            token = chunk["message"]["content"]
            full_response += token
            yield token

        self.history.append({
            "role": "assistant",
            "content": full_response
        })

    def reset(self):
        """Clear conversation history (keep system prompt if set)."""
        self.history = [m for m in self.history if m["role"] == "system"]
