# Qwen Local Chat

A local AI chat application powered by [Ollama](https://ollama.com) and Streamlit. Runs entirely on your machine — no data leaves your device.

## Tech stack

- **Streamlit** — chat UI
- **Ollama Python SDK** — local LLM inference (Qwen 2.5)
- **SQLite** — persistent conversation history (`database.py`)
- **PyMuPDF / python-docx** — file attachment parsing (PDF, DOCX)
- **streamlit-mic-recorder** — voice input

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed with at least one model pulled (e.g. `ollama pull qwen2.5:7b`)

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run the application

First, make sure the Ollama server is running:

```bash
ollama serve
```

Then launch the app:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Features

All required features from the assignment are implemented:

- **Multi-turn conversation** — full message history is sent to the model on every turn
- **Streaming output** — responses appear token-by-token via a background thread
- **Conversation reset** — clear history and start fresh at any time
- **Configurable system prompt** — editable in the sidebar, persisted per conversation
- **Model indicator** — active model and temperature shown above the chat

Additional features also implemented:

- **Model selector** — switch between any locally available Ollama model at runtime
- **Temperature slider** — adjust creativity (0 = deterministic, 1 = creative)
- **Conversation export** — download chat history as JSON or TXT
- **Token counter** — live input/output/total token usage in the sidebar
- **Persistent history** — all conversations saved to SQLite, survive restarts
- **Conversation search & rename** — filter and rename past conversations
- **File upload** — attach PDF, DOCX, TXT, MD, or CSV files as context; the file's text is extracted and prepended to your message so the model can answer questions about the document
- **Voice input** — dictate messages via microphone
- **Copy & regenerate** — copy any response or regenerate the last one
