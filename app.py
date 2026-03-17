import streamlit as st
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="none")
MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"

st.set_page_config(page_title="Qwen Local Chat", page_icon="🤖")
st.title("Qwen 2.5 - Local Chat by Asier")
st.caption("Running entirely on your machine. No data leaves this device.")

# Initialise session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful, concise, and friendly assistant."}
    ]

# Render existing messages
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# Handle new user input
if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        print(st.session_state.messages)
        for chunk in client.chat.completions.create(
            model=MODEL,
            messages=st.session_state.messages,
            stream=True
        ):
            token = chunk.choices[0].delta.content or ""
            full_response += token
            placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Sidebar
with st.sidebar:
    st.header("Settings")
    st.caption(f"Model: {MODEL}")
    if st.button("Clear conversation"):
        st.session_state.messages = [st.session_state.messages[0]]
        st.rerun()
