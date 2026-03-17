import streamlit as st
import ollama

st.set_page_config(page_title="Qwen Local Chat", page_icon="🤖")
st.title("Qwen 2.5 - Local Chat")
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

        for chunk in ollama.chat(
            model="qwen2.5:7b",
            messages=st.session_state.messages,
            stream=True
        ):
            full_response += chunk["message"]["content"]
            placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Sidebar
with st.sidebar:
    st.header("Settings")
    st.caption("Model: qwen2.5:7b")
    if st.button("Clear conversation"):
        st.session_state.messages = [st.session_state.messages[0]]
        st.rerun()
