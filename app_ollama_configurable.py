import streamlit as st
import ollama

st.set_page_config(page_title="Qwen Local Chat", page_icon="🤖")
st.title("Qwen 2.5 - Local Chat")
st.caption("Running entirely on your machine. No data leaves this device.")

# Sidebar
with st.sidebar:
    st.header("Settings")
    model = st.selectbox("Model", ["qwen2.5:3b", "qwen2.5:7b", "qwen2.5:14b"], index=1)

    system_prompt = st.text_area(
        "System prompt",
        value="You are a helpful, concise, and friendly assistant.",
        height=150
    )

    if st.button("Apply & reset conversation"):
        st.session_state.messages = [{"role": "system", "content": system_prompt}]
        st.rerun()

    if st.button("Clear conversation"):
        st.session_state.messages = [st.session_state.messages[0]]
        st.rerun()

# Initialise session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": system_prompt}
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
            model=model,
            messages=st.session_state.messages,
            stream=True
        ):
            full_response += chunk["message"]["content"]
            placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
