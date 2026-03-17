import json
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
if "total_input_tokens" not in st.session_state:
    st.session_state.total_input_tokens = 0
if "total_output_tokens" not in st.session_state:
    st.session_state.total_output_tokens = 0

# Sidebar
with st.sidebar:
    st.header("Settings")
    model = st.selectbox("Model", ["qwen2.5:3b", "qwen2.5:7b", "qwen2.5:14b"], index=1)

    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    st.caption("0 = deterministic (better for facts/code), 1 = creative (better for writing/brainstorming)")

    system_prompt = st.text_area(
        "System prompt",
        value="You are a helpful, concise, and friendly assistant.",
        height=150
    )

    if st.button("Apply & reset conversation"):
        st.session_state.messages = [{"role": "system", "content": system_prompt}]
        st.rerun()

    st.download_button(
        "Export conversation",
        data=json.dumps(st.session_state.messages, indent=2),
        file_name="conversation.json",
        mime="application/json"
    )

    if st.button("Clear conversation"):
        st.session_state.messages = [st.session_state.messages[0]]
        st.session_state.total_input_tokens = 0
        st.session_state.total_output_tokens = 0
        st.rerun()

    st.divider()
    st.subheader("Token usage")
    st.caption(f"Input: {st.session_state.total_input_tokens}")
    st.caption(f"Output: {st.session_state.total_output_tokens}")
    st.caption(f"Total: {st.session_state.total_input_tokens + st.session_state.total_output_tokens}")

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
            options={"temperature": temperature},
            stream=True
        ):
            full_response += chunk["message"]["content"]
            placeholder.markdown(full_response + "▌")
            if chunk.get("done"):
                st.session_state.total_input_tokens += chunk.get("prompt_eval_count", 0)
                st.session_state.total_output_tokens += chunk.get("eval_count", 0)

        placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.rerun()
