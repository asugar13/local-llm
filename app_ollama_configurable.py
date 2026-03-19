import json
import threading
import streamlit as st
import ollama
import database

database.init_db()

st.set_page_config(page_title="Qwen Local Chat", page_icon="🤖")
st.title("Qwen 2.5 - Local Chat")
st.caption("Running entirely on your machine. No data leaves this device.")

# Module-level state for thread communication (session state is not thread-safe)
if "_gen" not in st.__dict__:
    st._gen = {
        "buffer": "", "done": False, "stop": threading.Event(),
        "input_tokens": 0, "output_tokens": 0,
        "conversation_id": None,
    }

DEFAULT_SYSTEM = "You are a helpful, concise, and friendly assistant."
DEFAULT_MODEL  = "qwen2.5:7b"
MODELS         = ["qwen2.5:3b", "qwen2.5:7b", "qwen2.5:14b"]

# Initialise session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": DEFAULT_SYSTEM}]
if "total_input_tokens" not in st.session_state:
    st.session_state.total_input_tokens = 0
if "total_output_tokens" not in st.session_state:
    st.session_state.total_output_tokens = 0
if "is_generating" not in st.session_state:
    st.session_state.is_generating = False
if "pending_model" not in st.session_state:
    st.session_state.pending_model = DEFAULT_MODEL
if "pending_temperature" not in st.session_state:
    st.session_state.pending_temperature = 0.7
if "queued_prompt" not in st.session_state:
    st.session_state.queued_prompt = None
if "conversation_id" not in st.session_state:
    cid = database.create_conversation(DEFAULT_MODEL, DEFAULT_SYSTEM)
    st.session_state.conversation_id = cid
    st.session_state.title_set = False
if "title_set" not in st.session_state:
    st.session_state.title_set = False

# Sidebar
with st.sidebar:
    st.header("Conversations")

    if st.button("+ New conversation", use_container_width=True):
        current_system = st.session_state.messages[0]["content"]
        cid = database.create_conversation(st.session_state.pending_model, current_system)
        st.session_state.conversation_id = cid
        st.session_state.title_set = False
        st.session_state.messages = [{"role": "system", "content": current_system}]
        st.session_state.total_input_tokens = 0
        st.session_state.total_output_tokens = 0
        st.rerun()

    for conv in database.list_conversations():
        is_active = conv["id"] == st.session_state.conversation_id
        col_btn, col_del = st.columns([5, 1])
        with col_btn:
            label = f"**{conv['title']}**" if is_active else conv["title"]
            if st.button(label, key=f"conv_{conv['id']}", use_container_width=True):
                if not is_active:
                    data = database.load_conversation(conv["id"])
                    st.session_state.messages = (
                        [{"role": "system", "content": data["system_prompt"]}]
                        + data["messages"]
                    )
                    st.session_state.conversation_id = data["id"]
                    st.session_state.pending_model = data["model"]
                    st.session_state.title_set = True
                    st.session_state.total_input_tokens = 0
                    st.session_state.total_output_tokens = 0
                    st.rerun()
        with col_del:
            if st.button("🗑", key=f"del_{conv['id']}", help="Delete"):
                database.delete_conversation(conv["id"])
                if is_active:
                    cid = database.create_conversation(DEFAULT_MODEL, DEFAULT_SYSTEM)
                    st.session_state.conversation_id = cid
                    st.session_state.title_set = False
                    st.session_state.messages = [{"role": "system", "content": DEFAULT_SYSTEM}]
                    st.session_state.total_input_tokens = 0
                    st.session_state.total_output_tokens = 0
                st.rerun()

    st.divider()
    st.header("Settings")

    model_index = MODELS.index(st.session_state.pending_model) if st.session_state.pending_model in MODELS else 1
    model = st.selectbox("Model", MODELS, index=model_index)

    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=st.session_state.pending_temperature, step=0.1)
    st.caption("0 = deterministic (better for facts/code), 1 = creative (better for writing/brainstorming)")

    system_prompt = st.text_area(
        "System prompt",
        value=st.session_state.messages[0]["content"],
        height=150,
    )

    if st.button("Apply system prompt"):
        st.session_state.messages[0] = {"role": "system", "content": system_prompt}
        database.update_conversation_meta(st.session_state.conversation_id, model, system_prompt)
        st.rerun()

    st.download_button(
        "Export conversation",
        data=json.dumps(st.session_state.messages, indent=2),
        file_name="conversation.json",
        mime="application/json",
    )

    if st.button("Clear conversation"):
        cid = database.create_conversation(model, system_prompt)
        st.session_state.conversation_id = cid
        st.session_state.title_set = False
        st.session_state.messages = [{"role": "system", "content": system_prompt}]
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

# Fragment always defined for consistent run_every lifecycle
@st.fragment(run_every=0.1)
def streaming_display():
    if not st.session_state.is_generating:
        return

    gen = st._gen

    if gen["done"]:
        content = gen["buffer"]
        st.session_state.messages.append({"role": "assistant", "content": content})
        st.session_state.total_input_tokens += gen["input_tokens"]
        st.session_state.total_output_tokens += gen["output_tokens"]
        st.session_state.is_generating = False
        if gen["conversation_id"] is not None:
            database.save_message(gen["conversation_id"], "assistant", content)
        st.rerun(scope="app")
        return

    with st.chat_message("assistant"):
        st.markdown(gen["buffer"] + "▌")

    if st.button("⬛ Stop generating"):
        gen["stop"].set()
        content = gen["buffer"]
        st.session_state.messages.append({"role": "assistant", "content": content})
        st.session_state.is_generating = False
        if gen["conversation_id"] is not None:
            database.save_message(gen["conversation_id"], "assistant", content)
        st.rerun(scope="app")

streaming_display()


def start_generation(prompt: str, mdl: str, temp: float) -> None:
    cid = st.session_state.conversation_id

    if not st.session_state.title_set:
        database.set_conversation_title(cid, prompt)
        st.session_state.title_set = True

    database.save_message(cid, "user", prompt)
    database.update_conversation_meta(cid, mdl, st.session_state.messages[0]["content"])

    st.session_state.messages.append({"role": "user", "content": prompt})

    st._gen["buffer"] = ""
    st._gen["done"] = False
    st._gen["input_tokens"] = 0
    st._gen["output_tokens"] = 0
    st._gen["stop"].clear()
    st._gen["conversation_id"] = cid

    st.session_state.is_generating = True
    st.session_state.pending_model = mdl
    st.session_state.pending_temperature = temp

    messages_snapshot = list(st.session_state.messages)

    def generate(messages, m, t):
        gen = st._gen
        full_response = ""
        for chunk in ollama.chat(model=m, messages=messages, options={"temperature": t}, stream=True):
            if gen["stop"].is_set():
                break
            full_response += chunk["message"]["content"]
            gen["buffer"] = full_response
            if chunk.get("done"):
                gen["input_tokens"] = chunk.get("prompt_eval_count", 0)
                gen["output_tokens"] = chunk.get("eval_count", 0)
        gen["done"] = True

    threading.Thread(target=generate, args=(messages_snapshot, mdl, temp), daemon=True).start()


# Disable send button while generating
if st.session_state.is_generating:
    st.markdown("""
    <style>
    [data-testid="stChatInputSubmitButton"] { pointer-events: none; opacity: 0.3; }
    </style>
    """, unsafe_allow_html=True)

if prompt := st.chat_input("Type your message..."):
    if st.session_state.is_generating:
        st.session_state.queued_prompt = prompt
    else:
        start_generation(prompt, model, temperature)
        st.rerun()

if not st.session_state.is_generating and st.session_state.queued_prompt:
    queued = st.session_state.queued_prompt
    st.session_state.queued_prompt = None
    start_generation(queued, model, temperature)
    st.rerun()
