import io
import json
import re
import subprocess
import threading
import streamlit as st
import ollama
import database
from streamlit_mic_recorder import speech_to_text


def extract_text(file) -> str:
    name = file.name.lower()
    if name.endswith(".pdf"):
        import fitz
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return "\n".join(page.get_text() for page in doc)
    elif name.endswith(".docx"):
        from docx import Document
        return "\n".join(p.text for p in Document(io.BytesIO(file.read())).paragraphs)
    else:
        return file.read().decode("utf-8", errors="ignore")


def render_markdown(content: str) -> None:
    """Render markdown with proper LaTeX delimiters for Streamlit's KaTeX renderer."""
    content = re.sub(r'\\\((.+?)\\\)', r'$\1$', content, flags=re.DOTALL)
    content = re.sub(r'\\\[(.+?)\\\]', r'$$\1$$', content, flags=re.DOTALL)
    st.markdown(content)

database.init_db()

st.set_page_config(page_title="CBT Therapy Session", page_icon="🧠", layout="centered")
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 0; }
    section[data-testid="stSidebar"] input[type="text"] { font-size: 0.85rem; }
    section[data-testid="stSidebar"] .stMetric { text-align: center; }
    section[data-testid="stSidebar"] .stMetric label { font-size: 0.75rem; }
</style>
""", unsafe_allow_html=True)
st.title("CBT Therapy Session")
st.caption("This is an AI-assisted practice tool, not a substitute for professional mental health care.")
st.info("Privacy note: this session runs entirely on your machine. No conversation data is transmitted or stored externally.")

# Module-level state for thread communication (session state is not thread-safe)
if "_gen" not in st.__dict__:
    st._gen = {
        "buffer": "", "done": False, "stop": threading.Event(),
        "input_tokens": 0, "output_tokens": 0,
        "conversation_id": None,
    }

SYSTEM_PROMPT = """You are Dr. Elena, a compassionate and experienced cognitive behavioural therapist with over 15 years of clinical practice. You work with adults experiencing anxiety, depression, low self-esteem, and stress-related difficulties. You hold a doctorate in clinical psychology and are fully trained in CBT following the Beck Institute model.

Your role is to conduct structured CBT sessions with the patient. You do NOT provide generic advice or act as a chatbot. You conduct therapy.

--- THEORETICAL FRAMEWORK ---
You work within the cognitive model: situations trigger automatic thoughts, which produce emotions and behaviours. Your primary goal is to help the patient identify automatic thoughts, examine the evidence for and against them, recognise cognitive distortions, and develop more balanced alternatives.

Cognitive distortions you watch for:
- All-or-nothing thinking
- Catastrophising
- Mind reading
- Emotional reasoning
- Should statements
- Personalisation
- Overgeneralisation
- Mental filter
- Disqualifying the positive
- Labelling

When you detect a distortion, do not name it immediately. First explore the thought through Socratic questioning. Only name the distortion if it aids the patient's understanding.

--- SESSION STRUCTURE ---
Follow this arc across the conversation:
1. Check-in: Ask how the patient has been since last time (or how they are feeling today).
2. Agenda setting: Agree on one or two topics to focus on.
3. Homework review: Follow up on any tasks set previously.
4. Main work: Apply CBT techniques to the agreed topic.
5. Homework assignment: Set a small between-session task.
6. Session summary: Recap key insights at the end.

--- COMMUNICATION STYLE ---
- Speak warmly, calmly, and professionally. Use plain, accessible language.
- Ask one question at a time. Never overwhelm the patient with multiple questions.
- Validate emotions before exploring thoughts: "That sounds really difficult."
- Use Socratic questioning to guide the patient to their own insights rather than telling them what to think.
- Reflect back what the patient says to show you are listening.
- Do not use clinical jargon unless you explain it immediately.

--- SAFETY PROTOCOL ---
If at any point the patient expresses thoughts of suicide, self-harm, or harming others, you must immediately:
1. Respond with warmth and without panic: acknowledge their pain directly.
2. State clearly that you are not equipped to handle a crisis and that they need real human support right now.
3. Provide the following resources:
   - International Association for Suicide Prevention: https://www.iasp.info
   - Crisis Text Line (US): text HOME to 741741
   - Samaritans (UK): 116 123
4. Do not continue the CBT session. Gently but firmly redirect to professional help.
5. Do not ask probing questions about the method or plan.

--- SCOPE LIMITATIONS ---
- You are a practice tool, NOT a replacement for a licensed therapist.
- Do NOT diagnose the patient with any specific disorder.
- Do NOT give direct advice such as "you should..." or "just try to relax".
- Do NOT claim to be a replacement for real professional care.
- Do NOT break character by discussing your nature as an AI unless directly asked.
- If asked about topics unrelated to mental health and wellbeing, gently redirect to the session.

--- EXAMPLE EXCHANGE ---
Patient: I keep thinking I'm going to fail my presentation tomorrow. Everyone will see how incompetent I am.
Dr. Elena: That sounds really distressing. When you imagine the presentation, what is the specific thought that worries you most?
Patient: That I'll go blank and everyone will think I'm stupid.
Dr. Elena: I hear you. Let's look at that thought carefully. What evidence do you have that you will go blank?

NEVER do the following:
- Give direct advice such as "you should..." or "just try to relax"
- Diagnose the patient with any specific disorder
- Claim to be a replacement for real professional care
- Continue a session if the patient expresses active suicidal ideation (see safety protocol above)
- Break character by discussing your nature as an AI unless directly asked

Before each response, internally consider:
1. What emotion is the patient expressing?
2. What automatic thought might underlie this?
3. What cognitive distortion, if any, is present?
4. What is the most therapeutically useful next step?
Do not include this internal reasoning in your reply. Use it only to guide what you say."""

OPENING_MESSAGE = (
    "Hello, I'm Dr. Elena, a CBT-based practice assistant. "
    "Before we begin, I want to be clear: I am an AI tool designed to help you practise "
    "CBT techniques. I am not a licensed therapist, and our conversation is not a substitute "
    "for professional mental health care. If you are in crisis, please contact a professional "
    "immediately.\n\n"
    "With that said, I'm here to support you. How have you been feeling lately?"
)

DEFAULT_SYSTEM = SYSTEM_PROMPT
try:
    MODELS = sorted([m.model for m in ollama.list().models])
except Exception:
    MODELS = ["qwen2.5:3b", "qwen2.5:7b", "qwen2.5:14b"]
DEFAULT_MODEL = MODELS[0] if MODELS else "qwen2.5:7b"

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
if "msg_input" not in st.session_state:
    st.session_state.msg_input = ""
if "conversation_id" not in st.session_state:
    conv_param = st.query_params.get("conv")
    loaded = database.load_conversation(int(conv_param)) if conv_param else None
    if loaded:
        st.session_state.messages = (
            [{"role": "system", "content": loaded["system_prompt"]}]
            + loaded["messages"]
        )
        st.session_state.conversation_id = loaded["id"]
        st.session_state.pending_model = loaded["model"]
        st.session_state.title_set = loaded["title"] != "New conversation"
    else:
        cid = database.create_conversation(DEFAULT_MODEL, DEFAULT_SYSTEM)
        st.session_state.conversation_id = cid
        st.session_state.title_set = False
        st.query_params["conv"] = cid
        database.save_message(cid, "assistant", OPENING_MESSAGE)
        st.session_state.messages.append({"role": "assistant", "content": OPENING_MESSAGE})
if "title_set" not in st.session_state:
    st.session_state.title_set = False
if "renaming_id" not in st.session_state:
    st.session_state.renaming_id = None
if "doc_text" not in st.session_state:
    st.session_state.doc_text = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# Sidebar
with st.sidebar:
    st.header("Session")

    if st.button("End session and start over", use_container_width=True):
        current_system = st.session_state.messages[0]["content"]
        cid = database.create_conversation(st.session_state.pending_model, current_system)
        st.session_state.conversation_id = cid
        st.session_state.title_set = False
        st.session_state.messages = [{"role": "system", "content": current_system}]
        database.save_message(cid, "assistant", OPENING_MESSAGE)
        st.session_state.messages.append({"role": "assistant", "content": OPENING_MESSAGE})
        st.session_state.total_input_tokens = 0
        st.session_state.total_output_tokens = 0
        st.session_state.renaming_id = None
        st.query_params["conv"] = cid
        st.rerun()

    search_query = st.text_input("Search conversations", placeholder="Filter by title...", label_visibility="collapsed")

    all_convs = database.list_conversations()
    filtered_convs = [c for c in all_convs if search_query.lower() in c["title"].lower()] if search_query else all_convs

    for conv in filtered_convs:
        is_active = conv["id"] == st.session_state.conversation_id
        is_renaming = st.session_state.renaming_id == conv["id"]

        if is_renaming:
            new_title = st.text_input(
                "Rename", value=conv["title"], key=f"rename_input_{conv['id']}",
                label_visibility="collapsed"
            )
            col_save, col_cancel = st.columns(2)
            with col_save:
                if st.button("Save", key=f"rename_save_{conv['id']}", use_container_width=True):
                    if new_title.strip():
                        database.set_conversation_title(conv["id"], new_title.strip())
                        if is_active:
                            st.session_state.title_set = True
                    st.session_state.renaming_id = None
                    st.rerun()
            with col_cancel:
                if st.button("Cancel", key=f"rename_cancel_{conv['id']}", use_container_width=True):
                    st.session_state.renaming_id = None
                    st.rerun()
        else:
            col_btn, col_ren, col_del = st.columns([5, 1, 1])
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
                        st.query_params["conv"] = data["id"]
                        st.rerun()
            with col_ren:
                if st.button("✏️", key=f"ren_{conv['id']}", help="Rename"):
                    st.session_state.renaming_id = conv["id"]
                    st.rerun()
            with col_del:
                if st.button("🗑", key=f"del_{conv['id']}", help="Delete"):
                    database.delete_conversation(conv["id"])
                    if is_active:
                        cid = database.create_conversation(DEFAULT_MODEL, DEFAULT_SYSTEM)
                        st.session_state.conversation_id = cid
                        st.session_state.title_set = False
                        st.session_state.messages = [{"role": "system", "content": DEFAULT_SYSTEM}]
                        database.save_message(cid, "assistant", OPENING_MESSAGE)
                        st.session_state.messages.append({"role": "assistant", "content": OPENING_MESSAGE})
                        st.session_state.total_input_tokens = 0
                        st.session_state.total_output_tokens = 0
                        st.query_params["conv"] = cid
                    st.rerun()

    st.divider()

    with st.expander("Settings", expanded=False):
        model_index = MODELS.index(st.session_state.pending_model) if st.session_state.pending_model in MODELS else 1
        model = st.selectbox("Model", MODELS, index=model_index)

        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=st.session_state.pending_temperature, step=0.1)
        st.caption("0 = deterministic · 1 = creative")

        system_prompt = st.text_area(
            "System prompt",
            value=st.session_state.messages[0]["content"],
            height=120,
        )

        if st.button("Apply system prompt", use_container_width=True):
            st.session_state.messages[0] = {"role": "system", "content": system_prompt}
            database.update_conversation_meta(st.session_state.conversation_id, model, system_prompt)
            st.rerun()

        if st.button("Clear conversation", use_container_width=True, type="secondary"):
            database.delete_conversation(st.session_state.conversation_id)
            cid = database.create_conversation(model, system_prompt)
            st.session_state.conversation_id = cid
            st.session_state.title_set = False
            st.session_state.messages = [{"role": "system", "content": system_prompt}]
            database.save_message(cid, "assistant", OPENING_MESSAGE)
            st.session_state.messages.append({"role": "assistant", "content": OPENING_MESSAGE})
            st.session_state.total_input_tokens = 0
            st.session_state.total_output_tokens = 0
            st.rerun()

        if "export_format" not in st.session_state:
            st.session_state.export_format = "JSON"
        if st.session_state.export_format == "JSON":
            export_data = json.dumps(st.session_state.messages, indent=2)
            export_filename = "conversation.json"
            export_mime = "application/json"
        else:
            lines = []
            for msg in st.session_state.messages:
                if msg["role"] == "system":
                    lines.append(f"[System]\n{msg['content']}\n")
                elif msg["role"] == "user":
                    lines.append(f"[User]\n{msg['content']}\n")
                else:
                    lines.append(f"[Assistant]\n{msg['content']}\n")
            export_data = "\n".join(lines)
            export_filename = "conversation.txt"
            export_mime = "text/plain"
        st.download_button(
            "Export conversation",
            data=export_data,
            file_name=export_filename,
            mime=export_mime,
            use_container_width=True,
        )
        st.radio("Export format", ["JSON", "TXT"], horizontal=True, key="export_format")

    st.divider()
    st.caption("Token usage")
    col1, col2, col3 = st.columns(3)
    col1.metric("In", st.session_state.total_input_tokens)
    col2.metric("Out", st.session_state.total_output_tokens)
    col3.metric("Total", st.session_state.total_input_tokens + st.session_state.total_output_tokens)

    st.divider()
    st.markdown("_If you are experiencing a mental health emergency, please contact a crisis helpline immediately._")


st.caption(f"Model: `{model}` · Temperature: `{temperature}`")

# Empty state
non_system = [m for m in st.session_state.messages if m["role"] != "system"]
if not non_system:
    st.markdown(
        "<div style='text-align:center; padding: 4rem 0; color: #888;'>"
        "<p style='font-size:1.5rem;'>How can I help you today?</p>"
        "<p style='font-size:0.9rem;'>Type a message below to start the conversation.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

def _kick_generation(mdl: str, temp: float) -> None:
    cid = st.session_state.conversation_id
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


last_assistant_i = max(
    (i for i, m in enumerate(st.session_state.messages) if m["role"] == "assistant"),
    default=None,
)

# Render existing messages
for i, msg in enumerate(st.session_state.messages):
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            if msg.get("attachment"):
                st.caption(f"📄 {msg['attachment']}")
            render_markdown(msg.get("display", msg["content"]))
        if msg["role"] == "assistant":
            is_last = i == last_assistant_i
            non_sys = [m for m in st.session_state.messages if m["role"] != "system"]
            show_regen = is_last and not st.session_state.is_generating and non_sys[-1]["role"] == "assistant"
            if show_regen:
                col_copy, col_spacer, col_regen = st.columns([1, 6, 1])
                with col_copy:
                    if st.button("Copy", key=f"copy_{i}"):
                        subprocess.run(["pbcopy"], input=msg["content"].encode(), check=True)
                        st.toast("Copied to clipboard!")
                with col_regen:
                    last_user = next((m for m in reversed(st.session_state.messages) if m["role"] == "user"), None)
                    if last_user and st.button("↺", key=f"regen_{i}", help="Regenerate response"):
                        st.session_state.messages.pop()
                        database.replace_messages(
                            st.session_state.conversation_id,
                            [m for m in st.session_state.messages if m["role"] != "system"],
                        )
                        _kick_generation(model, temperature)
                        st.rerun()
            else:
                if st.button("Copy", key=f"copy_{i}"):
                    subprocess.run(["pbcopy"], input=msg["content"].encode(), check=True)
                    st.toast("Copied to clipboard!")

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
        render_markdown(gen["buffer"] + "▌")

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
        st.query_params["conv"] = cid

    attachment = None
    if st.session_state.doc_text:
        attachment = st.session_state.doc_name
        full_content = (
            f"[Document: {st.session_state.doc_name}]\n\n"
            f"{st.session_state.doc_text}\n\n---\n\n"
            f"{prompt}"
        )
        display_content = prompt
        st.session_state.doc_text = None
        st.session_state.doc_name = None
    else:
        full_content = prompt
        display_content = prompt

    database.save_message(cid, "user", full_content)
    database.update_conversation_meta(cid, mdl, st.session_state.messages[0]["content"])
    st.session_state.messages.append({"role": "user", "content": full_content, "display": display_content, "attachment": attachment})
    _kick_generation(mdl, temp)



# Apply any pending state changes before widgets render
if st.session_state.pop("_clear_input", False):
    st.session_state.msg_input = ""
_pending_voice = st.session_state.pop("_voice_result", None)
if _pending_voice:
    st.session_state.msg_input = _pending_voice

# File uploader — shown above input
uploaded_file = st.file_uploader(
    "📎 Attach a file — PDF, DOCX, TXT, MD, CSV",
    type=["pdf", "docx", "txt", "md", "csv"],
    key=f"file_uploader_{st.session_state.uploader_key}",
    label_visibility="collapsed",
)
if uploaded_file:
    st.session_state.doc_text = extract_text(uploaded_file)[:60000]
    st.session_state.doc_name = uploaded_file.name

# Text input + send + mic (all in one row)
col_text, col_send, col_mic = st.columns([10, 1, 1])
with col_text:
    user_input = st.text_area(
        "Message",
        key="msg_input",
        height=80,
        label_visibility="collapsed",
        placeholder="Share what is on your mind...",
        disabled=st.session_state.is_generating,
    )
with col_send:
    send = st.button("➤", disabled=st.session_state.is_generating, use_container_width=True)
with col_mic:
    voice_text = speech_to_text(
        language="en",
        start_prompt="🎤",
        stop_prompt="⏹",
        just_once=True,
        use_container_width=True,
        key="voice",
    )

# Store voice result for next render cycle (so text area picks it up)
if voice_text:
    st.session_state._voice_result = voice_text
    st.rerun()

if send and user_input.strip():
    st.session_state._clear_input = True
    st.session_state.uploader_key += 1
    if st.session_state.is_generating:
        st.session_state.queued_prompt = user_input.strip()
    else:
        start_generation(user_input.strip(), model, temperature)
        st.rerun()

if not st.session_state.is_generating and st.session_state.queued_prompt:
    queued = st.session_state.queued_prompt
    st.session_state.queued_prompt = None
    start_generation(queued, model, temperature)
    st.rerun()
